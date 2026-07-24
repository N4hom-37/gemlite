use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant, UNIX_EPOCH};

const BASE_URL: &str = "https://generativelanguage.googleapis.com";
const FILE_TTL: Duration = Duration::from_secs(47 * 3600);
const MAX_FILE_BYTES: u64 = 200 * 1024 * 1024;
const MAX_TOTAL_FILE_BYTES: u64 = 500 * 1024 * 1024;
const MAX_ERR_BODY_CHARS: usize = 1000;

#[derive(Deserialize)]
struct CfgEntry {
    #[serde(default)]
    api_key: String,
    #[serde(default)]
    system_prompt: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    enable_grounding: bool,
}

#[derive(Deserialize)]
struct GenResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(rename = "promptFeedback")]
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Deserialize)]
struct PromptFeedback {
    #[serde(rename = "blockReason", default)]
    block_reason: String,
}

#[derive(Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Content,
    #[serde(rename = "finishReason", default)]
    finish_reason: String,
}

#[derive(Deserialize, Default)]
struct Content {
    #[serde(default)]
    parts: Vec<Part>,
}

#[derive(Deserialize)]
struct Part {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct UploadResp {
    file: UploadFile,
}

#[derive(Deserialize)]
struct UploadFile {
    #[serde(default)]
    uri: String,
}

struct CachedFile {
    uri: String,
    at: Instant,
}

type FileCacheKey = (String, String, u64, u64);

#[derive(Deserialize)]
struct Turn {
    role: String,
    text: String,
}

struct ReadFile {
    path: String,
    data: Vec<u8>,
    mime: String,
    size: u64,
    modified: u64,
}

#[derive(Deserialize)]
struct GeminiErrorPayload {
    error: GeminiErrorDetails,
}

#[derive(Deserialize)]
struct GeminiErrorDetails {
    code: u16,
    message: String,
    status: String,
}

// ---------------------------------------------------------
// UNIFIED JSON SCHEMA STRUCTS
// ---------------------------------------------------------

#[derive(Serialize)]
struct ErrorDetail {
    api_key: String,
    code: u16,
    status: String,
    message: String,
}

#[derive(Serialize)]
struct FinalResponse {
    status: String,          // "success" or "error"
    model: String,           // always present, never empty
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<String>,  // Some(text) on success; field is omitted entirely on error
    errors: Vec<ErrorDetail>,
}

// ---------------------------------------------------------

/// Distinguishes a failure that happened before we ever reached the server
/// (DNS, connection refused, dropped connection, connect timeout) from one
/// where the server actually responded (even with an error status, or a
/// malformed/unexpected body). Only `Local` failures should abort the
/// api-key retry loop early; `Remote` failures mean it's worth trying the
/// next configured key.
enum AskFailure {
    Local(ErrorDetail),
    Remote(ErrorDetail),
}

impl AskFailure {
    fn is_local(&self) -> bool {
        matches!(self, AskFailure::Local(_))
    }

    fn into_detail(self) -> ErrorDetail {
        match self {
            AskFailure::Local(d) | AskFailure::Remote(d) => d,
        }
    }
}

fn configs() -> &'static RwLock<Arc<Vec<CfgEntry>>> {
    static C: OnceLock<RwLock<Arc<Vec<CfgEntry>>>> = OnceLock::new();
    C.get_or_init(|| RwLock::new(Arc::new(Vec::new())))
}

fn file_cache() -> &'static RwLock<HashMap<FileCacheKey, CachedFile>> {
    static C: OnceLock<RwLock<HashMap<FileCacheKey, CachedFile>>> = OnceLock::new();
    C.get_or_init(|| RwLock::new(HashMap::new()))
}

fn network_available() -> bool {
    use std::net::UdpSocket;
    UdpSocket::bind("0.0.0.0:0").and_then(|s| s.connect("8.8.8.8:53")).is_ok()
}

fn agent() -> &'static ureq::Agent {
    static A: OnceLock<ureq::Agent> = OnceLock::new();
    A.get_or_init(|| {
        ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout(Duration::from_secs(120))
            .build()
    })
}

fn detect_mime(path: &str, data: &[u8]) -> String {
    let guessed = mime_guess::from_path(path).first_or_octet_stream();
    let essence = guessed.essence_str();
    if essence != "application/octet-stream" {
        return essence.to_string();
    }
    infer::get(data)
        .map(|k| k.mime_type().to_string())
        .unwrap_or_else(|| "application/octet-stream".to_string())
}

fn read_file(path: &str) -> Result<(Vec<u8>, String, u64, u64), String> {
    let metadata = std::fs::metadata(path).map_err(|e| format!("cannot stat '{path}': {e}"))?;
    if metadata.len() > MAX_FILE_BYTES {
        return Err(format!("'{path}' exceeds max upload size of {MAX_FILE_BYTES} bytes"));
    }
    let data = std::fs::read(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    if data.is_empty() {
        return Err(format!("'{path}' is empty"));
    }
    let size = data.len() as u64;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mime = detect_mime(path, &data);
    Ok((data, mime, size, modified))
}

fn parse_turns(prompt: &str) -> Vec<Turn> {
    match serde_json::from_str::<Vec<Turn>>(prompt) {
        Ok(turns) if !turns.is_empty() => turns.into_iter().map(sanitize_role).collect(),
        _ => vec![Turn {
            role: "user".into(),
            text: prompt.into(),
        }],
    }
}

fn sanitize_role(mut t: Turn) -> Turn {
    if t.role != "user" && t.role != "model" {
        t.role = "user".into();
    }
    t
}

fn parse_file_paths(fp: &str) -> Result<Vec<String>, String> {
    if fp.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(fp).map_err(|e| format!("bad file_path JSON: {e}"))
}

fn mask_api_key(key: &str) -> String {
    if key.is_empty() {
        return String::new();
    }
    if key.len() > 10 {
        format!("{}...{}", &key[..6], &key[key.len() - 4..])
    } else {
        "...".to_string()
    }
}

fn redact(s: String, key: &str) -> String {
    if key.is_empty() {
        s
    } else {
        s.replace(key, "[REDACTED]")
    }
}

fn make_preflight(status: &str, message: &str) -> String {
    let payload = FinalResponse {
        status: "error".to_string(),
        model: "unknown".to_string(),
        result: None,
        errors: vec![ErrorDetail {
            api_key: String::new(),
            code: 0,
            status: status.to_string(),
            message: message.to_string(),
        }],
    };
    serde_json::to_string(&payload).unwrap_or_default()
}

fn make_local_api_error(api_key: &str, status: &str, message: &str) -> ErrorDetail {
    ErrorDetail {
        api_key: mask_api_key(api_key),
        code: 0,
        status: status.to_string(),
        message: redact(message.to_string(), api_key),
    }
}

fn describe_ureq_error(e: ureq::Error, api_key: &str) -> AskFailure {
    let is_local = matches!(&e, ureq::Error::Transport(_));
    let masked_key = mask_api_key(api_key);

    let mut api_err = match e {
        ureq::Error::Status(code, resp) => {
            let is_json = resp.content_type().starts_with("application/json");
            let body_str = resp.into_string().unwrap_or_default();

            if is_json {
                if let Ok(json_err) = serde_json::from_str::<GeminiErrorPayload>(&body_str) {
                    ErrorDetail {
                        api_key: masked_key,
                        code,
                        status: json_err.error.status,
                        message: json_err.error.message,
                    }
                } else {
                    ErrorDetail {
                        api_key: masked_key,
                        code,
                        status: "MALFORMED_JSON_RESPONSE".to_string(),
                        message: body_str.chars().take(MAX_ERR_BODY_CHARS).collect(),
                    }
                }
            } else {
                ErrorDetail {
                    api_key: masked_key,
                    code,
                    status: "HTTP_GATEWAY_ERROR".to_string(),
                    message: body_str.chars().take(MAX_ERR_BODY_CHARS).collect(),
                }
            }
        }
        ureq::Error::Transport(t) => ErrorDetail {
            api_key: masked_key,
            code: 0,
            status: "TRANSPORT_FAILURE".to_string(),
            message: t.to_string(),
        },
    };

    api_err.message = redact(api_err.message, api_key);
    if is_local {
        AskFailure::Local(api_err)
    } else {
        AskFailure::Remote(api_err)
    }
}

fn upload_file(api_key: &str, data: &[u8], mime: &str) -> Result<String, AskFailure> {
    let a = agent();
    let start = a
        .post(&format!("{BASE_URL}/upload/v1beta/files"))
        .query("key", api_key)
        .set("X-Goog-Upload-Protocol", "resumable")
        .set("X-Goog-Upload-Command", "start")
        .set("X-Goog-Upload-Header-Content-Length", &data.len().to_string())
        .set("X-Goog-Upload-Header-Content-Type", mime)
        .send_json(serde_json::json!({"file": {"displayName": "upload"}}))
        .map_err(|e| describe_ureq_error(e, api_key))?;

    let upload_url = start
        .header("X-Goog-Upload-URL")
        .ok_or_else(|| AskFailure::Remote(make_local_api_error(api_key, "MISSING_HEADER", "Upload start response missing X-Goog-Upload-URL")))?;

    let put = a
        .put(upload_url)
        .set("X-Goog-Upload-Command", "upload, finalize")
        .set("X-Goog-Upload-Offset", "0")
        .set("Content-Length", &data.len().to_string())
        .send_bytes(data)
        .map_err(|e| describe_ureq_error(e, api_key))?;

    let parsed: UploadResp = put
        .into_json()
        .map_err(|e| AskFailure::Remote(make_local_api_error(api_key, "BAD_UPLOAD_JSON", &format!("{e}"))))?;

    if parsed.file.uri.is_empty() {
        return Err(AskFailure::Remote(make_local_api_error(api_key, "EMPTY_FILE_URI", "Upload response had empty file uri")));
    }
    Ok(parsed.file.uri)
}

fn get_or_upload_file(
    api_key: &str,
    file_path: &str,
    data: &[u8],
    mime: &str,
    size: u64,
    modified: u64,
) -> Result<String, AskFailure> {
    let key: FileCacheKey = (api_key.to_string(), file_path.to_string(), size, modified);

    if let Ok(guard) = file_cache().read() {
        if let Some(cached) = guard.get(&key) {
            if cached.at.elapsed() < FILE_TTL {
                return Ok(cached.uri.clone());
            }
        }
    }

    let uri = upload_file(api_key, data, mime)?;

    if let Ok(mut guard) = file_cache().write() {
        guard.retain(|_, v| v.at.elapsed() < FILE_TTL);
        guard.insert(
            key,
            CachedFile {
                uri: uri.clone(),
                at: Instant::now(),
            },
        );
    }

    Ok(uri)
}

fn generate(cfg: &CfgEntry, turns: &[Turn], files: &[(String, String)]) -> Result<String, AskFailure> {
    let last_idx = turns.len().saturating_sub(1);
    let mut contents = Vec::with_capacity(turns.len());

    for (i, turn) in turns.iter().enumerate() {
        let mut parts = vec![serde_json::json!({"text": turn.text})];
        if i == last_idx {
            for (uri, mime) in files {
                parts.push(serde_json::json!({"fileData": {"fileUri": uri, "mimeType": mime}}));
            }
        }
        contents.push(serde_json::json!({"role": turn.role, "parts": parts}));
    }

    let mut payload = serde_json::json!({"contents": contents});
    let obj = payload.as_object_mut().unwrap();
    if cfg.enable_grounding {
        obj.insert("tools".into(), serde_json::json!([{"googleSearch": {}}]));
    }
    if !cfg.system_prompt.is_empty() {
        obj.insert(
            "systemInstruction".into(),
            serde_json::json!({"parts": [{"text": cfg.system_prompt}]}),
        );
    }

    let model = cfg.model.strip_prefix("models/").unwrap_or(&cfg.model);
    let resp = agent()
        .post(&format!("{BASE_URL}/v1beta/models/{model}:generateContent"))
        .query("key", &cfg.api_key)
        .send_json(payload)
        .map_err(|e| describe_ureq_error(e, &cfg.api_key))?;

    let parsed: GenResponse = resp
        .into_json()
        .map_err(|e| AskFailure::Remote(make_local_api_error(&cfg.api_key, "BAD_GENERATE_JSON", &format!("{e}"))))?;

    let candidate = match parsed.candidates.into_iter().next() {
        Some(c) => c,
        None => {
            let reason = parsed
                .prompt_feedback
                .map(|f| f.block_reason)
                .filter(|r| !r.is_empty());
            return Err(match reason {
                Some(r) => AskFailure::Remote(make_local_api_error(&cfg.api_key, "PROMPT_BLOCKED", &format!("prompt blocked ({r})"))),
                None => AskFailure::Remote(make_local_api_error(&cfg.api_key, "NO_CANDIDATES", "no candidates returned")),
            });
        }
    };

    let text: String = candidate.content.parts.into_iter().map(|p| p.text).collect();
    if text.is_empty() {
        return Err(AskFailure::Remote(make_local_api_error(
            &cfg.api_key,
            "EMPTY_RESPONSE",
            &format!("finishReason: {}", candidate.finish_reason),
        )));
    }
    Ok(text)
}

fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

fn ask_inner(prompt: *const c_char, file_path: *const c_char) -> String {
    let q = cstr_to_string(prompt);
    let fp = cstr_to_string(file_path);
    let turns = parse_turns(&q);

    let paths = match parse_file_paths(&fp) {
        Ok(p) => p,
        Err(e) => return make_preflight("BAD_FILE_PATH_JSON", &e),
    };

    let mut total_bytes: u64 = 0;
    for path in &paths {
        match std::fs::metadata(path) {
            Ok(m) => total_bytes += m.len(),
            Err(e) => return make_preflight("FILE_READ_ERROR", &format!("cannot stat '{path}': {e}")),
        }
    }
    if total_bytes > MAX_TOTAL_FILE_BYTES {
        return make_preflight(
            "TOTAL_FILE_SIZE_EXCEEDED",
            &format!("combined size of all files ({total_bytes} bytes) exceeds max total upload size of {MAX_TOTAL_FILE_BYTES} bytes"),
        );
    }

    let mut read_files: Vec<ReadFile> = Vec::with_capacity(paths.len());
    for path in &paths {
        match read_file(path) {
            Ok((data, mime, size, modified)) => read_files.push(ReadFile {
                path: path.clone(),
                data,
                mime,
                size,
                modified,
            }),
            Err(e) => return make_preflight("FILE_READ_ERROR", &e),
        }
    }

    let cfg_snapshot: Arc<Vec<CfgEntry>> = match configs().read() {
        Ok(g) => g.clone(),
        Err(_) => return make_preflight("LOCK_POISONED", "configuration lock is poisoned"),
    };

    let entries: Vec<&CfgEntry> = cfg_snapshot
        .iter()
        .filter(|c| !c.api_key.is_empty() && !c.model.is_empty())
        .collect();

    if entries.is_empty() {
        return make_preflight("NO_VALID_CONFIG", "no valid configuration found");
    }

    if !network_available() {
        return make_preflight("NETWORK_UNAVAILABLE", "no network connection, aborting before trying api keys");
    }

    let mut attempts_log: Vec<ErrorDetail> = Vec::new();
    let mut last_model = String::new();

    for &cfg in &entries {
        last_model = cfg.model.clone();
        let mut uploaded: Vec<(String, String)> = Vec::with_capacity(read_files.len());
        let mut ok = true;
        let mut local_failure = false;

        for rf in &read_files {
            match get_or_upload_file(&cfg.api_key, &rf.path, &rf.data, &rf.mime, rf.size, rf.modified) {
                Ok(uri) => uploaded.push((uri, rf.mime.clone())),
                Err(failure) => {
                    local_failure = failure.is_local();
                    attempts_log.push(failure.into_detail());
                    ok = false;
                    break;
                }
            }
        }

        if !ok {
            // A local/transport connection error means the network itself is
            // the problem, not this particular key - retrying other keys
            // won't help, so stop after this single attempt.
            if local_failure {
                break;
            }
            continue;
        }

        match generate(cfg, &turns, &uploaded) {
            Ok(text) => {
                // Success! Return immediately.
                let payload = FinalResponse {
                    status: "success".to_string(),
                    model: last_model,
                    result: Some(text),
                    errors: Vec::new(),
                };
                return serde_json::to_string(&payload).unwrap_or_default();
            }
            Err(failure) => {
                let is_local = failure.is_local();
                attempts_log.push(failure.into_detail());
                if is_local {
                    break;
                }
            }
        }
    }

    // All configurations failed
    let final_payload = FinalResponse {
        status: "error".to_string(),
        model: if last_model.is_empty() { "unknown".to_string() } else { last_model },
        result: None,
        errors: attempts_log,
    };

    serde_json::to_string(&final_payload).unwrap_or_else(|_| {
        make_preflight("JSON_SERIALIZATION_FAIL", "Failed to construct the unified JSON payload")
    })
}

#[no_mangle]
pub extern "C" fn Init(configs_json: *const c_char) -> bool {
    let s = cstr_to_string(configs_json);
    std::panic::catch_unwind(|| match serde_json::from_str::<Vec<CfgEntry>>(&s) {
        Ok(parsed) => match configs().write() {
            Ok(mut g) => {
                *g = Arc::new(parsed);
                true
            }
            Err(_) => false,
        },
        Err(_) => false,
    })
    .unwrap_or(false)
}

#[no_mangle]
pub extern "C" fn Ask(prompt: *const c_char, file_path: *const c_char) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ask_inner(prompt, file_path)
    }))
    .unwrap_or_else(|_| make_preflight("FATAL_PANIC", "Panic inside FFI boundary"));

    CString::new(result)
        .unwrap_or_else(|_| CString::new(make_preflight("NULL_BYTE_ERROR", "Null byte error")).unwrap())
        .into_raw()
}

#[no_mangle]
pub extern "C" fn FreeString(ptr: *mut c_char) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !ptr.is_null() {
            unsafe { drop(CString::from_raw(ptr)) };
        }
    }));
}
