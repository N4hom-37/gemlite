use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock, RwLock};
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
    #[serde(default)]
    details: Vec<GeminiErrorDetailEntry>,
}

#[derive(Deserialize)]
struct GeminiErrorDetailEntry {
    #[serde(default)]
    reason: String,
}

// Unified JSON response schema returned across the FFI boundary.

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
    result: Option<String>,  // Some(text) on success; omitted entirely on error
    errors: Vec<ErrorDetail>,
}

/// Classifies a failure for the api-key retry loop:
/// - `Local`: failed before reaching the server (DNS, connect timeout, etc) -- rotating keys
///   won't help, abort the whole retry loop.
/// - `RotateKey`: server said this key is out of quota (429) or invalid (400/403 with
///   `API_KEY_INVALID`) -- worth trying the next configured key.
/// - `Terminal`: any other server-side failure -- not a key problem, surface it immediately.
enum AskFailure {
    Local(ErrorDetail),
    RotateKey(ErrorDetail),
    Terminal(ErrorDetail),
}

impl AskFailure {
    fn should_rotate(&self) -> bool {
        matches!(self, AskFailure::RotateKey(_))
    }

    fn into_detail(self) -> ErrorDetail {
        match self {
            AskFailure::Local(d) | AskFailure::RotateKey(d) | AskFailure::Terminal(d) => d,
        }
    }
}

// --- Streaming ---
//
// The FFI boundary hands Python one value at a time, so a stream is a background thread plus
// a handle Python polls: StreamStart spawns the SSE-reading thread and returns a u64 handle;
// StreamNext polls for the next chunk (delta / done / error / pending); StreamClose cancels
// and releases the handle.
//
// Only 429/invalid-key failures rotate to the next key, same as the non-streaming path, with
// one extra rule: rotation is only allowed *before* the first text delta has reached the
// caller. Once real content has been relayed, a later failure is always terminal -- retrying
// then could duplicate or garble output.

/// One message sent from the background stream thread back to `StreamNext`.
enum StreamMsg {
    Delta(String),
    Done { model: String },
    Error { model: String, errors: Vec<ErrorDetail> },
}

fn stream_msg_to_json(msg: &StreamMsg) -> String {
    let value = match msg {
        StreamMsg::Delta(text) => serde_json::json!({"type": "delta", "text": text}),
        StreamMsg::Done { model } => serde_json::json!({"type": "done", "model": model}),
        StreamMsg::Error { model, errors } => {
            serde_json::json!({"type": "error", "model": model, "errors": errors})
        }
    };
    value.to_string()
}

struct StreamEntry {
    receiver: Mutex<mpsc::Receiver<StreamMsg>>,
    cancel: Arc<AtomicBool>,
}

fn stream_registry() -> &'static RwLock<HashMap<u64, StreamEntry>> {
    static R: OnceLock<RwLock<HashMap<u64, StreamEntry>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(HashMap::new()))
}

fn next_stream_id() -> u64 {
    static COUNTER: OnceLock<AtomicU64> = OnceLock::new();
    COUNTER.get_or_init(|| AtomicU64::new(1)).fetch_add(1, Ordering::Relaxed)
}

/// Position (within the *valid, rotation-ordered* key list) that the next call should start
/// from. Only advanced when a stream can't retry within its own call: an in-band 429/invalid-
/// key signal arriving after content has already been relayed (see `stream_worker`). Every
/// other rotate-eligible failure retries immediately within the same call and never touches this.
fn rotation_pointer() -> &'static AtomicUsize {
    static P: OnceLock<AtomicUsize> = OnceLock::new();
    P.get_or_init(|| AtomicUsize::new(0))
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

/// Reasons Google returns in `error.details[].reason` meaning the key itself is bad.
fn is_invalid_api_key(details: &[GeminiErrorDetailEntry]) -> bool {
    details.iter().any(|d| d.reason == "API_KEY_INVALID")
}

fn describe_ureq_error(e: ureq::Error, api_key: &str) -> AskFailure {
    let is_local = matches!(&e, ureq::Error::Transport(_));
    let masked_key = mask_api_key(api_key);

    let (mut api_err, rotate) = match e {
        ureq::Error::Status(code, resp) => {
            let is_json = resp.content_type().starts_with("application/json");
            let body_str = resp.into_string().unwrap_or_default();

            // 429 = quota/rate limit exceeded -> always worth rotating.
            let is_quota_limit = code == 429;

            if is_json {
                if let Ok(json_err) = serde_json::from_str::<GeminiErrorPayload>(&body_str) {
                    let is_bad_key = (code == 400 || code == 403)
                        && is_invalid_api_key(&json_err.error.details);
                    let rotate = is_quota_limit || is_bad_key;
                    (
                        ErrorDetail {
                            api_key: masked_key,
                            code,
                            status: json_err.error.status,
                            message: json_err.error.message,
                        },
                        rotate,
                    )
                } else {
                    (
                        ErrorDetail {
                            api_key: masked_key,
                            code,
                            status: "MALFORMED_JSON_RESPONSE".to_string(),
                            message: body_str.chars().take(MAX_ERR_BODY_CHARS).collect(),
                        },
                        is_quota_limit,
                    )
                }
            } else {
                (
                    ErrorDetail {
                        api_key: masked_key,
                        code,
                        status: "HTTP_GATEWAY_ERROR".to_string(),
                        message: body_str.chars().take(MAX_ERR_BODY_CHARS).collect(),
                    },
                    is_quota_limit,
                )
            }
        }
        ureq::Error::Transport(t) => (
            ErrorDetail {
                api_key: masked_key,
                code: 0,
                status: "TRANSPORT_FAILURE".to_string(),
                message: t.to_string(),
            },
            false,
        ),
    };

    api_err.message = redact(api_err.message, api_key);
    if is_local {
        AskFailure::Local(api_err)
    } else if rotate {
        AskFailure::RotateKey(api_err)
    } else {
        AskFailure::Terminal(api_err)
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
        .ok_or_else(|| AskFailure::Terminal(make_local_api_error(api_key, "MISSING_HEADER", "Upload start response missing X-Goog-Upload-URL")))?;

    let put = a
        .put(upload_url)
        .set("X-Goog-Upload-Command", "upload, finalize")
        .set("X-Goog-Upload-Offset", "0")
        .set("Content-Length", &data.len().to_string())
        .send_bytes(data)
        .map_err(|e| describe_ureq_error(e, api_key))?;

    let parsed: UploadResp = put
        .into_json()
        .map_err(|e| AskFailure::Terminal(make_local_api_error(api_key, "BAD_UPLOAD_JSON", &format!("{e}"))))?;

    if parsed.file.uri.is_empty() {
        return Err(AskFailure::Terminal(make_local_api_error(api_key, "EMPTY_FILE_URI", "Upload response had empty file uri")));
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

/// Uploads (or reuses cached uploads for) every file against one key. Stops at the first
/// failure so the caller can decide whether to rotate keys or abort.
fn upload_all_files(cfg: &CfgEntry, read_files: &[ReadFile]) -> Result<Vec<(String, String)>, AskFailure> {
    let mut uploaded = Vec::with_capacity(read_files.len());
    for rf in read_files {
        let uri = get_or_upload_file(&cfg.api_key, &rf.path, &rf.data, &rf.mime, rf.size, rf.modified)?;
        uploaded.push((uri, rf.mime.clone()));
    }
    Ok(uploaded)
}

fn build_payload(cfg: &CfgEntry, turns: &[Turn], files: &[(String, String)]) -> serde_json::Value {
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
    payload
}

/// Preflight result shared by the blocking and streaming code paths: files already read off
/// disk, a snapshot of the configured keys, and a rotation-ordered list of positions into the
/// *valid* subset of that snapshot (`valid[pos]` is the real index into `cfg_snapshot`).
struct Preflight {
    read_files: Vec<ReadFile>,
    cfg_snapshot: Arc<Vec<CfgEntry>>,
    valid: Vec<usize>,
    order: Vec<usize>,
}

impl Preflight {
    fn cfg_at(&self, pos: usize) -> &CfgEntry {
        &self.cfg_snapshot[self.valid[pos]]
    }
}

struct PreflightError {
    status: &'static str,
    message: String,
}

impl PreflightError {
    fn new(status: &'static str, message: String) -> Self {
        Self { status, message }
    }
}

/// Shared setup for `Ask` and `StreamStart`: parses/validates the file list, enforces size
/// limits, reads every file off disk, snapshots the configured keys, filters to valid entries,
/// and orders them starting from wherever `rotation_pointer` last left off.
fn run_preflight(fp: &str) -> Result<Preflight, PreflightError> {
    let paths = parse_file_paths(fp).map_err(|e| PreflightError::new("BAD_FILE_PATH_JSON", e))?;

    let mut total_bytes: u64 = 0;
    for path in &paths {
        let m = std::fs::metadata(path)
            .map_err(|e| PreflightError::new("FILE_READ_ERROR", format!("cannot stat '{path}': {e}")))?;
        total_bytes += m.len();
    }
    if total_bytes > MAX_TOTAL_FILE_BYTES {
        return Err(PreflightError::new(
            "TOTAL_FILE_SIZE_EXCEEDED",
            format!("combined size of all files ({total_bytes} bytes) exceeds max total upload size of {MAX_TOTAL_FILE_BYTES} bytes"),
        ));
    }

    let mut read_files: Vec<ReadFile> = Vec::with_capacity(paths.len());
    for path in &paths {
        let (data, mime, size, modified) =
            read_file(path).map_err(|e| PreflightError::new("FILE_READ_ERROR", e))?;
        read_files.push(ReadFile { path: path.clone(), data, mime, size, modified });
    }

    let cfg_snapshot: Arc<Vec<CfgEntry>> = configs()
        .read()
        .map_err(|_| PreflightError::new("LOCK_POISONED", "configuration lock is poisoned".to_string()))?
        .clone();

    let valid: Vec<usize> = cfg_snapshot
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.api_key.is_empty() && !c.model.is_empty())
        .map(|(i, _)| i)
        .collect();

    if valid.is_empty() {
        return Err(PreflightError::new("NO_VALID_CONFIG", "no valid configuration found".to_string()));
    }
    if !network_available() {
        return Err(PreflightError::new(
            "NETWORK_UNAVAILABLE",
            "no network connection, aborting before trying api keys".to_string(),
        ));
    }

    let n = valid.len();
    let start = rotation_pointer().load(Ordering::Relaxed) % n;
    let order: Vec<usize> = (0..n).map(|i| (start + i) % n).collect();

    Ok(Preflight { read_files, cfg_snapshot, valid, order })
}

/// Opens a `streamGenerateContent` (SSE) request. Failing here means no response bytes were
/// read yet, so the normal `describe_ureq_error` rotate/terminal classification still applies.
fn open_stream(cfg: &CfgEntry, turns: &[Turn], files: &[(String, String)]) -> Result<Box<dyn Read + Send + 'static>, AskFailure> {
    let payload = build_payload(cfg, turns, files);
    let model = cfg.model.strip_prefix("models/").unwrap_or(&cfg.model);
    let resp = agent()
        .post(&format!("{BASE_URL}/v1beta/models/{model}:streamGenerateContent"))
        .query("key", &cfg.api_key)
        .query("alt", "sse")
        .send_json(payload)
        .map_err(|e| describe_ureq_error(e, &cfg.api_key))?;
    Ok(Box::new(resp.into_reader()))
}

/// What one `data: {...}` SSE line turned out to be.
enum SseLineKind {
    Content(GenResponse),
    /// Google sometimes signals a mid-stream failure as an in-band `{"error": {...}}` object
    /// (the HTTP status already committed to 200).
    ErrorChunk(GeminiErrorDetails),
    Malformed(String),
}

fn parse_sse_data(data: &str) -> SseLineKind {
    let value = match serde_json::from_str::<serde_json::Value>(data) {
        Ok(v) => v,
        Err(_) => return SseLineKind::Malformed("line was not valid JSON".to_string()),
    };
    if value.get("error").is_some() {
        return match serde_json::from_value::<GeminiErrorPayload>(value) {
            Ok(payload) => SseLineKind::ErrorChunk(payload.error),
            Err(e) => SseLineKind::Malformed(format!("unparseable in-band error chunk: {e}")),
        };
    }
    match serde_json::from_value::<GenResponse>(value) {
        Ok(gr) => SseLineKind::Content(gr),
        Err(e) => SseLineKind::Malformed(format!("malformed SSE chunk: {e}")),
    }
}

/// How a stream ended.
enum SseOutcome {
    /// Clean EOF, or the caller cancelled/dropped the receiver. Caller checks `any_delta` to
    /// tell success from no-content-at-all.
    Ended,
    /// In-band error that is itself 429 or an invalid-key signal.
    RotateEligible(ErrorDetail),
    /// Any other failure (non-key in-band error, malformed chunk, I/O error) -- never rotate-eligible.
    Terminal(ErrorDetail),
}

/// Reads SSE lines off `reader`, forwarding each non-empty text delta through `tx`. Sets
/// `*any_delta = true` on the first delta -- callers use this to decide whether a
/// `RotateEligible` failure can still retry within this call, or must finish gracefully and
/// defer rotation to the next call.
fn pump_sse(reader: Box<dyn Read + Send>, tx: &mpsc::Sender<StreamMsg>, cancel: &AtomicBool, any_delta: &mut bool, api_key: &str) -> SseOutcome {
    let mut buf = BufReader::new(reader);
    let mut line = String::new();

    loop {
        if cancel.load(Ordering::Relaxed) {
            return SseOutcome::Ended;
        }
        line.clear();
        let n = match buf.read_line(&mut line) {
            Ok(n) => n,
            Err(e) => return SseOutcome::Terminal(make_local_api_error(api_key, "STREAM_READ_ERROR", &e.to_string())),
        };
        if n == 0 {
            return SseOutcome::Ended; // clean EOF
        }

        let trimmed = line.trim_end();
        let data = match trimmed.strip_prefix("data: ") {
            Some(d) if !d.is_empty() => d,
            _ => continue,
        };

        match parse_sse_data(data) {
            SseLineKind::Content(parsed) => {
                if let Some(candidate) = parsed.candidates.into_iter().next() {
                    let text: String = candidate.content.parts.into_iter().map(|p| p.text).collect();
                    if !text.is_empty() {
                        *any_delta = true;
                        if tx.send(StreamMsg::Delta(text)).is_err() {
                            return SseOutcome::Ended; // receiver gone -- caller cancelled/closed
                        }
                    }
                }
            }
            SseLineKind::ErrorChunk(details) => {
                let rotate_eligible = details.code == 429 || ((details.code == 400 || details.code == 403) && is_invalid_api_key(&details.details));
                let detail = ErrorDetail {
                    api_key: mask_api_key(api_key),
                    code: details.code,
                    status: details.status,
                    message: redact(details.message, api_key),
                };
                return if rotate_eligible {
                    SseOutcome::RotateEligible(detail)
                } else {
                    SseOutcome::Terminal(detail)
                };
            }
            SseLineKind::Malformed(msg) => {
                return SseOutcome::Terminal(make_local_api_error(api_key, "MALFORMED_SSE_CHUNK", &msg));
            }
        }
    }
}

fn generate(cfg: &CfgEntry, turns: &[Turn], files: &[(String, String)]) -> Result<String, AskFailure> {
    let payload = build_payload(cfg, turns, files);
    let model = cfg.model.strip_prefix("models/").unwrap_or(&cfg.model);
    let resp = agent()
        .post(&format!("{BASE_URL}/v1beta/models/{model}:generateContent"))
        .query("key", &cfg.api_key)
        .send_json(payload)
        .map_err(|e| describe_ureq_error(e, &cfg.api_key))?;

    let parsed: GenResponse = resp
        .into_json()
        .map_err(|e| AskFailure::Terminal(make_local_api_error(&cfg.api_key, "BAD_GENERATE_JSON", &format!("{e}"))))?;

    let candidate = match parsed.candidates.into_iter().next() {
        Some(c) => c,
        None => {
            let reason = parsed
                .prompt_feedback
                .map(|f| f.block_reason)
                .filter(|r| !r.is_empty());
            return Err(match reason {
                Some(r) => AskFailure::Terminal(make_local_api_error(&cfg.api_key, "PROMPT_BLOCKED", &format!("prompt blocked ({r})"))),
                None => AskFailure::Terminal(make_local_api_error(&cfg.api_key, "NO_CANDIDATES", "no candidates returned")),
            });
        }
    };

    let text: String = candidate.content.parts.into_iter().map(|p| p.text).collect();
    if text.is_empty() {
        return Err(AskFailure::Terminal(make_local_api_error(
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
    let turns = parse_turns(&cstr_to_string(prompt));
    let fp = cstr_to_string(file_path);

    let pre = match run_preflight(&fp) {
        Ok(p) => p,
        Err(e) => return make_preflight(e.status, &e.message),
    };

    let mut attempts_log: Vec<ErrorDetail> = Vec::new();
    let mut last_model = String::new();

    for &pos in &pre.order {
        let cfg = pre.cfg_at(pos);
        last_model = cfg.model.clone();

        // Only move to the next key if this one is out of quota or invalid; any other
        // failure (network down, malformed response, etc) isn't something a different key
        // would fix, so surface it immediately instead of burning through the rest.
        let uploaded = match upload_all_files(cfg, &pre.read_files) {
            Ok(u) => u,
            Err(failure) => {
                let rotate = failure.should_rotate();
                attempts_log.push(failure.into_detail());
                if rotate {
                    continue;
                }
                break;
            }
        };

        match generate(cfg, &turns, &uploaded) {
            Ok(text) => {
                let payload = FinalResponse {
                    status: "success".to_string(),
                    model: last_model,
                    result: Some(text),
                    errors: Vec::new(),
                };
                return serde_json::to_string(&payload).unwrap_or_default();
            }
            Err(failure) => {
                let rotate = failure.should_rotate();
                attempts_log.push(failure.into_detail());
                if !rotate {
                    break;
                }
            }
        }
    }

    let final_payload = FinalResponse {
        status: "error".to_string(),
        model: if last_model.is_empty() { "unknown".to_string() } else { last_model },
        result: None,
        errors: attempts_log,
    };

    serde_json::to_string(&final_payload)
        .unwrap_or_else(|_| make_preflight("JSON_SERIALIZATION_FAIL", "Failed to construct the unified JSON payload"))
}

fn stream_send_error(tx: &mpsc::Sender<StreamMsg>, status: &str, message: String) {
    let _ = tx.send(StreamMsg::Error {
        model: "unknown".to_string(),
        errors: vec![ErrorDetail { api_key: String::new(), code: 0, status: status.to_string(), message }],
    });
}

/// Runs on a dedicated background thread per stream. Mirrors `ask_inner`'s preflight + per-key
/// retry loop, but reports progress incrementally over `tx` instead of building one JSON string.
fn stream_worker(fp: String, turns: Vec<Turn>, tx: mpsc::Sender<StreamMsg>, cancel: Arc<AtomicBool>) {
    let pre = match run_preflight(&fp) {
        Ok(p) => p,
        Err(e) => return stream_send_error(&tx, e.status, e.message),
    };

    let mut attempts_log: Vec<ErrorDetail> = Vec::new();
    let mut last_model = String::new();
    let n = pre.order.len();

    for &pos in &pre.order {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let cfg = pre.cfg_at(pos);
        last_model = cfg.model.clone();

        let uploaded = match upload_all_files(cfg, &pre.read_files) {
            Ok(u) => u,
            Err(failure) => {
                let rotate = failure.should_rotate();
                attempts_log.push(failure.into_detail());
                if rotate {
                    continue;
                }
                let _ = tx.send(StreamMsg::Error { model: last_model, errors: attempts_log });
                return;
            }
        };

        match open_stream(cfg, &turns, &uploaded) {
            Ok(reader) => {
                let mut any_delta = false;
                match pump_sse(reader, &tx, &cancel, &mut any_delta, &cfg.api_key) {
                    SseOutcome::Ended => {
                        if any_delta {
                            let _ = tx.send(StreamMsg::Done { model: last_model });
                        } else {
                            // Stream ended with no content (e.g. a blocked prompt) -- not a
                            // key problem, terminal just like NO_CANDIDATES above.
                            attempts_log.push(ErrorDetail {
                                api_key: mask_api_key(&cfg.api_key),
                                code: 0,
                                status: "NO_CANDIDATES".to_string(),
                                message: "stream ended with no content".to_string(),
                            });
                            let _ = tx.send(StreamMsg::Error { model: last_model, errors: attempts_log });
                        }
                        return;
                    }
                    SseOutcome::RotateEligible(detail) => {
                        if any_delta {
                            // Content already reached the caller -- finish gracefully as if
                            // the model simply stopped, and defer key rotation to the next call.
                            rotation_pointer().store((pos + 1) % n, Ordering::Relaxed);
                            let _ = tx.send(StreamMsg::Done { model: last_model });
                        } else {
                            // Nothing sent yet -- safe to rotate within this same call.
                            attempts_log.push(detail);
                            continue;
                        }
                        return;
                    }
                    SseOutcome::Terminal(detail) => {
                        attempts_log.push(detail);
                        let _ = tx.send(StreamMsg::Error { model: last_model, errors: attempts_log });
                        return;
                    }
                }
            }
            Err(failure) => {
                let rotate = failure.should_rotate();
                attempts_log.push(failure.into_detail());
                if rotate {
                    continue;
                }
                let _ = tx.send(StreamMsg::Error { model: last_model, errors: attempts_log });
                return;
            }
        }
    }

    // All configured keys were exhausted (each rotate-eligible) without opening a stream.
    let _ = tx.send(StreamMsg::Error {
        model: if last_model.is_empty() { "unknown".to_string() } else { last_model },
        errors: attempts_log,
    });
}

#[no_mangle]
pub extern "C" fn StreamStart(prompt: *const c_char, file_path: *const c_char) -> u64 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let q = cstr_to_string(prompt);
        let fp = cstr_to_string(file_path);
        let turns = parse_turns(&q);

        let (tx, rx) = mpsc::channel::<StreamMsg>();
        let cancel = Arc::new(AtomicBool::new(false));
        let id = next_stream_id();

        let inserted = stream_registry().write().map(|mut g| {
            g.insert(
                id,
                StreamEntry {
                    receiver: Mutex::new(rx),
                    cancel: cancel.clone(),
                },
            );
        });
        if inserted.is_err() {
            return 0;
        }

        std::thread::spawn(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                stream_worker(fp, turns, tx, cancel);
            }));
        });

        id
    }))
    .unwrap_or(0)
}

fn stream_next_inner(handle: u64) -> String {
    const POLL_INTERVAL: Duration = Duration::from_millis(200);

    let unknown_handle = || {
        stream_msg_to_json(&StreamMsg::Error {
            model: "unknown".to_string(),
            errors: vec![ErrorDetail {
                api_key: String::new(),
                code: 0,
                status: "UNKNOWN_HANDLE".to_string(),
                message: "stream handle not found (already closed?)".to_string(),
            }],
        })
    };
    let lock_poisoned = || {
        stream_msg_to_json(&StreamMsg::Error {
            model: "unknown".to_string(),
            errors: vec![ErrorDetail {
                api_key: String::new(),
                code: 0,
                status: "LOCK_POISONED".to_string(),
                message: "stream registry lock is poisoned".to_string(),
            }],
        })
    };

    let guard = match stream_registry().read() {
        Ok(g) => g,
        Err(_) => return lock_poisoned(),
    };
    let entry = match guard.get(&handle) {
        Some(e) => e,
        None => return unknown_handle(),
    };
    let rx_guard = match entry.receiver.lock() {
        Ok(g) => g,
        Err(_) => return lock_poisoned(),
    };

    match rx_guard.recv_timeout(POLL_INTERVAL) {
        Ok(msg) => stream_msg_to_json(&msg),
        // Nothing new yet -- hand control back to Python so it can check for cancellation
        // between calls instead of blocking here indefinitely.
        Err(mpsc::RecvTimeoutError::Timeout) => serde_json::json!({"type": "pending"}).to_string(),
        // Worker dropped its sender without an explicit Done/Error -- fail closed rather than hang.
        Err(mpsc::RecvTimeoutError::Disconnected) => stream_msg_to_json(&StreamMsg::Done { model: "unknown".to_string() }),
    }
}

#[no_mangle]
pub extern "C" fn StreamNext(handle: u64) -> *mut c_char {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stream_next_inner(handle))).unwrap_or_else(|_| {
        stream_msg_to_json(&StreamMsg::Error {
            model: "unknown".to_string(),
            errors: vec![ErrorDetail {
                api_key: String::new(),
                code: 0,
                status: "FATAL_PANIC".to_string(),
                message: "Panic inside FFI boundary".to_string(),
            }],
        })
    });

    CString::new(result)
        .unwrap_or_else(|_| {
            CString::new(stream_msg_to_json(&StreamMsg::Error {
                model: "unknown".to_string(),
                errors: vec![ErrorDetail {
                    api_key: String::new(),
                    code: 0,
                    status: "NULL_BYTE_ERROR".to_string(),
                    message: "Null byte error".to_string(),
                }],
            }))
            .unwrap()
        })
        .into_raw()
}

/// Cancels and releases a stream handle. Safe to call more than once, and safe even if the
/// background thread is mid-read: dropping the receiver here makes its next send fail and it
/// exits on its own. StreamNext on this handle returns UNKNOWN_HANDLE immediately after.
#[no_mangle]
pub extern "C" fn StreamClose(handle: u64) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if let Ok(mut guard) = stream_registry().write() {
            if let Some(entry) = guard.remove(&handle) {
                entry.cancel.store(true, Ordering::Relaxed);
            }
        }
    }));
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
