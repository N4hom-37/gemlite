# gemlite

⚡ A minimal Gemini API client. HTTP calls, file uploads, and multi-key failover live in a small Rust `cdylib`; a thin Python `ctypes` wrapper drives it. Ships as a normal wheel — no Rust toolchain needed to install or use it.

## 📦 Install

```bash
pip install gemlite
```

Prebuilt wheels cover:

| Platform      | Arch          |
|---------------|---------------|
| Linux (glibc) | x86_64, arm64 |
| Linux (musl)  | x86_64, arm64 |
| Android       | armv7         |
| Windows       | x86_64, arm64 |

pip picks the right wheel for your platform automatically. If none matches, build the Rust core yourself (`cargo build --release --target <triple>`) and pass `lib_path=` explicitly to `Gemini(...)`.

## 🐍 Usage

```python
from gemlite import Gemini

ai = Gemini(
    apikey="YOUR_API_KEY",           # or a list of keys for automatic failover
    model="gemini-2.5-flash",
    system_prompt="you are a helpful assistant",
    enable_grounding=False,          # turn on Google Search grounding
    history=True,                    # keep multi-turn context across calls
)

answer = ai.ask("Summarize this file for me.", file_path="notes.txt")
print(answer)
```

- `apikey` accepts a single key or a list; with a list, each key/model config is tried in order and it only raises once every config has failed.
- `file_path` accepts `None`, a single path, or a list. Files are uploaded to the Gemini Files API and cached (per key, path, size, mtime) for ~47 hours to avoid re-uploads.
- On failure, `ask()` raises `ExceptionGroup` with one `APIError` per failed attempt:

```python
try:
    ai.ask("...")
except ExceptionGroup as eg:
    for err in eg.exceptions:
        print(err)          # "[429] Resource exhausted..."
        print(err.raw())    # {"api_key": "AIza12...bcd", "code": 429, "status": "...", "message": "..."}
```

🔒 API keys are always masked (`AIza12...wxyz`) before an error reaches Python, so it's safe to log `APIError` output directly.

## 💬 CLI

Installing the package also gives you a chat REPL:

```bash
export GEMINI_API_KEYS="key1,key2"   # comma-separated for failover
gemlite
# or: python -m gemlite
```

Attach files inline with an `@file:` directive — a bracketed, comma-separated list of quoted paths:

```
You: describe @file:["./photo.png", "./photo2.jpg"]
```

The directive is stripped from the prompt before it's sent; any path that doesn't exist on disk is silently dropped.

## ✅ Requirements

- Python 3.11+ (uses `ExceptionGroup`, standard library since 3.11)
- A Gemini API key

## 🔧 Building from source / adding a platform

The Rust core (`src/lib.rs`, `Cargo.toml`, `.cargo/config.toml`) builds a `cdylib` named `gemlite`. `.github/workflows/main.yml` cross-compiles it for every target above and packages each into a correctly platform-tagged wheel (`manylinux_*`, `musllinux_*`, `win_amd64`, `win_arm64`, or `android_armv7` for Android), staging exactly one library file into `lib/` per wheel before it's built. At runtime, `core.py` doesn't need to know which platform it's on — it just looks for whatever single `*gemlite*` file pip installed into `lib/`. To add a target, add a Cargo target + matrix entry in the workflow; no changes to `core.py` are needed.
