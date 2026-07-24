"""Python wrapper around the ``gemlite`` Rust ``cdylib``.

The heavy lifting (HTTP calls, file uploads, multi-key failover) happens in a
small native library built from ``src/lib.rs``. This module is a thin
``ctypes`` binding on top of it, plus a couple of string-formatting helpers
used by the bundled ``gemlite`` REPL.

Layout:
    * :class:`StrTools`  -- text preprocessing/formatting helpers for the CLI.
    * :class:`_Lib`       -- loads and caches the native library handle.
    * :class:`APIError`   -- one failed request, as raised by :meth:`Gemini.ask`.
    * :class:`Gemini`     -- the public client.
    * :func:`main`        -- entry point for the ``gemlite`` console script.
"""

from __future__ import annotations

import re
import ctypes
try:
    import readline  # noqa: F401  (enables arrow-key history in the REPL, when available)
except ImportError:
    pass
from ast import literal_eval
from pathlib import Path
from random import randint
from json import dumps, loads
from wcwidth import wcswidth
from os import environ, path, get_terminal_size
from typing import Dict, List, Optional, Union
from ctypes import CDLL, cast, c_bool, c_char_p, c_void_p


class StrTools:
    """Text helpers used to format prompts and responses for the CLI REPL.

    Not needed when using :class:`Gemini` as a library -- these only exist to
    make ``gemlite``'s terminal output pleasant to read.
    """

    # Strips ANSI escape codes, raw control characters, and invisible
    # zero-width/formatting unicode that would otherwise mess up terminal
    # width calculations.
    ipattern = re.compile(
        r'\x1b\[[0-9;]*[a-zA-Z]'
        r'|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)'
        r'|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
        r'|[\u200b\u200c\u200d\u2060\ufeff]'
    )
    # Matches an inline `@file: [path1, path2]` directive in a CLI prompt.
    fpattern = re.compile(r'@file:\s*\[([^\]]+)\]')

    @staticmethod
    def preprocessor(text: str) -> tuple[str, List[str]]:
        """Extract an inline ``@file: [...]`` directive from a prompt.

        Returns the prompt text with the directive removed, plus the list of
        referenced paths that actually exist on disk (non-existent paths are
        silently dropped).
        """
        m = StrTools.fpattern.search(text)
        if not m:
            return text, []
        files = [f for f in literal_eval(f"[{m.group(1)}]") if path.isfile(f)]
        return (text[:m.start()] + text[m.end():]).strip(), files

    @staticmethod
    def paragraph(string: str, threshold: tuple[int, int] = (3, 5)) -> str:
        """Insert occasional paragraph breaks into a wall of model output.

        Splits on sentence-ending periods and randomly (every ``threshold[0]``
        to ``threshold[1]`` sentences) inserts a blank line, purely to make
        long unbroken responses easier to read in a terminal.
        """
        newpgh = []
        sentences = StrTools.ipattern.sub("", string).split(".")
        for idx, s in enumerate(sentences):
            if idx % randint(*threshold) == 0 and wcswidth(s) > 15:
                if idx + 1 <= len(sentences) - 1 and sentences[idx + 1].startswith(" "):
                    newpgh.append(s + ".\n\n")
                    continue
            newpgh.append(s + ".")
        return "".join(newpgh)

    @staticmethod
    def wraptext(string: str, padl: int, padr: int, ign1: bool = False) -> str:
        """Word-wrap ``string`` to the current terminal width, with padding.

        Args:
            string: Text to wrap (ANSI/control characters are stripped first).
            padl: Number of spaces to pad each line with on the left.
            padr: Number of spaces to pad each line with on the right.
            ign1: If True, don't left-pad the first line (useful when it
                continues a prefix already printed on the same line, e.g.
                ``"AI : "``).
        """
        newstr, width = [], get_terminal_size().columns - padl - padr
        if width <= 0:
            return ""
        lines = StrTools.ipattern.sub("", string).splitlines()
        for idx, line in enumerate(lines):
            if wcswidth(line) > width:
                newstr.append(f"{'' if (ign1 and idx == 0) else ' ' * padl}{line[:width - 1]}{' ' * padr}")
                lines.insert(idx + 1, line[width - 1:])
            else:
                newstr.append(f"{'' if (ign1 and idx == 0) else ' ' * padl}{line}{' ' * padr}")
        return "\n".join(newstr)


def _find_bundled_library() -> str:
    """Locate the prebuilt native library shipped inside this package.

    Each published wheel is platform-specific and is built with exactly one
    native library staged into ``lib/`` before packaging (see
    ``.github/workflows/main.yml``, which does ``rm -f lib/*.so lib/*.dll``
    before copying in the single file for that target). Because of that,
    there's no need to reimplement OS/arch/libc detection here -- pip already
    resolved that when it picked this wheel. We just need to find whatever
    single ``*gemlite*`` library file made it into the package.

    Raises:
        RuntimeError: if no library is bundled, or if more than one is found
            (e.g. a dev checkout with leftover build artifacts), in which
            case the caller should pass ``lib_path`` explicitly instead.
    """
    lib_dir = Path(__file__).parent / "lib"
    candidates = sorted(lib_dir.glob("*gemlite*"))

    if not candidates:
        raise RuntimeError(
            f"No bundled native library found in {lib_dir}. This platform may not have "
            "a prebuilt wheel -- build the Rust core yourself and pass lib_path= explicitly."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found multiple candidate libraries in {lib_dir}: {[c.name for c in candidates]}. "
            "Pass lib_path= explicitly to disambiguate."
        )
    return str(candidates[0])


class _Lib:
    """Loads the native ``cdylib`` and declares its FFI signatures.

    A process-wide singleton: repeated construction with the same
    ``lib_path`` reuses the already-loaded ``CDLL`` instead of reloading it,
    since native libraries generally shouldn't be loaded more than once per
    process.
    """

    _instance: Optional["_Lib"] = None

    def __new__(cls, lib_path: str) -> "_Lib":
        if cls._instance is None or cls._instance._path != lib_path:
            inst = super().__new__(cls)
            inst._path = lib_path
            inst._cdll = CDLL(lib_path)

            # bool Init(const char* configs_json)
            inst._cdll.Init.argtypes, inst._cdll.Init.restype = [c_char_p], c_bool
            # char* Ask(const char* turns_json, const char* file_paths_json)
            inst._cdll.Ask.argtypes, inst._cdll.Ask.restype = [c_char_p, c_char_p], c_void_p
            # void FreeString(char* ptr) -- must be called on every pointer Ask() returns
            inst._cdll.FreeString.argtypes, inst._cdll.FreeString.restype = [c_void_p], None

            cls._instance = inst
        return cls._instance

    @property
    def cdll(self) -> ctypes.CDLL:
        """The loaded ``ctypes.CDLL`` handle, with FFI signatures declared."""
        return self._cdll


class APIError(Exception):
    """Raised for a single failed Gemini API request.

    When multiple API keys are configured, :meth:`Gemini.ask` retries each
    one in order and only raises once all of them have failed -- in that case
    it raises an ``ExceptionGroup`` containing one ``APIError`` per attempt.
    """

    def __init__(self, detail: Dict[str, object]) -> None:
        self.detail = detail
        super().__init__(f"[{detail.get('code')}] {detail.get('message', '')}")

    def raw(self) -> Dict[str, object]:
        """The underlying error payload from the Rust core.

        Contains ``api_key`` (masked, e.g. ``"AIza12...wxyz"``), ``code``,
        ``status``, and ``message``. Safe to log directly.
        """
        return self.detail


class Gemini:
    """A Gemini API client backed by the native ``gemlite`` library.

    Example:
        >>> ai = Gemini(apikey="YOUR_API_KEY")
        >>> ai.ask("Summarize this file for me.", file_path="notes.txt")
    """

    __slots__ = ("_lib", "_history", "_enable_history")

    def __init__(
        self,
        apikey: Union[str, List[str]],
        model: str = "gemini-2.5-flash",
        system_prompt: str = "you are a helpful assistant",
        enable_grounding: bool = False,
        lib_path: Optional[str] = None,
        history: bool = True,
    ) -> None:
        """Configure the client and initialize the native library.

        Args:
            apikey: A single API key, or a list of keys for automatic
                failover -- each config is tried in order on failure.
            model: Gemini model name (e.g. ``"gemini-2.5-flash"``).
            system_prompt: System instruction applied to every request.
            enable_grounding: Enable Google Search grounding.
            lib_path: Explicit path to the native library. Defaults to the
                single prebuilt library bundled with this platform's wheel.
            history: Whether :meth:`ask` maintains multi-turn conversation
                history by default (can be overridden per call).

        Raises:
            ValueError: if the native library rejects the config payload.
        """
        keys = [apikey] if isinstance(apikey, str) else list(apikey)
        entries = [
            {"api_key": key, "system_prompt": system_prompt, "model": model, "enable_grounding": enable_grounding}
            for key in keys
        ]

        self._lib = _Lib(lib_path or _find_bundled_library()).cdll
        if not self._lib.Init(dumps(entries).encode("utf-8")):
            raise ValueError("Init failed: invalid config payload")

        self._history: List[Dict[str, str]] = []
        self._enable_history = history

    def reset(self) -> None:
        """Clear the stored conversation history."""
        self._history.clear()

    @property
    def history(self) -> List[Dict[str, str]]:
        """A copy of the conversation history as ``{"role", "text"}`` turns."""
        return list(self._history)

    @staticmethod
    def _normalize_paths(file_path: Union[str, List[str], None]) -> List[str]:
        """Coerce the ``file_path`` argument into a plain list of paths."""
        if not file_path:
            return []
        return [file_path] if isinstance(file_path, str) else list(file_path)

    def ask(
        self,
        question: str,
        file_path: Union[str, List[str], None] = None,
        use_history: Optional[bool] = None,
    ) -> str:
        """Send a prompt to Gemini and return the response text.

        Args:
            question: The prompt to send.
            file_path: Optional path, or list of paths, to upload alongside
                the prompt (uploaded via the Gemini Files API and cached for
                ~47 hours by the native core to avoid re-uploading).
            use_history: Whether to include and update conversation history
                for this call. Defaults to the ``history`` setting passed to
                the constructor.

        Returns:
            The model's response text.

        Raises:
            APIError: if a single configured key failed.
            ExceptionGroup: if multiple keys were configured and all failed
                (one ``APIError`` per attempt).
            RuntimeError: on an unexpected/empty response from the native
                library.
        """
        use_history = self._enable_history if use_history is None else use_history
        turns = (self._history if use_history else []) + [{"role": "user", "text": question}]

        ptr = self._lib.Ask(dumps(turns).encode("utf-8"), dumps(self._normalize_paths(file_path)).encode("utf-8"))
        if not ptr:
            raise RuntimeError("Ask returned a null pointer")
        try:
            raw = cast(ptr, c_char_p).value
        finally:
            # The native side allocated this string; we must free it regardless
            # of how the block above exits.
            self._lib.FreeString(ptr)
        if not raw:
            raise RuntimeError("Ask returned an empty response")

        answer = loads(raw.decode("utf-8"))
        if answer["status"] == "error":
            errs = [APIError(e) for e in answer["errors"]]
            if len(errs) == 1:
                raise errs[0]
            raise ExceptionGroup("gemini request failed", errs)

        result: str = answer["result"]
        if use_history:
            self._history.append({"role": "user", "text": question})
            self._history.append({"role": "model", "text": result})
        return result


def main() -> None:
    """Entry point for the `gemlite` console script (and `python -m gemlite`)."""
    keys = [k.strip() for k in environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        raise SystemExit("Set GEMINI_API_KEYS (comma-separated for multi-key failover) before running.")

    gem = Gemini(apikey=keys, model=environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    print("gemlite v4.0")
    while True:
        try:
            prompt, files = StrTools.preprocessor(input("\nYou: "))
            print("AI : ...", end="", flush=True)
            resp = gem.ask(prompt, file_path=files)
            print(f"{chr(8) * 3}{StrTools.wraptext(StrTools.paragraph(resp, (2,4)), 5, 2, ign1=True)}")
        except (KeyboardInterrupt, EOFError):
            print("\b\bExiting...")
            break
        except APIError as e:
            print(f"\b\b\bError: {e}")
        except ExceptionGroup as eg:
            print(f"\b\b\bAll configured keys failed: {'; '.join(str(x) for x in eg.exceptions)}")


if __name__ == "__main__":
    main()
