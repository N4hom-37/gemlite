"""gemlite: a Gemini API client with a Rust/ctypes core.

HTTP calls, file uploads, and multi-key failover are handled by a small
native library; this package is the Python-facing wrapper around it.

    >>> from gemlite import Gemini
    >>> ai = Gemini(apikey="YOUR_API_KEY")
    >>> ai.ask("Hello!")
"""

from .core import Gemini, StrTools, APIError

__version__ = "1.0.0"
__all__ = ["Gemini", "StrTools", "APIError"]
