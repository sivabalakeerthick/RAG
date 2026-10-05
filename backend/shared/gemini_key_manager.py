"""
Multi-Key Rotation Manager for Gemini API Keys.
Rotates requests across multiple free-tier Gemini API keys in a round-robin
fashion, expanding total throughput (e.g. 6 keys = 6 x 12 RPM = 72 RPM budget).
"""
import itertools
import threading
from shared.config import settings


class GeminiKeyManager:
    """Thread-safe round-robin rotator across configured Gemini API keys."""

    def __init__(self):
        self._keys: list[str] = []
        self._cycler = None
        self._lock = threading.Lock()

    def _sync(self):
        current_keys = settings.gemini_keys_list
        if current_keys != self._keys:
            self._keys = current_keys
            self._cycler = itertools.cycle(current_keys) if current_keys else None

    def get_key(self) -> str:
        with self._lock:
            self._sync()
            if not self._keys or not self._cycler:
                return settings.GEMINI_API_KEY
            return next(self._cycler)

    @property
    def key_count(self) -> int:
        with self._lock:
            self._sync()
            return max(1, len(self._keys))


_key_manager = GeminiKeyManager()


def get_gemini_api_key() -> str:
    """Get the next Gemini API key in round-robin sequence."""
    return _key_manager.get_key()


def get_gemini_key_count() -> int:
    """Total number of active configured Gemini API keys."""
    return _key_manager.key_count
