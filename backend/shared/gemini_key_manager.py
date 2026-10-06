"""
Multi-Key Rotation Manager for Gemini API Keys.

Rotates requests across multiple free-tier Gemini API keys in strict 1-to-N round-robin
order, expanding aggregate throughput:
  - 1 key  = 15 RPM
  - 6 keys = 6 x 15 RPM = 90 RPM aggregate capacity
  - At 1 call/second (60 RPM load), you have 30 unused calls/minute safety headroom.
"""
import itertools
import logging
import os
import threading
from shared.config import settings

logger = logging.getLogger(__name__)


class GeminiKeyManager:
    """Thread-safe round-robin rotator across configured Gemini API keys."""

    def __init__(self):
        self._keys: list[str] = []
        self._cycler = None
        self._lock = threading.Lock()
        self._call_counter = 0

    def _sync(self):
        current_keys = settings.gemini_keys_list
        if current_keys != self._keys:
            self._keys = current_keys
            if current_keys:
                # Stagger the initial position by process ID so independent
                # microservices (agentic on 8005, rag on 8002, judge on 8003, document on 8001)
                # don't all hit Key 1 at the same exact second.
                offset = os.getpid() % len(current_keys)
                staggered = current_keys[offset:] + current_keys[:offset]
                self._cycler = itertools.cycle(staggered)
            else:
                self._cycler = None

    def get_key(self) -> str:
        with self._lock:
            self._sync()
            if not self._keys or not self._cycler:
                return settings.GEMINI_API_KEY
            key = next(self._cycler)
            self._call_counter += 1
            key_index = (self._keys.index(key) + 1) if key in self._keys else 1
            logger.debug(
                "[GeminiKeyManager] Call #%d routed to Key %d/%d (suffix: ...%s)",
                self._call_counter, key_index, len(self._keys), key[-6:],
            )
            return key

    @property
    def key_count(self) -> int:
        with self._lock:
            self._sync()
            return max(1, len(self._keys))


_key_manager = GeminiKeyManager()


def get_gemini_api_key() -> str:
    """Get the next Gemini API key in strict 1-to-N round-robin sequence."""
    return _key_manager.get_key()


def get_gemini_key_count() -> int:
    """Total number of active configured Gemini API keys."""
    return _key_manager.key_count
