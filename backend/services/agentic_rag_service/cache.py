"""
In-memory TTL cache for agentic query responses.

Purpose is budget, not speed: a cache hit costs 0 Gemini calls. Repeated or
double-submitted questions are the most common way a demo burns through the
free-tier quota.

Keyed on the normalised question so "What is the dress code?" and
"  what is the DRESS CODE?  " share one entry.
"""
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from services.agentic_rag_service.config import CACHE_MAX_ENTRIES, CACHE_TTL_SECONDS


def normalise_key(question: str) -> str:
    """Whitespace-collapsed, lowercased question — the cache key."""
    return " ".join(question.lower().split())


class TTLCache:
    """Thread-safe LRU + TTL cache. Small enough that O(n) eviction is fine."""

    def __init__(self, ttl: float = CACHE_TTL_SECONDS, max_entries: int = CACHE_MAX_ENTRIES):
        self._ttl = ttl
        self._max_entries = max_entries
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            expires_at, value = entry
            if expires_at <= now:
                # Expired — drop it and report a miss.
                del self._store[key]
                self._misses += 1
                return None

            # Refresh LRU position.
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        now = time.monotonic()
        with self._lock:
            self._store[key] = (now + self._ttl, value)
            self._store.move_to_end(key)

            # Drop anything already expired, then trim to size.
            expired = [k for k, (exp, _) in self._store.items() if exp <= now]
            for k in expired:
                del self._store[k]
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "ttlSeconds": self._ttl,
            }


@lru_cache
def get_cache() -> TTLCache:
    """Process-wide singleton."""
    return TTLCache()
