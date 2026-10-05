"""
In-memory TTL cache with Semantic Vector Caching for agentic query responses.

Purpose is budget and latency:
  1. Exact Match: 0 API calls, 0 DB queries (<0.1ms).
  2. Semantic Match: detects rephrased questions (e.g., 'What is leave policy' vs
     'How many annual leaves do I get') via query embedding cosine similarity >= 0.93.
     Skips searching 100% of chunks, 0 DB queries, and 0 generation calls.
"""
import math
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any

from services.agentic_rag_service.config import (
    CACHE_MAX_ENTRIES,
    CACHE_TTL_SECONDS,
    SEMANTIC_CACHE_THRESHOLD,
)


def normalise_key(question: str) -> str:
    """Whitespace-collapsed, lowercased question — the cache key."""
    return " ".join(question.lower().split())


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


class TTLCache:
    """Thread-safe LRU + TTL cache with semantic vector lookup."""

    def __init__(self, ttl: float = CACHE_TTL_SECONDS, max_entries: int = CACHE_MAX_ENTRIES):
        self._ttl = ttl
        self._max_entries = max_entries
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._vectors: OrderedDict[str, tuple[float, list[float], Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._semantic_hits = 0
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
                del self._store[key]
                if key in self._vectors:
                    del self._vectors[key]
                self._misses += 1
                return None

            self._store.move_to_end(key)
            self._hits += 1
            return value

    def get_semantic(
        self,
        query_vector: list[float],
        threshold: float = SEMANTIC_CACHE_THRESHOLD,
    ) -> Any | None:
        """Find cached response via vector cosine similarity without scanning any chunks."""
        if not query_vector:
            return None
        now = time.monotonic()
        best_sim = 0.0
        best_val = None
        best_key = None

        with self._lock:
            for k, (exp, vec, val) in list(self._vectors.items()):
                if exp <= now:
                    del self._vectors[k]
                    if k in self._store:
                        del self._store[k]
                    continue

                sim = _cosine_similarity(query_vector, vec)
                if sim > best_sim:
                    best_sim = sim
                    best_val = val
                    best_key = k

            if best_sim >= threshold and best_val is not None:
                self._semantic_hits += 1
                if best_key and best_key in self._store:
                    self._store.move_to_end(best_key)
                return best_val

        return None

    def set(self, key: str, value: Any, vector: list[float] | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            self._store[key] = (now + self._ttl, value)
            self._store.move_to_end(key)

            if vector:
                self._vectors[key] = (now + self._ttl, vector, value)
                self._vectors.move_to_end(key)

            # Evict expired
            expired = [k for k, (exp, _) in self._store.items() if exp <= now]
            for k in expired:
                del self._store[k]
                if k in self._vectors:
                    del self._vectors[k]

            while len(self._store) > self._max_entries:
                oldest_k, _ = self._store.popitem(last=False)
                if oldest_k in self._vectors:
                    del self._vectors[oldest_k]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._vectors.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._store),
                "hits": self._hits,
                "semanticHits": self._semantic_hits,
                "misses": self._misses,
                "ttlSeconds": self._ttl,
            }


@lru_cache
def get_cache() -> TTLCache:
    """Process-wide singleton."""
    return TTLCache()
