"""
Async sliding-window rate limiter for Gemini free-tier protection.

Why a sliding window and not a token bucket: the Gemini free tier enforces
"N requests per rolling 60s", which is exactly a sliding window. A token bucket
with a refill rate would allow a burst of 12 immediately after a quiet minute
*plus* the refill drip, which can still trip a hard rolling-window quota.

Usage:
    limiter = get_rate_limiter()
    await limiter.acquire()          # waits for a slot, raises RateLimitExceeded
"""
import asyncio
import time
from collections import deque
from functools import lru_cache

from shared.gemini_key_manager import get_gemini_key_count
from services.agentic_rag_service.config import (
    RATE_LIMIT_MAX_CALLS,
    RATE_LIMIT_MAX_WAIT_SECONDS,
    RATE_LIMIT_WINDOW_SECONDS,
)


class RateLimitExceeded(Exception):
    """Raised when no slot frees up within RATE_LIMIT_MAX_WAIT_SECONDS."""

    def __init__(self, retry_after: float):
        self.retry_after = max(0.0, retry_after)
        super().__init__(
            f"Gemini rate limit reached. Retry in {self.retry_after:.1f}s."
        )


class SlidingWindowRateLimiter:
    """Allows at most `max_calls` acquisitions per rolling `window` seconds across all configured keys."""

    def __init__(
        self,
        max_calls: int = RATE_LIMIT_MAX_CALLS,
        window: float = RATE_LIMIT_WINDOW_SECONDS,
        max_wait: float = RATE_LIMIT_MAX_WAIT_SECONDS,
    ):
        self._base_max_calls = max_calls
        self._window = window
        self._max_wait = max_wait
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    @property
    def max_calls(self) -> int:
        return self._base_max_calls * get_gemini_key_count()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    async def acquire(self) -> None:
        """
        Reserve one call slot, sleeping until one is free.

        Raises RateLimitExceeded if the wait would exceed max_wait, so a caller
        gets a clean 429 instead of hanging on a long queue.
        """
        deadline = time.monotonic() + self._max_wait

        while True:
            async with self._lock:
                now = time.monotonic()
                self._evict(now)

                if len(self._calls) < self.max_calls:
                    # Slot reserved while still holding the lock
                    self._calls.append(now)
                    return

                # Oldest call leaves the window at this time.
                wait_for = (self._calls[0] + self._window) - now

            if time.monotonic() + wait_for > deadline:
                raise RateLimitExceeded(retry_after=wait_for)

            # Sleep outside the lock so other coroutines can make progress.
            await asyncio.sleep(min(wait_for, 0.5) + 0.01)

    def snapshot(self) -> dict:
        """Non-blocking view of current usage, for the health endpoint."""
        now = time.monotonic()
        cutoff = now - self._window
        used = sum(1 for t in self._calls if t > cutoff)
        effective_limit = self.max_calls
        return {
            "used": used,
            "limit": effective_limit,
            "remaining": max(0, effective_limit - used),
            "windowSeconds": self._window,
            "keysConfigured": get_gemini_key_count(),
        }


@lru_cache
def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Process-wide singleton. One uvicorn worker == one limiter."""
    return SlidingWindowRateLimiter()
