"""Authentication and request limits for the Flask API."""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque

LOCAL_HOST = "127.0.0.1"
PUBLIC_HOST = "0.0.0.0"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def public_enabled() -> bool:
    "Whether the operator explicitly asked to expose the server publicly."
    return _truthy(os.getenv("NOVA_PUBLIC")) or bool(os.getenv("SPACE_ID"))


def resolve_host(default: str = LOCAL_HOST) -> str:
    """Bind localhost by default; bind all interfaces only on explicit opt-in."""
    return PUBLIC_HOST if public_enabled() else default


def token_ok(provided: str | None) -> bool:
    "Constant-time comparison of a presented token against ``NOVA_API_TOKEN``."
    expected = os.getenv("NOVA_API_TOKEN", "").strip() or None
    if expected is None:
        return True
    if provided is None:
        return False

    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


class RateLimiter:
    """Thread-safe per-key sliding-window limiter, no external dependency."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0, gc_threshold: int = 4096):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._gc_threshold = gc_threshold

    def allow(self, key: str, *, now: float | None = None) -> bool:
        "Record a request within the sliding-window budget."
        now = time.monotonic() if now is None else now
        with self._lock:
            cutoff = now - self.window
            if len(self._hits) > self._gc_threshold:
                stale_keys = [
                    client_key
                    for client_key, requests in self._hits.items()
                    if not requests or requests[-1] <= cutoff
                ]
                for client_key in stale_keys:
                    del self._hits[client_key]
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True


class ConcurrencyGuard:
    "A non-blocking concurrency cap: a fixed number of slots, no queueing."

    def __init__(self, max_concurrent: int):
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrent))

    def acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()
