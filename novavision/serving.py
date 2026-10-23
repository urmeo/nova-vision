from __future__ import annotations

import hmac
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from numbers import Integral, Real

LOCAL_HOST = "127.0.0.1"
PUBLIC_HOST = "0.0.0.0"


def is_loopback_host(host: str) -> bool:
    match = re.fullmatch(
        r"(?:localhost|127\.0\.0\.1|\[::1\])(?::([0-9]+))?",
        host,
        flags=re.IGNORECASE | re.ASCII,
    )
    if match is None:
        return False
    try:
        return match[1] is None or 1 <= int(match[1]) <= 65535
    except ValueError:
        return False


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def public_enabled() -> bool:
    return _truthy(os.getenv("NOVA_PUBLIC")) or bool(os.getenv("SPACE_ID"))


def resolve_host(default: str = LOCAL_HOST) -> str:
    return PUBLIC_HOST if public_enabled() else default


def token_ok(provided: str | None) -> bool:
    expected = os.getenv("NOVA_API_TOKEN", "").strip() or None
    if expected is None:
        return not public_enabled()
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


def _integer_at_least(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _finite_number(value: object, name: str) -> float:
    error = f"{name} must be a finite number"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(error)
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(error) from None
    if not math.isfinite(number):
        raise ValueError(error)
    return number


class RateLimiter:

    def __init__(self, max_requests: int, window_seconds: float = 60.0, gc_threshold: int = 4096):
        self.max_requests = _integer_at_least(max_requests, "max_requests", 1)
        self.window = _finite_number(window_seconds, "window_seconds")
        if self.window <= 0:
            raise ValueError("window_seconds must be greater than zero")
        self._gc_threshold = _integer_at_least(gc_threshold, "gc_threshold", 0)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        if now is not None:
            now = _finite_number(now, "now")
        with self._lock:
            now = time.monotonic() if now is None else now
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

    def __init__(self, max_concurrent: int):
        self._semaphore = threading.BoundedSemaphore(
            _integer_at_least(max_concurrent, "max_concurrent", 1)
        )

    def acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()
