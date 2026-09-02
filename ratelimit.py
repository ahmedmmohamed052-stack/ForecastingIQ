"""
🚦  Rate limiting — independent of subscription quotas.

Subscription quotas (billing.py) answer "has this user paid for more runs
this cycle?". This module answers a different question: "is this identity
sending requests faster than any legitimate use would?" — which matters
even for a user who still has quota left (a bug in their client retry loop,
or a bot that found the endpoint, can still hammer the server).

Implementation: a simple in-memory sliding-window counter keyed by uid
(falls back to client IP for unauthenticated attempts, so failed-auth
spam is also throttled). This is correct and sufficient for a single
server process. If you later run multiple server instances/workers behind
a load balancer, this under-counts (each process has its own memory) —
at that point move the counter to Redis (e.g. via `slowapi` or a small
custom INCR+EXPIRE script) instead of this dict.
"""
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException


class SlidingWindowLimiter:
    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str, max_calls: int, window_seconds: int):
        """Raises HTTPException(429) if `key` has exceeded max_calls in the window."""
        now = time.time()
        with self._lock:
            q = self._hits[key]
            cutoff = now - window_seconds
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= max_calls:
                retry_after = int(window_seconds - (now - q[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded ({max_calls} per "
                           f"{window_seconds}s). Try again in ~{retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )
            q.append(now)


_train_limiter = SlidingWindowLimiter()
_forecast_limiter = SlidingWindowLimiter()


def enforce_train_rate_limit(identity: str, per_hour: int):
    _train_limiter.check(identity, max_calls=per_hour, window_seconds=3600)


def enforce_forecast_rate_limit(identity: str, per_minute: int):
    _forecast_limiter.check(identity, max_calls=per_minute, window_seconds=60)
