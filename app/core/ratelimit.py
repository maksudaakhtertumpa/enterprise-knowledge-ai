"""In-process rate limiting: token buckets per workspace and per client address.

Why: one workspace key can otherwise queue unlimited LLM calls (each takes seconds
of local GPU time) or uploads, starving every other user of the same server; and
failed key guesses cost a database lookup each. The limits:

- `ask`, `search`, `ingest` (upload / reprocess): per workspace, after auth;
- `auth_failures`: per client address. Each *failed* key check spends a token;
  when they run out, that address gets 429 for every authenticated call until
  the bucket refills. Successful requests never spend auth tokens.

A token bucket allows short bursts (`burst` requests at once) and a sustained
`per_minute` rate. The buckets live in this process's memory: correct for the
single-process deployment this project uses (one uvicorn worker, Phase 8 Docker).
Several workers or replicas would each have their own buckets; that setup needs
a shared store such as Redis, which this project deliberately doesn't run.
"""

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from ipaddress import ip_address, ip_network

from fastapi import Request

from app.core.config import Settings
from app.core.errors import AppError


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, retry_after: float) -> None:
        seconds = max(1, math.ceil(retry_after))
        super().__init__(message, details={"retry_after_seconds": seconds})
        self.headers = {"Retry-After": str(seconds)}


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """Thread-safe token buckets keyed by string, with bounded memory."""

    def __init__(
        self,
        per_minute: int,
        burst: int,
        *,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_minute < 1 or burst < 1:
            raise ValueError("per_minute and burst must be at least 1")
        self.rate = per_minute / 60.0  # tokens per second
        self.burst = float(burst)
        self.max_keys = max_keys
        self._clock = clock
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def _refill(self, key: str, now: float) -> _Bucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(self.burst, now)
            self._buckets[key] = bucket
            while len(self._buckets) > self.max_keys:
                # Oldest-touched first. A dropped key comes back with a full bucket,
                # which errs on the side of allowing requests, never of blocking them.
                self._buckets.popitem(last=False)
        else:
            bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            self._buckets.move_to_end(key)
        return bucket

    def acquire(self, key: str) -> float | None:
        """Spend one token. Returns None if allowed, else seconds until one is available."""
        with self._lock:
            bucket = self._refill(key, self._clock())
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return None
            return (1 - bucket.tokens) / self.rate

    def blocked(self, key: str) -> float | None:
        """Like acquire() but spends nothing: seconds to wait if the bucket is empty."""
        with self._lock:
            bucket = self._refill(key, self._clock())
            return None if bucket.tokens >= 1 else (1 - bucket.tokens) / self.rate


class RateLimits:
    """The app's limiters, built from settings and kept on app.state."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.rate_limit_enabled
        self.limiters = {
            "ask": TokenBucketLimiter(
                settings.rate_limit_ask_per_minute, settings.rate_limit_ask_burst
            ),
            "search": TokenBucketLimiter(
                settings.rate_limit_search_per_minute, settings.rate_limit_search_burst
            ),
            "ingest": TokenBucketLimiter(
                settings.rate_limit_ingest_per_minute, settings.rate_limit_ingest_burst
            ),
            "auth_failures": TokenBucketLimiter(
                settings.rate_limit_auth_failures_per_minute,
                settings.rate_limit_auth_failures_burst,
            ),
        }
        self.trusted_proxies = [ip_network(p, strict=False) for p in settings.trusted_proxies]

    def check(self, name: str, key: str) -> None:
        if not self.enabled:
            return
        wait = self.limiters[name].acquire(key)
        if wait is not None:
            raise RateLimitedError(
                f"Too many {name} requests for this workspace. Try again shortly.", wait
            )

    def check_auth_allowed(self, client: str) -> None:
        if not self.enabled:
            return
        wait = self.limiters["auth_failures"].blocked(client)
        if wait is not None:
            raise RateLimitedError("Too many failed API key attempts. Try again shortly.", wait)

    def record_auth_failure(self, client: str) -> None:
        if self.enabled:
            self.limiters["auth_failures"].acquire(client)

    def client_address(self, request: Request) -> str:
        """The caller's address. X-Forwarded-For is believed only from trusted proxies."""
        peer = request.client.host if request.client else "unknown"
        if not self.trusted_proxies or not self._trusted(peer):
            return peer
        forwarded = request.headers.get("x-forwarded-for", "")
        # Walk right to left: the right-most address not belonging to a trusted proxy
        # is the client. Left-most entries are whatever the client chose to send.
        for hop in reversed([h.strip() for h in forwarded.split(",") if h.strip()]):
            if not self._trusted(hop):
                return hop
        return peer

    def _trusted(self, address: str) -> bool:
        try:
            ip = ip_address(address)
        except ValueError:
            return False
        return any(ip in network for network in self.trusted_proxies)


def get_rate_limits(request: Request) -> RateLimits:
    return request.app.state.rate_limits
