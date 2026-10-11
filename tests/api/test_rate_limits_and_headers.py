"""Rate limits (429 + Retry-After), auth-failure throttling, and API security headers."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from app.core.config import Settings
from app.core.ratelimit import RateLimits, TokenBucketLimiter
from app.core.security import create_workspace_with_key
from app.main import create_app

API = "/api/v1"


# ------------------------------------------------------------ token bucket
class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_bucket_allows_a_burst_then_the_sustained_rate():
    clock = FakeClock()
    limiter = TokenBucketLimiter(per_minute=6, burst=3, clock=clock)  # 1 token / 10 s
    assert [limiter.acquire("a") for _ in range(3)] == [None, None, None]
    wait = limiter.acquire("a")
    assert wait == pytest.approx(10.0)
    assert limiter.acquire("b") is None  # keys are independent
    clock.now += 5
    assert limiter.acquire("a") == pytest.approx(5.0)  # half a token so far
    clock.now += 5
    assert limiter.acquire("a") is None
    clock.now += 3600  # refills, but never above the burst
    assert [limiter.acquire("a") for _ in range(4)][-1] is not None


def test_blocked_checks_without_spending():
    clock = FakeClock()
    limiter = TokenBucketLimiter(per_minute=60, burst=1, clock=clock)
    assert limiter.blocked("x") is None and limiter.blocked("x") is None
    limiter.acquire("x")
    assert limiter.blocked("x") == pytest.approx(1.0)


def test_bucket_memory_is_bounded():
    limiter = TokenBucketLimiter(per_minute=1, burst=1, max_keys=100, clock=FakeClock())
    for i in range(1000):
        limiter.acquire(f"k{i}")
    assert len(limiter._buckets) == 100


def test_bucket_rejects_nonsense_limits():
    with pytest.raises(ValueError):
        TokenBucketLimiter(per_minute=0, burst=1)


# --------------------------------------------------------- client address
def _request(peer: str, forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "client": (peer, 1234), "headers": headers})


def _limits(tmp_path: Path, **extra) -> RateLimits:
    return RateLimits(Settings(data_dir=tmp_path, _env_file=None, **extra))


def test_forwarded_for_is_ignored_unless_the_peer_is_a_trusted_proxy(tmp_path: Path):
    limits = _limits(tmp_path)
    assert limits.client_address(_request("203.0.113.9", "1.2.3.4")) == "203.0.113.9"

    proxied = _limits(tmp_path, trusted_proxies=["10.0.0.0/8"])
    # From the proxy: the right-most untrusted hop is the client; a spoofed
    # left-most entry added by the client is not believed.
    request = _request("10.0.0.2", "6.6.6.6, 198.51.100.7, 10.0.0.3")
    assert proxied.client_address(request) == "198.51.100.7"
    assert proxied.client_address(_request("203.0.113.9", "1.2.3.4")) == "203.0.113.9"
    assert proxied.client_address(_request("10.0.0.2")) == "10.0.0.2"
    assert proxied.client_address(_request("10.0.0.2", "not-an-ip")) == "not-an-ip"


def test_invalid_trusted_proxy_is_a_configuration_error(tmp_path: Path):
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, trusted_proxies=["not-a-network"], _env_file=None)


# ------------------------------------------------------------------ API
@pytest.fixture
def limited_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "rate_limit_ask_per_minute": 1,
            "rate_limit_ask_burst": 2,
            "rate_limit_search_per_minute": 1,
            "rate_limit_search_burst": 3,
            "rate_limit_ingest_per_minute": 1,
            "rate_limit_ingest_burst": 1,
            "rate_limit_auth_failures_per_minute": 1,
            "rate_limit_auth_failures_burst": 3,
        }
    )


@pytest.fixture
def limited(limited_settings: Settings) -> Iterator[tuple[TestClient, dict, dict]]:
    app = create_app(limited_settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        keys = []
        with app.state.session_factory() as db:
            for name in ("Acme", "Globex"):
                keys.append(create_workspace_with_key(db, name)[1])
            db.commit()
        yield client, {"X-API-Key": keys[0]}, {"X-API-Key": keys[1]}


def _assert_429(resp) -> None:
    assert resp.status_code == 429, resp.text
    body = resp.json()["error"]
    assert body["code"] == "rate_limited"
    retry = int(resp.headers["Retry-After"])
    assert 1 <= retry <= 60 and body["details"]["retry_after_seconds"] == retry


def test_ask_is_limited_per_workspace(limited):
    client, auth, other = limited
    for _ in range(2):
        assert (
            client.post(f"{API}/ask", headers=auth, json={"question": "leave?"}).status_code == 200
        )
    _assert_429(client.post(f"{API}/ask", headers=auth, json={"question": "leave?"}))
    # Another workspace has its own bucket; cheap reads are not limited.
    assert client.post(f"{API}/ask", headers=other, json={"question": "leave?"}).status_code == 200
    assert client.get(f"{API}/workspace", headers=auth).status_code == 200
    assert client.get(f"{API}/ask/history", headers=auth).status_code == 200


def test_search_and_uploads_have_their_own_buckets(limited):
    client, auth, _ = limited
    for _ in range(3):
        assert client.post(f"{API}/search", headers=auth, json={"query": "x"}).status_code == 200
    _assert_429(client.post(f"{API}/search", headers=auth, json={"query": "x"}))
    files = {"file": ("a.txt", b"Annual leave is 14 days.", "text/plain")}
    assert client.post(f"{API}/documents", headers=auth, files=files).status_code == 202
    files = {"file": ("b.txt", b"Parking is on B2.", "text/plain")}
    _assert_429(client.post(f"{API}/documents", headers=auth, files=files))
    assert client.post(f"{API}/ask", headers=auth, json={"question": "x"}).status_code == 200


def test_repeated_bad_keys_make_the_address_wait(limited):
    client, auth, _ = limited
    bad = {"X-API-Key": "eka_" + "x" * 43}
    for headers in (bad, {}, {"X-API-Key": "nope"}):  # every kind of failure counts
        assert client.get(f"{API}/workspace", headers=headers).status_code == 401
    # Bucket empty: even a valid key waits, so guessing can't continue in between.
    _assert_429(client.get(f"{API}/workspace", headers=auth))
    assert client.get("/health").status_code == 200  # probes are unauthenticated, unaffected


def test_successful_requests_do_not_spend_auth_tokens(limited):
    client, auth, _ = limited
    for _ in range(10):
        assert client.get(f"{API}/workspace", headers=auth).status_code == 200


def test_limits_can_be_switched_off(limited_settings: Settings):
    app = create_app(limited_settings.model_copy(update={"rate_limit_enabled": False}))
    with TestClient(app) as client:
        with app.state.session_factory() as db:
            key = create_workspace_with_key(db, "Acme")[1]
            db.commit()
        for _ in range(5):
            assert (
                client.get(f"{API}/workspace", headers={"X-API-Key": "eka_bad"}).status_code == 401
            )
        resp = client.post(f"{API}/search", headers={"X-API-Key": key}, json={"query": "x"})
        assert resp.status_code == 200


# --------------------------------------------------------------- headers
SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
}


def test_every_api_response_carries_security_headers(client, auth):
    responses = [
        client.get(f"{API}/workspace", headers=auth),  # 200
        client.get(f"{API}/workspace"),  # 401
        client.get(f"{API}/nope", headers=auth),  # 404
        client.post(f"{API}/search", headers=auth, json={}),  # 422
        client.post(
            f"{API}/documents",
            headers={**auth, "Content-Length": str(10**10)},
            content=b"x",
        ),  # 413 from the size-limit middleware
        client.get("/health"),
    ]
    assert [r.status_code for r in responses] == [200, 401, 404, 422, 413, 200]
    for resp in responses:
        for name, value in SECURITY_HEADERS.items():
            assert resp.headers.get(name) == value, (resp.request.url, name)


def test_docs_pages_skip_the_csp_and_are_off_in_production(client, settings: Settings):
    docs = client.get("/docs")
    assert docs.status_code == 200 and "content-security-policy" not in docs.headers
    assert docs.headers["x-frame-options"] == "DENY"

    prod = create_app(settings.model_copy(update={"environment": "production"}))
    with TestClient(prod) as c:
        assert [c.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404] * 3
        assert c.get("/health").status_code == 200
    forced = create_app(settings.model_copy(update={"environment": "production", "api_docs": True}))
    with TestClient(forced) as c:
        assert c.get("/openapi.json").status_code == 200


# ------------------------------------------------------------------ logs
def test_api_keys_and_questions_never_reach_the_logs(client, auth, caplog):
    secret_question = "What is the salary of employee 4417-SECRET?"
    with caplog.at_level("DEBUG"):
        client.get(f"{API}/workspace", headers=auth)
        client.get(f"{API}/workspace", headers={"X-API-Key": "eka_" + "W" * 43})
        client.post(f"{API}/ask", headers=auth, json={"question": secret_question})
        client.post(f"{API}/search", headers=auth, json={"query": secret_question})
    assert caplog.records, "expected access-log records"
    for record in caplog.records:
        dump = repr(record.__dict__) + record.getMessage()
        assert auth["X-API-Key"] not in dump and "W" * 43 not in dump
        assert "4417-SECRET" not in dump, record.__dict__
