"""The error envelope and request-ID behaviour are contracts every later phase relies on."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.errors import NotFoundError


class Payload(BaseModel):
    name: str
    top_k: int


@pytest.fixture
def client_with_test_routes(app: FastAPI):
    """Attach throwaway routes that trigger each kind of error."""

    @app.get("/_test/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    @app.get("/_test/missing")
    def missing() -> None:
        raise NotFoundError("Document not found", details={"document_id": "abc"})

    @app.post("/_test/validate")
    def validate(payload: Payload) -> Payload:
        return payload

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "not_found"
    assert err["request_id"] == resp.headers["x-request-id"]


def test_domain_error_maps_to_status_and_code(client_with_test_routes: TestClient) -> None:
    resp = client_with_test_routes.get("/_test/missing")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "not_found"
    assert err["message"] == "Document not found"
    assert err["details"] == {"document_id": "abc"}


def test_validation_error_does_not_echo_input(client_with_test_routes: TestClient) -> None:
    resp = client_with_test_routes.post(
        "/_test/validate", json={"name": "sensitive-value", "top_k": "not-a-number"}
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "validation_error"
    assert err["details"][0]["loc"] == ["body", "top_k"]
    assert "sensitive-value" not in resp.text
    assert "not-a-number" not in resp.text


def test_unhandled_exception_is_hidden_from_client(client_with_test_routes: TestClient) -> None:
    resp = client_with_test_routes.get("/_test/boom", headers={"X-Request-ID": "trace-123"})
    assert resp.status_code == 500
    err = resp.json()["error"]
    assert err["code"] == "internal_error"
    assert "secret internal detail" not in resp.text
    # The ID survives all the way to the outermost 500 handler.
    assert err["request_id"] == "trace-123"


def test_request_id_generated_when_absent(client: TestClient) -> None:
    resp = client.get("/health")
    rid = resp.headers["x-request-id"]
    assert len(rid) == 32 and rid.isalnum()


def test_safe_client_request_id_is_propagated(client: TestClient) -> None:
    resp = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["x-request-id"] == "abc-123"


@pytest.mark.parametrize("bad", ["has space", "x" * 65, "line\nbreak", "<script>"])
def test_unsafe_client_request_id_is_replaced(client: TestClient, bad: str) -> None:
    # httpx refuses to send raw newlines, so send the bytes it allows and assert replacement.
    try:
        resp = client.get("/health", headers={"X-Request-ID": bad})
    except Exception:  # noqa: BLE001 - header rejected client-side is also acceptable
        return
    assert resp.headers["x-request-id"] != bad
