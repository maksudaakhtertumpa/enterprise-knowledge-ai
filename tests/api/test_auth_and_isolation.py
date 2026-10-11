"""Authentication and cross-workspace isolation — the security contract of the API."""

import pytest

from app.cli import main as cli
from app.core.security import hash_api_key
from app.db.models import ApiKey, utcnow

API = "/api/v1"


def upload(client, auth, name="a.txt", data=b"Confidential salary bands."):
    return client.post(f"{API}/documents", headers=auth, files={"file": (name, data, "text/plain")})


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-API-Key": ""},
        {"X-API-Key": "not-an-eka-key"},
        {"X-API-Key": "eka_" + "x" * 43},  # right shape, unknown key
        {"X-API-Key": "eka_" + "x" * 500},  # oversized
    ],
)
def test_requests_without_valid_key_get_401(client, workspace, headers):
    for method, path in [("get", "/documents"), ("get", "/workspace"), ("post", "/documents")]:
        resp = getattr(client, method)(f"{API}{path}", headers=headers)
        assert resp.status_code == 401, (method, path)
        assert resp.json()["error"] == {
            "code": "unauthorized",
            "message": "Missing or invalid API key",
            "details": None,
            "request_id": resp.headers["x-request-id"],
        }


def test_workspace_endpoint_identifies_caller(client, auth, workspace):
    assert client.get(f"{API}/workspace", headers=auth).json() == {
        "id": workspace["id"],
        "name": "Acme",
    }


def test_revoked_key_is_rejected(client, auth, app, workspace):
    assert client.get(f"{API}/workspace", headers=auth).status_code == 200
    with app.state.session_factory() as db:
        key = db.query(ApiKey).filter_by(key_hash=hash_api_key(workspace["key"])).one()
        key.revoked_at = utcnow()
        db.commit()
    assert client.get(f"{API}/workspace", headers=auth).status_code == 401


def test_only_key_hash_is_stored(app, workspace):
    with app.state.session_factory() as db:
        stored = db.query(ApiKey).one()
    assert stored.key_hash == hash_api_key(workspace["key"])
    assert workspace["key"] not in (stored.key_hash, stored.key_prefix)


def test_other_workspace_cannot_see_or_touch_documents(client, auth, other_auth):
    doc_id = upload(client, auth).json()["id"]

    # Listing: the other workspace sees nothing.
    assert client.get(f"{API}/documents", headers=other_auth).json()["total"] == 0

    # Direct access by ID: indistinguishable from "does not exist".
    for method, path in [
        ("get", f"/documents/{doc_id}"),
        ("get", f"/documents/{doc_id}/chunks"),
        ("post", f"/documents/{doc_id}/reprocess"),
        ("delete", f"/documents/{doc_id}"),
    ]:
        resp = getattr(client, method)(f"{API}{path}", headers=other_auth)
        assert resp.status_code == 404, (method, path)
        assert resp.json()["error"]["message"] == "Document not found"

    # And the owner still has it, untouched.
    assert client.get(f"{API}/documents/{doc_id}", headers=auth).json()["status"] == "ready"


def test_duplicate_check_does_not_leak_across_workspaces(client, auth, other_auth):
    upload(client, auth)
    resp = upload(client, other_auth)
    assert resp.status_code == 202, "a 409 here would reveal the other workspace holds the file"


# ----------------------------------------------------------------------- CLI
def test_cli_create_list_issue_revoke(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EKA_DATA_DIR", str(tmp_path))
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        assert cli(["create-workspace", "Acme HR"]) == 0
        out = capsys.readouterr().out
        key = next(line.split()[-1] for line in out.splitlines() if line.startswith("API key"))
        assert key.startswith("eka_")

        assert cli(["create-workspace", "Acme HR"]) == 1  # duplicate name
        assert cli(["issue-key", "Acme HR", "--label", "laptop"]) == 0
        capsys.readouterr()
        assert cli(["list-workspaces"]) == 0
        listing = capsys.readouterr().out
        assert "Acme HR" in listing and "laptop" in listing and key not in listing

        assert cli(["revoke-key", key[:12]]) == 0
        assert cli(["revoke-key", key[:12]]) == 1  # already revoked
        assert cli(["issue-key", "Nope"]) == 1
    finally:
        get_settings.cache_clear()


def test_cli_prepare_models_reports_each_model(monkeypatch, capsys):
    monkeypatch.setenv("EKA_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("EKA_RERANKER_BACKEND", "hashing")
    monkeypatch.setenv("EKA_ENTAILMENT_BACKEND", "lexical")
    monkeypatch.setenv("EKA_LLM_BACKEND", "extractive")
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        assert cli(["prepare-models"]) == 0
        out = capsys.readouterr().out
        assert "hashing-v1" in out and "overlap-v1" in out and "All models ready." in out
        assert "test answer: Employees receive 14 days" in out
        assert "Entailment model: lexical-v1" in out
        assert "test scores supported=" in out

        # Ollama not running: fail with install instructions.
        monkeypatch.setenv("EKA_LLM_BACKEND", "ollama")
        monkeypatch.setenv("EKA_OLLAMA_URL", "http://127.0.0.1:9")
        get_settings.cache_clear()
        assert cli(["prepare-models"]) == 1
        err = capsys.readouterr().err
        assert "ollama.com/download" in err and "ollama pull qwen3.5:4b" in err
        monkeypatch.setenv("EKA_LLM_BACKEND", "extractive")

        monkeypatch.setenv("EKA_RERANKER_BACKEND", "none")
        get_settings.cache_clear()
        assert cli(["prepare-models"]) == 0
        assert "Reranker: disabled" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()
