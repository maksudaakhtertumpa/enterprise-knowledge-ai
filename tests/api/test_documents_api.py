"""End-to-end tests through HTTP: upload -> background ingestion -> inspect -> delete.

TestClient runs background tasks before returning the response, so a
document's final status is visible immediately after the upload call.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.security import create_workspace_with_key
from app.db.models import Chunk, Document, DocumentStatus
from app.main import create_app
from tests.factories import blank_pdf, encrypt_pdf, make_docx, make_pdf

API = "/api/v1"
LEAVE = (
    "Employees are entitled to fourteen days of annual leave per calendar year. "
    "Leave must be approved by the line manager at least two weeks in advance."
)


def upload(client, auth, name, data, content_type="application/octet-stream"):
    return client.post(f"{API}/documents", headers=auth, files={"file": (name, data, content_type)})


def sample_pdf() -> bytes:
    return make_pdf(
        [["Leave Policy", LEAVE] * 3, ["Claims are reimbursed within 30 days."]],
        outline={1: "Leave", 2: "Claims"},
        title="HR Handbook",
    )


# ------------------------------------------------------------------- upload
@pytest.mark.parametrize(
    ("name", "data", "expect_pages"),
    [
        ("handbook.pdf", sample_pdf(), 2),
        ("handbook.docx", make_docx([("h1", "Leave"), ("p", LEAVE)]), None),
        ("notes.txt", f"# Leave\n\n{LEAVE}\n".encode(), None),
    ],
)
def test_upload_each_format_reaches_ready(client, auth, name, data, expect_pages):
    resp = upload(client, auth, name, data)
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["filename"] == name
    assert body["status"] == "pending"  # the response is sent before processing

    doc = client.get(f"{API}/documents/{body['id']}", headers=auth).json()
    assert doc["status"] == "ready", doc
    assert doc["page_count"] == expect_pages
    assert doc["chunk_count"] >= 1
    assert doc["processed_at"] is not None
    assert doc["metadata"]["chunking"]["chunk_size_tokens"] == 400

    chunks = client.get(f"{API}/documents/{body['id']}/chunks", headers=auth).json()
    assert chunks["total"] == doc["chunk_count"]
    assert any("fourteen days" in c["text"] for c in chunks["items"])
    assert any(c["heading"] == "Leave" for c in chunks["items"])


def test_pdf_chunks_carry_page_numbers(client, auth):
    doc_id = upload(client, auth, "h.pdf", sample_pdf()).json()["id"]
    items = client.get(f"{API}/documents/{doc_id}/chunks", headers=auth).json()["items"]
    claims = next(c for c in items if "reimbursed" in c["text"])
    assert claims["page_end"] == 2 and claims["heading"] == "Claims"
    assert all(c["page_start"] is not None for c in items)


def test_chunks_are_exact_slices_of_extracted_text(client, auth, app):
    doc_id = upload(client, auth, "h.pdf", sample_pdf()).json()["id"]
    with app.state.session_factory() as db:
        doc = db.get(Document, doc_id)
        for chunk in doc.chunks:
            assert doc.extracted_text[chunk.char_start : chunk.char_end] == chunk.text


def test_duplicate_upload_returns_409_with_existing_id(client, auth, app, settings):
    first = upload(client, auth, "a.txt", b"Same content.").json()
    resp = upload(client, auth, "renamed.txt", b"Same content.")
    assert resp.status_code == 409
    assert resp.json()["error"]["details"] == {"existing_document_id": first["id"]}
    files = [p for p in settings.uploads_dir.rglob("*") if p.is_file()]
    assert len(files) == 1, "the rejected duplicate must not be left on disk"


def test_same_file_allowed_in_different_workspaces(client, auth, other_auth):
    assert upload(client, auth, "a.txt", b"Shared.").status_code == 202
    assert upload(client, other_auth, "a.txt", b"Shared.").status_code == 202


@pytest.mark.parametrize(
    ("name", "data", "status", "code"),
    [
        ("malware.exe", b"MZ...", 415, "unsupported_type"),
        ("fake.pdf", b"this is plain text", 415, "content_mismatch"),
        ("fake.docx", b"%PDF-1.7 nope", 415, "content_mismatch"),
        ("empty.txt", b"", 400, "app_error"),
    ],
)
def test_bad_uploads_rejected_and_not_stored(client, auth, settings, name, data, status, code):
    resp = upload(client, auth, name, data)
    assert resp.status_code == status, resp.text
    assert resp.json()["error"]["code"] == code
    assert not [p for p in settings.uploads_dir.rglob("*") if p.is_file()]


def test_missing_file_field_is_422(client, auth):
    resp = client.post(f"{API}/documents", headers=auth, data={"nope": "1"})
    assert resp.status_code == 422


def test_oversize_upload_rejected_with_413(tmp_path):
    settings = Settings(
        environment="test",
        data_dir=tmp_path,
        max_upload_mb=1,
        embedding_backend="hashing",
        reranker_backend="hashing",
        entailment_backend="lexical",
        llm_backend="extractive",
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        with app.state.session_factory() as db:
            _, key = create_workspace_with_key(db, "Tiny")
            db.commit()
        auth = {"X-API-Key": key}

        # Slightly over the file limit but under the body limit: caught while streaming to disk.
        resp = upload(client, auth, "big.txt", b"a" * (1024 * 1024 + 1))
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "payload_too_large"

        # Far over: rejected by Content-Length before the body is read.
        resp = upload(client, auth, "huge.txt", b"a" * (3 * 1024 * 1024))
        assert resp.status_code == 413
        assert not [p for p in settings.uploads_dir.rglob("*") if p.is_file()]


# ------------------------------------------------------- processing failures
@pytest.mark.parametrize(
    ("name", "data", "error_code"),
    [
        ("locked.pdf", encrypt_pdf(make_pdf([[LEAVE]]), "s3cret"), "encrypted"),
        ("scan.pdf", blank_pdf(2), "no_text"),
    ],
)
def test_unprocessable_documents_fail_with_actionable_message(client, auth, name, data, error_code):
    doc_id = upload(client, auth, name, data).json()["id"]
    doc = client.get(f"{API}/documents/{doc_id}", headers=auth).json()
    assert doc["status"] == "failed"
    assert doc["metadata"]["error_code"] == error_code
    assert doc["error"] and doc["chunk_count"] == 0


def test_unexpected_crash_is_reported_generically(client, auth, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("secret internals")

    monkeypatch.setattr("app.services.ingestion.pipeline.extract", boom)
    doc_id = upload(client, auth, "a.txt", b"Hello world.").json()["id"]
    doc = client.get(f"{API}/documents/{doc_id}", headers=auth).json()
    assert doc["status"] == "failed"
    assert "secret internals" not in doc["error"]


# ---------------------------------------------------------- list / paginate
def test_list_paginates_newest_first_and_filters_by_status(client, auth):
    ids = [upload(client, auth, f"d{i}.txt", f"Doc {i}.".encode()).json()["id"] for i in range(5)]
    upload(client, auth, "scan.pdf", blank_pdf())  # will fail

    page = client.get(f"{API}/documents?limit=2&offset=0", headers=auth).json()
    assert page["total"] == 6 and len(page["items"]) == 2

    ready = client.get(f"{API}/documents?status=ready&limit=100", headers=auth).json()
    assert {d["id"] for d in ready["items"]} == set(ids)
    failed = client.get(f"{API}/documents?status=failed", headers=auth).json()
    assert failed["total"] == 1


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "status=bogus"])
def test_list_validates_query(client, auth, query):
    assert client.get(f"{API}/documents?{query}", headers=auth).status_code == 422


def test_chunk_pagination(client, auth):
    text = " ".join(f"Sentence number {i} about leave policy." for i in range(400))
    doc_id = upload(client, auth, "long.txt", text.encode()).json()["id"]
    first = client.get(f"{API}/documents/{doc_id}/chunks?limit=2", headers=auth).json()
    second = client.get(f"{API}/documents/{doc_id}/chunks?limit=2&offset=2", headers=auth).json()
    assert first["total"] > 4
    assert [c["ordinal"] for c in first["items"]] == [0, 1]
    assert [c["ordinal"] for c in second["items"]] == [2, 3]


# ---------------------------------------------------- reprocess and delete
def test_reprocess_with_new_chunk_settings(client, auth, settings):
    text = " ".join(f"Sentence number {i} about leave policy." for i in range(200))
    doc_id = upload(client, auth, "long.txt", text.encode()).json()["id"]
    before = client.get(f"{API}/documents/{doc_id}", headers=auth).json()["chunk_count"]

    settings.chunk_size_tokens = 128
    settings.chunk_overlap_tokens = 16
    resp = client.post(f"{API}/documents/{doc_id}/reprocess", headers=auth)
    assert resp.status_code == 202
    after = client.get(f"{API}/documents/{doc_id}", headers=auth).json()
    assert after["status"] == "ready" and after["chunk_count"] > before
    chunks = client.get(f"{API}/documents/{doc_id}/chunks?limit=200", headers=auth).json()
    assert [c["ordinal"] for c in chunks["items"]] == list(range(after["chunk_count"]))


def test_reprocess_refused_while_processing(client, auth, app):
    doc_id = upload(client, auth, "a.txt", b"Hello.").json()["id"]
    with app.state.session_factory() as db:
        db.get(Document, doc_id).status = DocumentStatus.PROCESSING
        db.commit()
    assert client.post(f"{API}/documents/{doc_id}/reprocess", headers=auth).status_code == 409
    assert client.delete(f"{API}/documents/{doc_id}", headers=auth).status_code == 409


def test_delete_removes_row_chunks_and_file(client, auth, app, settings):
    doc_id = upload(client, auth, "h.pdf", sample_pdf()).json()["id"]
    assert client.delete(f"{API}/documents/{doc_id}", headers=auth).status_code == 204
    assert client.get(f"{API}/documents/{doc_id}", headers=auth).status_code == 404
    with app.state.session_factory() as db:
        assert db.query(Chunk).filter_by(document_id=doc_id).count() == 0
    assert not [p for p in settings.uploads_dir.rglob("*") if p.is_file()]
    # Deleting frees the hash, so the same file can be uploaded again.
    assert upload(client, auth, "h.pdf", sample_pdf()).status_code == 202


def test_unknown_and_malformed_ids(client, auth):
    assert client.get(f"{API}/documents/{uuid.uuid4()}", headers=auth).status_code == 404
    assert client.get(f"{API}/documents/not-a-uuid", headers=auth).status_code == 422


def test_restart_marks_interrupted_jobs_failed(tmp_path):
    settings = Settings(
        environment="test",
        data_dir=tmp_path,
        embedding_backend="hashing",
        reranker_backend="hashing",
        entailment_backend="lexical",
        llm_backend="extractive",
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        with app.state.session_factory() as db:
            _, key = create_workspace_with_key(db, "W")
            db.commit()
        doc_id = upload(client, {"X-API-Key": key}, "a.txt", b"Hi.").json()["id"]
        with app.state.session_factory() as db:
            db.get(Document, doc_id).status = DocumentStatus.PROCESSING
            db.commit()

    restarted = create_app(settings)  # same data dir = same database
    with TestClient(restarted) as client:
        doc = client.get(f"{API}/documents/{doc_id}", headers={"X-API-Key": key}).json()
        assert doc["status"] == "failed" and "interrupted" in doc["error"]
