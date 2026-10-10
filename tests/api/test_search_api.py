"""Search API: source metadata, both modes, filters, isolation and index consistency.

These use the offline hashing embedder, which matches on shared words. That's
enough to test plumbing (indexing, filters, isolation, lifecycle). Semantic
quality is measured by the benchmark with a real model, not here.
"""

import pytest

from app.db.models import Document, DocumentStatus
from tests.factories import make_docx, make_pdf

API = "/api/v1"
LEAVE = (
    "Employees are entitled to fourteen days of annual leave per calendar year. "
    "Leave must be approved by the line manager at least two weeks in advance."
)
CLAIMS = "Travel claims are reimbursed within thirty days when original receipts are attached."
MODES = ["bm25", "dense", "hybrid"]


def upload(client, auth, name, data):
    resp = client.post(f"{API}/documents", headers=auth, files={"file": (name, data, "x")})
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


def search(client, auth, query, mode="bm25", **extra):
    return client.post(f"{API}/search", headers=auth, json={"query": query, "mode": mode, **extra})


@pytest.fixture
def corpus(client, auth):
    return {
        "pdf": upload(
            client,
            auth,
            "handbook.pdf",
            make_pdf(
                [["Leave Policy", LEAVE], ["Claims", CLAIMS]],
                outline={1: "Leave", 2: "Claims"},
                title="Staff Handbook",
            ),
        ),
        "docx": upload(
            client,
            auth,
            "remote.docx",
            make_docx([("h1", "Remote Work"), ("p", "Staff may work from home two days a week.")]),
        ),
        "txt": upload(client, auth, "it.txt", b"# Passwords\n\nRotate passwords every 90 days.\n"),
    }


@pytest.mark.parametrize("mode", MODES)
def test_search_returns_ranked_results_with_sources(client, auth, corpus, mode):
    resp = search(client, auth, "travel claims receipts reimbursed", mode)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == mode
    top = body["results"][0]
    assert top["rank"] == 1
    assert "receipts" in top["text"]
    assert top["source"] == {
        "document_id": corpus["pdf"],
        "filename": "handbook.pdf",
        "title": "Staff Handbook",
        "file_type": "pdf",
        "heading": "Claims",
        "page_start": 2,
        "page_end": 2,
        "char_start": top["source"]["char_start"],
        "char_end": top["source"]["char_end"],
    }
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert set(body["timings_ms"]) >= {"hydrate_ms", "total_ms"}


def test_dense_reports_embedding_model_and_timing(client, auth, corpus):
    body = search(client, auth, "passwords", "dense").json()
    assert body["embedding_model"] == "hashing-v1"
    assert "embed_ms" in body["timings_ms"]


def test_bm25_reports_matched_terms(client, auth, corpus):
    top = search(client, auth, "rotate passwords", "bm25").json()["results"][0]
    assert top["source"]["filename"] == "it.txt"
    assert top["source"]["page_start"] is None and top["source"]["heading"] == "Passwords"
    assert set(top["matched_terms"]) == {"rotate", "passwords"}


def test_char_offsets_point_into_extracted_text(client, auth, app, corpus):
    top = search(client, auth, "annual leave", "bm25").json()["results"][0]
    with app.state.session_factory() as db:
        text = db.get(Document, top["source"]["document_id"]).extracted_text
    assert text[top["source"]["char_start"] : top["source"]["char_end"]] == top["text"]


@pytest.mark.parametrize("mode", MODES)
def test_document_filter(client, auth, corpus, mode):
    body = search(client, auth, "days", mode, document_ids=[corpus["docx"]]).json()
    assert body["results"]
    assert {r["source"]["document_id"] for r in body["results"]} == {corpus["docx"]}


def test_top_k_is_respected_and_capped(client, auth, corpus, settings):
    assert len(search(client, auth, "days", "dense", top_k=1).json()["results"]) == 1
    settings.search_max_top_k = 2
    body = search(client, auth, "days", "dense", top_k=50).json()
    assert body["top_k"] == 2 and len(body["results"]) <= 2


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 1001},
        {"query": "ok", "mode": "rerank"},
        {"query": "ok", "top_k": 0},
        {"query": "ok", "document_ids": ["not-a-uuid"]},
        {},
    ],
)
def test_invalid_requests_are_422(client, auth, payload):
    assert client.post(f"{API}/search", headers=auth, json=payload).status_code == 422


def test_search_requires_api_key(client):
    assert client.post(f"{API}/search", json={"query": "x"}).status_code == 401


def test_empty_workspace_returns_no_results(client, auth):
    for mode in MODES:
        assert search(client, auth, "anything", mode).json()["results"] == []


def test_bm25_query_with_no_usable_terms(client, auth, corpus):
    assert search(client, auth, "?!?", "bm25").json()["results"] == []


# ------------------------------------------------------------------ isolation
@pytest.mark.parametrize("mode", MODES)
def test_other_workspace_cannot_find_our_documents(client, auth, other_auth, corpus, mode):
    assert search(client, other_auth, "annual leave travel claims", mode).json()["results"] == []
    # Even when it names our document IDs explicitly.
    body = search(client, other_auth, "leave", mode, document_ids=[corpus["pdf"]]).json()
    assert body["results"] == []


# -------------------------------------------------------- index consistency
@pytest.mark.parametrize("mode", MODES)
def test_deleted_document_disappears_from_both_indexes(client, auth, app, corpus, mode):
    assert client.delete(f"{API}/documents/{corpus['txt']}", headers=auth).status_code == 204
    hits = search(client, auth, "rotate passwords", mode).json()["results"]
    assert all(r["source"]["document_id"] != corpus["txt"] for r in hits)
    services = app.state.services
    ws_id = client.get(f"{API}/workspace", headers=auth).json()["id"]
    total_chunks = sum(
        client.get(f"{API}/documents/{d}", headers=auth).json()["chunk_count"]
        for d in (corpus["pdf"], corpus["docx"])
    )
    assert services.vectors.count(ws_id) == total_chunks  # no orphan vectors


def test_reprocess_does_not_duplicate_index_entries(client, auth, app, corpus):
    before = search(client, auth, "annual leave", "bm25", top_k=50).json()["results"]
    assert (
        client.post(f"{API}/documents/{corpus['pdf']}/reprocess", headers=auth).status_code == 202
    )
    after = search(client, auth, "annual leave", "bm25", top_k=50).json()["results"]
    assert len(after) == len(before)
    assert {r["chunk_id"] for r in after}.isdisjoint({r["chunk_id"] for r in before})
    ws_id = client.get(f"{API}/workspace", headers=auth).json()["id"]
    chunks = sum(
        client.get(f"{API}/documents/{d}", headers=auth).json()["chunk_count"]
        for d in corpus.values()
    )
    assert app.state.services.vectors.count(ws_id) == chunks


@pytest.mark.parametrize("mode", MODES)
def test_documents_not_ready_are_invisible(client, auth, app, corpus, mode):
    with app.state.session_factory() as db:
        db.get(Document, corpus["txt"]).status = DocumentStatus.PROCESSING
        db.commit()
    hits = search(client, auth, "rotate passwords", mode).json()["results"]
    assert all(r["source"]["document_id"] != corpus["txt"] for r in hits)


def test_embedding_failure_marks_document_failed_and_unsearchable(client, auth, app, monkeypatch):
    def broken(_texts):
        raise RuntimeError("GPU on fire")

    monkeypatch.setattr(app.state.services.embedder, "embed_documents", broken)
    doc_id = upload(client, auth, "x.txt", b"Unique zebra crossing policy.")
    doc = client.get(f"{API}/documents/{doc_id}", headers=auth).json()
    assert doc["status"] == "failed"
    assert doc["metadata"]["error_code"] == "indexing_failed"
    assert "GPU" not in doc["error"]
    assert search(client, auth, "zebra crossing", "bm25").json()["results"] == []


def test_changed_embedding_model_requires_reindex(client, auth, app, corpus, monkeypatch):
    monkeypatch.setattr(app.state.services.embedder, "model_name", "another-model")
    resp = search(client, auth, "leave", "dense")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "index_out_of_date"
    # Keyword search is unaffected.
    assert search(client, auth, "leave", "bm25").status_code == 200


def test_reindex_rebuilds_vectors_for_new_model(client, auth, app, corpus, monkeypatch):
    from app.services.retrieval.indexing import reindex_vectors

    monkeypatch.setattr(app.state.services.embedder, "model_name", "another-model")
    docs, chunks = reindex_vectors(app.state.services)
    assert docs == 3 and chunks > 0
    assert search(client, auth, "leave", "dense").status_code == 200


def test_document_metadata_records_embedding_model(client, auth, corpus):
    doc = client.get(f"{API}/documents/{corpus['pdf']}", headers=auth).json()
    assert doc["metadata"]["embedding"]["model"] == "hashing-v1"
    assert doc["metadata"]["chunking"]["tokenizer"] == "hashing-v1"


def test_readiness_includes_vector_store_and_model(client):
    checks = {c["name"]: c for c in client.get("/health/ready").json()["checks"]}
    assert checks["vector_store"]["ok"] and checks["embedding_model"]["ok"]


# ----------------------------------------------------------------- hybrid / RRF
def test_hybrid_explains_each_result(client, auth, corpus, settings):
    body = search(client, auth, "travel claims receipts", "hybrid", top_k=5).json()
    assert body["embedding_model"] == "hashing-v1"
    assert {"bm25_ms", "embed_ms", "dense_ms", "fuse_ms"} <= set(body["timings_ms"])
    for r in body["results"]:
        ranks = r["retriever_ranks"]
        assert ranks and set(ranks) <= {"bm25", "dense"}
        assert set(r["retriever_scores"]) == set(ranks)
        # The reported score is exactly the weighted RRF formula over the reported ranks.
        weights = {"bm25": settings.hybrid_bm25_weight, "dense": settings.hybrid_dense_weight}
        expected = sum(weights[name] / (settings.rrf_k + rank) for name, rank in ranks.items())
        assert r["score"] == pytest.approx(expected, abs=1e-6)
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_top_hit_found_by_both_retrievers(client, auth, corpus):
    top = search(client, auth, "rotate passwords every 90 days", "hybrid").json()["results"][0]
    assert top["source"]["filename"] == "it.txt"
    assert set(top["retriever_ranks"]) == {"bm25", "dense"}


def test_hybrid_still_answers_when_bm25_finds_nothing(client, auth, corpus):
    body = search(client, auth, "zzqx quux", "hybrid").json()
    assert body["results"], "dense alone should still contribute"
    assert all(set(r["retriever_ranks"]) == {"dense"} for r in body["results"])


def test_hybrid_ranks_count_only_visible_chunks(client, auth, app, corpus):
    """A stale index entry must not occupy a rank and push real results down."""
    with app.state.session_factory() as db:
        db.get(Document, corpus["pdf"]).status = DocumentStatus.PROCESSING
        db.commit()
    results = search(client, auth, "days", "hybrid", top_k=10).json()["results"]
    assert results and all(r["source"]["document_id"] != corpus["pdf"] for r in results)
    for retriever in ("bm25", "dense"):
        ranks = sorted(
            r["retriever_ranks"][retriever] for r in results if retriever in r["retriever_ranks"]
        )
        assert ranks[0] == 1


def test_hybrid_weights_and_k_come_from_settings(client, auth, corpus, settings):
    bm25_order = [
        r["chunk_id"] for r in search(client, auth, "days leave", "bm25").json()["results"]
    ]
    settings.hybrid_dense_weight = 0.0
    hybrid = search(client, auth, "days leave", "hybrid").json()["results"]
    assert [r["chunk_id"] for r in hybrid][: len(bm25_order)] == bm25_order
    settings.hybrid_dense_weight = 1.0
    settings.hybrid_bm25_weight = 1.0
    settings.rrf_k = 33
    top = search(client, auth, "days leave", "hybrid").json()["results"][0]
    assert top["score"] == pytest.approx(
        sum(1 / (33 + r) for r in top["retriever_ranks"].values()), abs=1e-6
    )


def test_default_mode_comes_from_settings(client, auth, corpus, settings):
    body = client.post(f"{API}/search", headers=auth, json={"query": "passwords"}).json()
    assert body["mode"] == "dense"  # measured best on the M4 benchmark
    settings.search_default_mode = "bm25"
    body = client.post(f"{API}/search", headers=auth, json={"query": "passwords"}).json()
    assert body["mode"] == "bm25"


def test_hybrid_also_refuses_mismatched_vectors(client, auth, app, corpus, monkeypatch):
    monkeypatch.setattr(app.state.services.embedder, "model_name", "another-model")
    assert search(client, auth, "leave", "hybrid").status_code == 409
