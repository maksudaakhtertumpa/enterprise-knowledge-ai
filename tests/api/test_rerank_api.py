"""Reranking through the search API, using the offline overlap reranker.

The overlap reranker scores a passage by the share of query words it contains,
so the expected order is predictable. Real cross-encoder quality is measured
by the benchmark on Apple silicon, not here.
"""

import pytest

from tests.factories import make_pdf

API = "/api/v1"


def upload(client, auth, name, data):
    resp = client.post(f"{API}/documents", headers=auth, files={"file": (name, data, "x")})
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


def search(client, auth, **body):
    return client.post(f"{API}/search", headers=auth, json=body)


@pytest.fixture
def corpus(client, auth):
    upload(client, auth, "it.txt", b"# Passwords\n\nRotate passwords every 90 days.\n")
    upload(client, auth, "leave.txt", b"# Leave\n\nAnnual leave is fourteen days per year.\n")
    upload(
        client,
        auth,
        "handbook.pdf",
        make_pdf([["Claims are reimbursed within thirty days with receipts."]], title="Handbook"),
    )


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_rerank_reports_scores_and_original_ranks(client, auth, corpus, mode):
    body = search(client, auth, query="rotate passwords 90 days", mode=mode, rerank=True).json()
    assert body["reranked"] is True
    assert body["reranker_model"] == "overlap-v1"
    assert "rerank_ms" in body["timings_ms"]
    results = body["results"]
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert {r["retrieval_rank"] for r in results} <= set(range(1, 21))
    assert results[0]["source"]["filename"] == "it.txt"


def test_reranker_can_promote_a_result_from_beyond_top_k(client, auth, corpus, settings):
    """First stage returns `rerank_candidates`; the reranker may pick any of them."""
    query = "annual leave fourteen days"
    first_stage = search(client, auth, query=query, mode="dense", top_k=3).json()["results"]
    reranked = search(client, auth, query=query, mode="dense", top_k=1, rerank=True).json()
    top = reranked["results"][0]
    assert len(reranked["results"]) == 1
    assert top["source"]["filename"] == "leave.txt"
    assert top["retrieval_rank"] == next(
        r["rank"] for r in first_stage if r["chunk_id"] == top["chunk_id"]
    )


def test_rerank_default_comes_from_settings(client, auth, corpus, settings):
    body = search(client, auth, query="passwords").json()
    assert body["reranked"] is False and body["reranker_model"] is None
    assert all(r["retrieval_rank"] is None for r in body["results"])
    settings.rerank_by_default = True
    assert search(client, auth, query="passwords").json()["reranked"] is True
    assert search(client, auth, query="passwords", rerank=False).json()["reranked"] is False


def test_rerank_candidates_setting_limits_the_pool(client, auth, corpus, settings):
    settings.rerank_candidates = 1
    body = search(client, auth, query="days", mode="bm25", top_k=1, rerank=True).json()
    plain = search(client, auth, query="days", mode="bm25", top_k=1).json()
    # With a pool of one, reranking can't change the winner.
    assert body["results"][0]["chunk_id"] == plain["results"][0]["chunk_id"]
    assert body["results"][0]["retrieval_rank"] == 1


def test_rerank_with_no_results_is_fine(client, auth):
    body = search(client, auth, query="anything", mode="bm25", rerank=True).json()
    assert body["results"] == [] and body["reranked"] is True


def test_rerank_requested_but_disabled_is_503(client, auth, corpus, app, monkeypatch):
    monkeypatch.setattr(app.state.services, "reranker", None)
    resp = search(client, auth, query="passwords", rerank=True)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "reranking_disabled"
    assert search(client, auth, query="passwords").status_code == 200  # plain search unaffected


def test_rerank_isolated_by_workspace(client, auth, other_auth, corpus):
    body = search(client, other_auth, query="rotate passwords", rerank=True).json()
    assert body["results"] == []


def test_readiness_reports_reranker(client):
    checks = {c["name"]: c for c in client.get("/health/ready").json()["checks"]}
    assert checks["reranker"] == {
        "name": "reranker",
        "ok": True,
        "detail": "overlap-v1",
        "required": True,
    }
