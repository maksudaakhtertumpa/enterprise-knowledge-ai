from app.core.config import Settings
from app.services.retrieval.reranker import (
    CrossEncoderReranker,
    HashingReranker,
    build_reranker,
)


def test_overlap_reranker_scores_share_of_query_terms():
    r = HashingReranker()
    assert r.score("annual leave days", ["annual leave is 14 days", "claims", "leave"]) == [
        1.0,
        0.0,
        1 / 3,
    ]
    assert r.score("", ["anything"]) == [0.0]
    assert r.score("q", []) == []


def test_build_reranker_selects_backend():
    assert build_reranker(Settings(reranker_backend="none", _env_file=None)) is None
    assert isinstance(
        build_reranker(Settings(reranker_backend="hashing", _env_file=None)), HashingReranker
    )
    real = build_reranker(Settings(reranker_model="some/model", _env_file=None))
    assert isinstance(real, CrossEncoderReranker)
    assert real.model_name == "some/model" and not real.is_loaded  # lazy: nothing downloaded
