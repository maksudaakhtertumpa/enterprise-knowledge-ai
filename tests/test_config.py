import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("EKA_ENVIRONMENT", "production")
    monkeypatch.setenv("EKA_MAX_UPLOAD_MB", "50")
    monkeypatch.setenv("EKA_DATA_DIR", str(tmp_path))
    s = Settings(_env_file=None)
    assert s.environment == "production"
    assert s.max_upload_mb == 50
    assert s.uploads_dir == tmp_path / "uploads"


def test_list_env_var_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EKA_CORS_ORIGINS", '["http://a.test","http://b.test"]')
    assert Settings(_env_file=None).cors_origins == ["http://a.test", "http://b.test"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment", "staging"),
        ("max_upload_mb", 0),
        ("max_upload_mb", 500),
        ("api_prefix", "api/v1"),
        ("api_prefix", "/api/v1/"),
        ("log_level", "VERBOSE"),
        ("rrf_k", 0),
        ("hybrid_bm25_weight", -1),
    ],
)
def test_invalid_settings_fail_fast(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_hybrid_weights_cannot_both_be_zero() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hybrid_bm25_weight=0, hybrid_dense_weight=0)


def test_shipped_search_defaults_match_measured_decisions() -> None:
    """These defaults were chosen from M4 benchmarks (eval/README.md). Change them
    only together with new measurements."""
    s = Settings(_env_file=None)
    assert s.search_default_mode == "dense"
    assert s.rerank_by_default is True
    assert s.rerank_candidates == 10
    assert (s.rrf_k, s.hybrid_bm25_weight, s.hybrid_candidates) == (10, 0.25, 20)
