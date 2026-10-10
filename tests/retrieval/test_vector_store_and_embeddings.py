import math
import uuid

import pytest

from app.core.config import Settings
from app.services.retrieval.embeddings import (
    EmbeddingConfigError,
    HashingEmbedder,
    build_embedder,
    check_compatibility,
)
from app.services.retrieval.vector_store import IndexMismatchError, VectorStore, collection_name


@pytest.fixture
def store(tmp_path):
    vector_store = VectorStore(tmp_path / "chroma")
    yield vector_store
    vector_store.close()


def ws() -> str:
    return str(uuid.uuid4())


def unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def add(store, workspace, doc, vectors, model="m"):
    ids = [str(uuid.uuid4()) for _ in vectors]
    store.upsert(
        workspace,
        model=model,
        ids=ids,
        embeddings=vectors,
        metadatas=[{"document_id": doc, "ordinal": i} for i in range(len(vectors))],
    )
    return ids


# --------------------------------------------------------------- vector store
def test_query_orders_by_cosine_similarity(store):
    w = ws()
    near, far = add(store, w, "d", [unit(1, 0.1, 0), unit(0, 1, 0)])
    hits = store.query(w, model="m", embedding=unit(1, 0, 0), limit=5)
    assert [h.chunk_id for h in hits] == [near, far]
    assert hits[0].score == pytest.approx(0.995, abs=1e-3)
    assert hits[1].score == pytest.approx(0.0, abs=1e-6)


def test_query_limit_and_empty_workspace(store):
    w = ws()
    add(store, w, "d", [unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)])
    assert len(store.query(w, model="m", embedding=unit(1, 1, 1), limit=2)) == 2
    assert store.query(ws(), model="m", embedding=unit(1, 0, 0), limit=5) == []


def test_document_filter_and_delete(store):
    w = ws()
    add(store, w, "a", [unit(1, 0, 0)])
    (b,) = add(store, w, "b", [unit(1, 0, 0)])
    hits = store.query(w, model="m", embedding=unit(1, 0, 0), limit=5, document_ids=["b"])
    assert [h.chunk_id for h in hits] == [b]
    store.delete_document(w, "a")
    assert store.count(w) == 1
    store.delete_document(ws(), "a")  # unknown workspace: no-op


def test_collections_are_per_workspace(store):
    a, b = ws(), ws()
    add(store, a, "d", [unit(1, 0, 0)])
    assert store.query(b, model="m", embedding=unit(1, 0, 0), limit=5) == []
    assert collection_name(a) != collection_name(b)


def test_model_mismatch_is_refused(store):
    w = ws()
    add(store, w, "d", [unit(1, 0, 0)], model="model-a")
    with pytest.raises(IndexMismatchError, match="reindex"):
        store.query(w, model="model-b", embedding=unit(1, 0, 0), limit=5)
    with pytest.raises(IndexMismatchError):
        store.query(w, model="model-a", embedding=unit(1, 0, 0, 0), limit=5)  # wrong dimension


def test_drop_workspace(store):
    w = ws()
    add(store, w, "d", [unit(1, 0, 0)])
    store.drop_workspace(w)
    store.drop_workspace(w)  # idempotent
    assert store.count(w) == 0


def test_store_is_lazy(tmp_path):
    VectorStore(tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_close_is_idempotent_and_store_reopens(store):
    w = ws()
    add(store, w, "d", [unit(1, 0, 0)])
    store.close()
    store.close()
    assert store.count(w) == 1  # reopens lazily from disk


# ---------------------------------------------------------------- embeddings
def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    v1, v2 = e.embed_documents(["annual leave policy", "annual leave policy"])
    assert v1 == v2 and len(v1) == e.dimension
    assert math.sqrt(sum(x * x for x in v1)) == pytest.approx(1.0)
    assert e.embed_query("annual leave policy") == v1


def test_hashing_embedder_reflects_word_overlap_only():
    e = HashingEmbedder()
    q = e.embed_query("annual leave days")

    def cos(t: str) -> float:
        return sum(a * b for a, b in zip(q, e.embed_query(t), strict=True))

    assert cos("annual leave is fourteen days") > cos("travel claims are monthly")
    # A paraphrase with no shared words scores ~0: this embedder is NOT semantic.
    assert abs(cos("vacation entitlement")) < 0.2


def test_build_embedder_selects_backend():
    hashing = build_embedder(Settings(embedding_backend="hashing", _env_file=None))
    assert isinstance(hashing, HashingEmbedder)
    real = build_embedder(Settings(embedding_model="some/model", _env_file=None))
    assert real.model_name == "some/model" and not real.is_loaded  # lazy: nothing downloaded


class _ShortModel(HashingEmbedder):
    @property
    def max_tokens(self) -> int | None:
        return 256


def test_compatibility_check_rejects_chunks_longer_than_model():
    with pytest.raises(EmbeddingConfigError, match="EKA_CHUNK_SIZE_TOKENS"):
        check_compatibility(_ShortModel(), Settings(chunk_size_tokens=400, _env_file=None))
    check_compatibility(_ShortModel(), Settings(chunk_size_tokens=192, _env_file=None))
    check_compatibility(HashingEmbedder(), Settings(chunk_size_tokens=4096, _env_file=None))
