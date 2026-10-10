"""Exercises the real SentenceTransformerEmbedder code path offline.

A tiny randomly-initialised BERT is built on disk, so these tests need torch
and sentence-transformers but no network. They check plumbing (device
selection, batching, normalisation, query/document encoding, tokenizer-based
counting, the context-length guard), not embedding quality — quality is the
benchmark's job, with the real model.

Skipped automatically when the ML extra isn't installed (e.g. the lean CI job).
"""

import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from app.core.config import Settings  # noqa: E402
from app.services.retrieval.embeddings import (  # noqa: E402
    EmbeddingConfigError,
    SentenceTransformerEmbedder,
    check_compatibility,
    resolve_device,
)

WORDS = "annual leave days claims travel receipts password rotate policy staff".split()


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from sentence_transformers import SentenceTransformer

    try:  # sentence-transformers >= 6
        from sentence_transformers.sentence_transformer import modules as models
    except ImportError:  # pragma: no cover - older versions
        from sentence_transformers import models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    root = tmp_path_factory.mktemp("tiny-st")
    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    config = BertConfig(
        vocab_size=len(WORDS) + 5,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    hf_dir = root / "hf"
    BertModel(config).save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)

    transformer = models.Transformer(str(hf_dir), max_seq_length=128)
    dim_getter = getattr(transformer, "get_embedding_dimension", None) or (
        transformer.get_word_embedding_dimension
    )
    pooling = models.Pooling(dim_getter(), pooling_mode="mean")
    st_dir = root / "st"
    SentenceTransformer(modules=[transformer, pooling]).save(str(st_dir))
    return str(st_dir)


@pytest.fixture(scope="module")
def embedder(tiny_model_dir):
    e = SentenceTransformerEmbedder(tiny_model_dir, device="auto", batch_size=4)
    assert not e.is_loaded  # lazy until first use
    e.load()
    return e


def test_device_resolution_prefers_accelerators():
    import torch

    device = resolve_device("auto")
    if torch.cuda.is_available():
        assert device == "cuda"
    elif torch.backends.mps.is_available():  # e.g. MacBook M-series
        assert device == "mps"
    else:
        assert device == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_loaded_model_metadata(embedder):
    info = embedder.describe()
    assert info["backend"] == "sentence-transformers"
    assert info["dimension"] == 32 == embedder.dimension
    assert info["max_tokens"] == 128 == embedder.max_tokens
    assert info["device"] in {"cpu", "mps", "cuda"}


def test_documents_are_batched_and_normalised(embedder):
    texts = [f"annual leave days {i}" for i in range(10)]  # > batch_size
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == 10 and all(len(v) == 32 for v in vectors)
    for v in vectors:
        assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-5)
    assert embedder.embed_documents([]) == []


def test_query_embedding_is_deterministic(embedder):
    assert embedder.embed_query("travel claims") == pytest.approx(
        embedder.embed_query("travel claims"), abs=1e-6
    )


def test_tokens_are_counted_with_the_model_tokenizer(embedder):
    assert embedder.count_tokens("annual leave days") == 3
    assert embedder.count_tokens("annual zzzz") == 2  # unknown word -> one [UNK]


def test_context_guard_uses_real_model_limit(embedder):
    with pytest.raises(EmbeddingConfigError):
        check_compatibility(
            embedder, Settings(chunk_size_tokens=100, chunk_overlap_tokens=8, _env_file=None)
        )
    check_compatibility(
        embedder, Settings(chunk_size_tokens=64, chunk_overlap_tokens=8, _env_file=None)
    )
