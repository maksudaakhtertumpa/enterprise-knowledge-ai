"""The real CrossEncoderReranker code path, offline, with a tiny random model.

Checks plumbing (loading, device and dtype selection, sigmoid scores, batching),
not ranking quality — quality is the benchmark's job with bge-reranker-v2-m3.
Skipped automatically without the ML extra.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

import torch  # noqa: E402

from app.services.retrieval.reranker import CrossEncoderReranker  # noqa: E402

WORDS = "annual leave days claims travel receipts password rotate policy staff".split()


@pytest.fixture(scope="module")
def tiny_cross_encoder(tmp_path_factory) -> str:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    root = tmp_path_factory.mktemp("tiny-ce")
    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    config = BertConfig(
        vocab_size=len(WORDS) + 5,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=1,  # one relevance score per (query, passage) pair
    )
    torch.manual_seed(0)
    BertForSequenceClassification(config).save_pretrained(root)
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(root)
    return str(root)


@pytest.fixture(scope="module")
def reranker(tiny_cross_encoder):
    r = CrossEncoderReranker(tiny_cross_encoder, device="auto", batch_size=2, max_length=64)
    assert not r.is_loaded
    r.load()
    return r


def test_scores_are_probabilities_one_per_passage(reranker):
    passages = ["annual leave days", "travel claims receipts", "rotate password", "staff policy"]
    scores = reranker.score("annual leave", passages)  # 4 passages, batch size 2
    assert len(scores) == 4
    assert all(0.0 < s < 1.0 for s in scores)
    assert reranker.score("annual leave", []) == []


def test_scores_are_deterministic(reranker):
    a = reranker.score("leave", ["annual leave days", "claims"])
    b = reranker.score("leave", ["annual leave days", "claims"])
    assert a == pytest.approx(b, abs=1e-5)


def test_device_and_precision_follow_hardware(reranker):
    info = reranker.describe()
    assert info["backend"] == "cross-encoder"
    assert info["device"] in {"cpu", "mps", "cuda"}
    # Half precision on GPUs (your M4 uses MPS -> float16), full precision on CPU.
    assert info["dtype"] == ("float32" if info["device"] == "cpu" else "float16")


def test_half_precision_kwarg_really_loads_float16(tiny_cross_encoder):
    """Guards the `dtype` model kwarg name, which changed across transformers versions."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(tiny_cross_encoder, device="cpu", model_kwargs={"dtype": torch.float16})
    assert next(model.model.parameters()).dtype == torch.float16


def test_full_precision_when_disabled(tiny_cross_encoder):
    r = CrossEncoderReranker(tiny_cross_encoder, device="cpu", half_precision=False)
    r.load()
    assert r.describe()["dtype"] == "float32"
