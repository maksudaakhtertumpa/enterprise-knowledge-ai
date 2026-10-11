"""The real NLI CrossEncoder code path, offline, with a tiny random 3-label model.

Checks plumbing (loading, label lookup, softmax probabilities, batching), not
quality: quality is measured by the answer benchmark with the real model.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

import torch  # noqa: E402

from app.services.generation.entailment import (  # noqa: E402
    EntailmentUnavailableError,
    NLIEntailmentChecker,
)

WORDS = "annual leave days staff receive fourteen medical carried forward unused".split()


def _tiny(root, id2label):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]))
    config = BertConfig(
        vocab_size=len(WORDS) + 5,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    torch.manual_seed(0)
    BertForSequenceClassification(config).save_pretrained(root)
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(root)
    return str(root)


@pytest.fixture(scope="module")
def tiny_nli(tmp_path_factory):
    # Same label order as MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7.
    return _tiny(
        tmp_path_factory.mktemp("nli"), {0: "entailment", 1: "neutral", 2: "contradiction"}
    )


def test_returns_entailment_probabilities(tiny_nli):
    checker = NLIEntailmentChecker(tiny_nli, device="cpu", batch_size=2, max_length=64)
    assert not checker.is_loaded
    pairs = [
        ("staff receive fourteen annual leave days", "staff receive fourteen days"),
        ("staff receive fourteen annual leave days", "medical leave carried forward"),
        ("unused leave carried forward", "unused leave carried forward"),
    ]
    scores = checker.entailment(pairs)  # loads lazily; 3 pairs, batch size 2
    assert checker.is_loaded and checker.entailment_index == 0
    assert len(scores) == 3 and all(0.0 < s < 1.0 for s in scores)
    assert checker.entailment(pairs) == scores  # deterministic
    assert checker.entailment([]) == []
    assert checker.describe()["model"] == tiny_nli


def test_finds_the_entailment_label_wherever_it_is(tmp_path):
    path = _tiny(tmp_path, {0: "contradiction", 1: "ENTAILMENT", 2: "neutral"})
    checker = NLIEntailmentChecker(path, device="cpu")
    checker.load()
    assert checker.entailment_index == 1


def test_refuses_a_model_without_an_entailment_label(tmp_path):
    path = _tiny(tmp_path, {0: "negative", 1: "positive"})
    with pytest.raises(EntailmentUnavailableError, match="not an NLI model"):
        NLIEntailmentChecker(path, device="cpu").load()
