"""Sentence-level entailment: offline stand-in, helpers, wiring into answers."""

import pytest

from app.core.config import Settings
from app.services.generation.entailment import (
    LexicalEntailmentChecker,
    NLIEntailmentChecker,
    build_entailment_checker,
    strip_markers,
)
from app.services.generation.verify import KeptSentence, VerificationResult, keep_only

PREMISE = (
    "Staff with less than two years of service receive 14 working days of annual leave. "
    "Unused annual leave of up to 5 days may be carried forward."
)


def test_strip_markers():
    assert strip_markers("Staff get 14 days [1].") == "Staff get 14 days."
    assert strip_markers("A [1][2] and B [3]") == "A and B"


def test_lexical_stand_in_checks_numbers_negations_and_words():
    lex = LexicalEntailmentChecker()
    supported, wrong_number, negated, empty = lex.entailment(
        [
            (PREMISE, "Staff receive 14 working days of annual leave."),
            (PREMISE, "Staff receive 40 working days of annual leave."),
            (PREMISE, "Unused annual leave cannot be carried forward."),
            (PREMISE, "[1]"),
        ]
    )
    assert supported == 1.0
    assert wrong_number == 0.0  # 40 isn't in the premise
    assert negated == 0.0  # negation only on one side
    assert empty == 0.0
    assert lex.entailment([]) == []


def test_builder_respects_backend_and_mode():
    base = {"_env_file": None}
    assert build_entailment_checker(Settings(entailment_backend="none", **base)) is None
    assert build_entailment_checker(Settings(answer_entailment_mode="off", **base)) is None
    assert isinstance(
        build_entailment_checker(Settings(entailment_backend="lexical", **base)),
        LexicalEntailmentChecker,
    )
    nli = build_entailment_checker(Settings(**base))
    assert isinstance(nli, NLIEntailmentChecker) and not nli.is_loaded  # lazy: no download


@pytest.mark.parametrize(
    "bad", [{"answer_entailment_threshold": 1.5}, {"answer_entailment_mode": "warn"}]
)
def test_settings_are_validated(bad):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **bad)


def test_keep_only_rebuilds_answer_and_citations():
    from app.services.generation.verify import VerifiedCitation

    c1 = VerifiedCitation(1, "quote one", "exact", 0)
    c2 = VerifiedCitation(2, "quote two", "exact", 0)
    result = VerificationResult(
        answer="A [1]. B [2].",
        citations=[c1, c2],
        sentences_total=2,
        sentences=[KeptSentence("A [1].", (1,)), KeptSentence("B [2].", (2,))],
    )
    only_b = keep_only(result, [result.sentences[1]])
    assert only_b.answer == "B [2]." and only_b.citations == [c2]
    assert not keep_only(result, []).supported


# ------------------------------------------------------------ premise windows
def test_premise_windows_cover_runs_of_consecutive_sentences():
    from app.services.generation.entailment import premise_windows

    passage = "# Parking\n\nStaff park on B2. Visitors register at reception. EV bays are reserved."
    windows = premise_windows(passage, 2)
    assert windows[:4] == [
        "Parking",
        "Staff park on B2.",
        "Visitors register at reception.",
        "EV bays are reserved.",
    ]
    assert "Staff park on B2. Visitors register at reception." in windows
    assert len(windows) == 4 + 3  # singles + pairs
    assert premise_windows("", 3) == []


class RecordingChecker:
    """Scores 0.9 only when the premise is exactly the claim; records every pair."""

    model_name = "recording"
    is_loaded = True

    def __init__(self):
        self.pairs = []

    def load(self):
        return None

    def describe(self):
        return {}

    def entailment(self, pairs):
        self.pairs.extend(pairs)
        return [0.9 if p.strip() == h else 0.1 for p, h in pairs]


def test_score_claims_keeps_the_best_window_per_claim():
    from app.services.generation.entailment import score_claims

    checker = RecordingChecker()
    passages = ["Staff park on B2. Visitors register at reception.", "EV bays are reserved."]
    claims = [
        (passages, "Visitors register at reception."),  # one sentence of the first passage
        ([passages[1]], "EV bays are free."),  # nothing supports it
    ]
    assert score_claims(checker, claims, window=2) == [0.9, 0.1]
    assert len(checker.pairs) == (1 + 3 + 1) + 1  # whole text + windows, deduplicated

    whole_only = RecordingChecker()
    assert score_claims(whole_only, claims, granularity="passage") == [0.1, 0.1]
    assert len(whole_only.pairs) == 2
