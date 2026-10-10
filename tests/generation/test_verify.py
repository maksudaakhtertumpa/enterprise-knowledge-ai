import json

import pytest

from app.services.generation.verify import (
    ModelAnswer,
    ModelOutputError,
    ProposedCitation,
    normalize_words,
    parse_model_output,
    split_sentences,
    verify,
)

PASSAGES = {
    1: "Staff with less than two years of service receive 14 working days per year. "
    "Leave must be approved by the reporting manager.",
    2: "Travel claims are reimbursed within 30 days when original receipts are attached.",
}


def answer(text: str, *cites: tuple[int, str], insufficient: bool = False) -> ModelAnswer:
    return ModelAnswer(text, insufficient, [ProposedCitation(n, q) for n, q in cites])


# ------------------------------------------------------------------- parsing
def test_parse_plain_json():
    raw = json.dumps(
        {
            "answer": "14 days [1].",
            "insufficient_evidence": False,
            "citations": [{"passage": 1, "quote": "14 working days"}],
        }
    )
    parsed = parse_model_output(raw)
    assert parsed.answer == "14 days [1]."
    assert parsed.citations == [ProposedCitation(1, "14 working days")]


def test_parse_tolerates_fences_and_chatter():
    raw = (
        'Sure!\n```json\n{"answer": "x [1].", "insufficient_evidence": false, "citations": []}\n```'
    )
    assert parse_model_output(raw).answer == "x [1]."


def test_parse_skips_malformed_citation_items():
    raw = json.dumps(
        {
            "answer": "a",
            "insufficient_evidence": False,
            "citations": [
                {"passage": "2", "quote": "ok quote"},  # numeric string: accepted
                {"passage": "two", "quote": "bad"},
                {"quote": "no passage"},
                {"passage": 1, "quote": "   "},
                "junk",
            ],
        }
    )
    assert parse_model_output(raw).citations == [ProposedCitation(2, "ok quote")]


@pytest.mark.parametrize(
    "raw",
    ["", "no json here", "{not: json}", "[1, 2]", '{"answer": 5, "citations": []}'],
)
def test_parse_rejects_non_answers(raw):
    with pytest.raises(ModelOutputError):
        parse_model_output(raw)


# ----------------------------------------------------------------- sentences
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A per year. [1]", ["A per year. [1]"]),
        ("A [1]. B [2].", ["A [1].", "B [2]."]),
        ("A. [1] B. [2]", ["A. [1]", "B. [2]"]),
        ("Use e.g. the portal [1]. Then wait [2].", ["Use e.g. the portal [1].", "Then wait [2]."]),
        ("Ya [1]! Betul? [2]", ["Ya [1]!", "Betul? [2]"]),
        ("", []),
    ],
)
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_normalize_words_ignores_case_punctuation_and_unicode_forms():
    assert normalize_words("“Fourteen” WORKING-days…") == ["fourteen", "working", "days"]
    assert normalize_words("ﬁle") == ["file"]  # ligature folded


# -------------------------------------------------------------- verification
def test_exact_citation_verified_with_offset():
    result = verify(
        answer("You get 14 working days [1].", (1, "receive 14 working days")), PASSAGES
    )
    assert result.answer == "You get 14 working days [1]."
    (c,) = result.citations
    assert (c.passage, c.match) == (1, "exact")
    assert PASSAGES[1][c.offset : c.offset + len(c.quote)] == "receive 14 working days"


def test_quote_with_different_punctuation_and_case_still_exact():
    q = "LEAVE must be approved, by the reporting manager"
    (c,) = verify(answer("Approval needed [1].", (1, q)), PASSAGES).citations
    assert c.match == "exact"


def test_fuzzy_quote_accepted_and_labelled():
    # A filler word ("the") added to a long quote: near-exact.
    q = "Travel claims are reimbursed within 30 days when the original receipts are attached"
    (c,) = verify(answer("Paid within 30 days [2].", (2, q)), PASSAGES).citations
    assert c.match == "fuzzy" and c.offset is None


@pytest.mark.parametrize(
    "quote",
    [
        # Long quotes with one meaningful change must still be rejected.
        "Staff with less than two years of service receive 40 working days per year",
        "Staff with less than five years of service receive 14 working days per year",
        "Leave must not be approved by the reporting manager",
        "Travel claims are rejected within 30 days when original receipts are attached",
        "Travel claims are reimbursed within 30 days when receipts are attached",  # dropped word
    ],
)
def test_fuzzy_never_accepts_changed_facts(quote):
    passage = 1 if "Travel" not in quote else 2
    result = verify(answer(f"Claim [{passage}].", (passage, quote)), PASSAGES)
    assert not result.supported, quote
    assert result.rejected[0].reason == "quote_not_found"


@pytest.mark.parametrize(
    ("cite", "reason"),
    [
        ((7, "receive 14 working days"), "unknown_passage"),  # passage never shown
        ((1, "receive 40 working days"), "quote_not_found"),  # altered fact
        ((2, "receive 14 working days"), "quote_not_found"),  # right text, wrong passage
        ((1, "days"), "quote_too_short"),
    ],
)
def test_bad_citations_rejected_and_sentence_removed(cite, reason):
    result = verify(answer(f"You get the leave [{cite[0]}].", cite), PASSAGES)
    assert result.answer == "" and not result.supported
    assert [r.reason for r in result.rejected] == [reason]
    assert result.sentences_removed == 1


def test_unsupported_sentences_removed_supported_kept():
    model = answer(
        "You get 14 working days [1]. The CEO also gets a car. Claims take 30 days [2]. "
        "Bonuses are 3 months [3].",
        (1, "receive 14 working days"),
        (2, "reimbursed within 30 days"),
        (3, "Bonuses are 3 months"),
    )
    result = verify(model, PASSAGES)
    assert result.answer == "You get 14 working days [1]. Claims take 30 days [2]."
    assert result.sentences_total == 4 and result.sentences_removed == 2
    assert [c.passage for c in result.citations] == [1, 2]


def test_invalid_marker_stripped_from_otherwise_supported_sentence():
    model = answer(
        "Leave is 14 days [1][9].", (1, "14 working days per year"), (9, "anything here")
    )
    result = verify(model, PASSAGES)
    assert result.answer == "Leave is 14 days [1]."


def test_marker_without_quote_does_not_count():
    result = verify(answer("Claims take 30 days [2].", (1, "14 working days per year")), PASSAGES)
    assert not result.supported


def test_insufficient_flag_yields_no_answer_even_with_text():
    model = answer("Probably 14 days [1].", (1, "14 working days"), insufficient=True)
    result = verify(model, PASSAGES)
    assert result.answer == "" and result.sentences_total == 0


def test_exact_preferred_over_fuzzy_for_same_passage():
    model = answer(
        "Claims are paid [2].",
        (2, "Travel claims are reimbursed within 30 days when the original receipts are attached"),
        (2, "reimbursed within 30 days"),
    )
    (c,) = verify(model, PASSAGES).citations
    assert c.match == "exact" and c.quote == "reimbursed within 30 days"
