"""Citation verification: keep only what the retrieved evidence actually supports.

The model returns JSON: an answer with [n] markers plus (passage, quote) pairs.
Nothing it says is trusted:

1. A citation is valid only if its passage number was given to the model AND its
   quote really occurs in that passage (compared word by word, so punctuation,
   case and whitespace differences don't matter). A near-exact quote that differs
   only in filler words ("the", "of") is accepted and labelled "fuzzy"; a changed
   number, negation or other content word is not.
2. Each answer sentence must carry at least one marker that points to a valid
   citation. Sentences without one are removed, never shown as fact.
3. If no sentence survives, the result is "insufficient evidence".

Source names, page numbers and offsets are attached later from the database by
passage number, so the model cannot invent a source or a page.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

_MARKER = re.compile(r"\[(\d{1,3})\]")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
# Sentence ends: . ! ? (optionally followed by markers/quotes/brackets) then space.
# Not before a lowercase letter, so "e.g. for" or "approx. two" don't split.
# A marker after the full stop ("... per year. [1]") stays with its sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])((?:\s*\[\d{1,3}\])*)[\"'”’)]*\s+(?=[^\sa-z\[])")
_MIN_QUOTE_WORDS = 2
_FUZZY_THRESHOLD = 0.75  # share of quote words that must match, in order
# Words whose presence or absence doesn't change a factual claim (English + Malay).
# Deliberately excludes negations (not, no, never, tidak, bukan) and all numbers.
_FILLER_WORDS = frozenset(
    "a an the of to in on at for by with from as and or is are was were be been "
    "that this these those it its their our your which who will shall may can "
    "yang dan di ke untuk dengan pada ini itu adalah ialah".split()
)


class ModelOutputError(ValueError):
    """The model's reply was not the JSON object we asked for."""


@dataclass(slots=True)
class ProposedCitation:
    passage: int
    quote: str


@dataclass(slots=True)
class ModelAnswer:
    answer: str
    insufficient_evidence: bool
    citations: list[ProposedCitation]


@dataclass(slots=True)
class VerifiedCitation:
    passage: int
    quote: str
    match: str  # "exact" | "fuzzy"
    offset: int | None  # character offset of the quote in the passage text (exact only)


@dataclass(slots=True)
class RejectedCitation:
    passage: int
    quote: str
    reason: str  # "unknown_passage" | "quote_not_found" | "quote_too_short"


@dataclass(slots=True)
class KeptSentence:
    text: str  # with its [n] markers
    passages: tuple[int, ...]  # verified passages it cites


@dataclass(slots=True)
class VerificationResult:
    answer: str  # only verified sentences, with markers renumbered-free (original numbers)
    citations: list[VerifiedCitation]
    rejected: list[RejectedCitation] = field(default_factory=list)
    sentences_total: int = 0
    sentences_removed: int = 0
    sentences: list[KeptSentence] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return bool(self.answer.strip()) and bool(self.citations)


# --------------------------------------------------------------------- parsing
def parse_model_output(raw: str) -> ModelAnswer:
    """Parse the JSON reply, tolerating code fences or text around the object."""
    data = _load_json_object(raw)
    if not isinstance(data, dict):
        raise ModelOutputError("reply is not a JSON object")
    answer = data.get("answer", "")
    insufficient = data.get("insufficient_evidence", False)
    raw_citations = data.get("citations", [])
    if not isinstance(answer, str) or not isinstance(raw_citations, list):
        raise ModelOutputError("reply has the wrong field types")
    citations = []
    for item in raw_citations[:50]:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("passage"))
        except (TypeError, ValueError):
            continue
        quote = item.get("quote")
        if isinstance(quote, str) and quote.strip():
            citations.append(ProposedCitation(passage=number, quote=quote.strip()[:1000]))
    return ModelAnswer(
        answer=answer.strip()[:4000],
        insufficient_evidence=bool(insufficient),
        citations=citations,
    )


def _load_json_object(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ModelOutputError("reply contains no JSON object")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ModelOutputError("reply contains malformed JSON") from exc


# ---------------------------------------------------------------- verification
def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return [w for w in _NON_WORD.split(text) if w]


def _find_quote(quote: str, passage: str) -> tuple[str, int | None] | None:
    """Return (match kind, offset) if the quote is supported by the passage."""
    q_words = normalize_words(quote)
    p_words = normalize_words(passage)
    if not q_words:
        return None
    q_str, p_str = " ".join(q_words), " ".join(p_words)
    if f" {q_str} " in f" {p_str} ":
        offset = passage.find(quote)
        if offset == -1:  # same words, different punctuation/case: locate case-insensitively
            offset = passage.casefold().find(quote.casefold())
        return "exact", (offset if offset >= 0 else None)
    # Near-exact: a small model may add or drop a filler word ("the", "of"). Accept
    # that, but only that: every content word, number and negation must line up, so
    # "14" -> "40" or "must be" -> "must not be" is still rejected. Short quotes
    # must match exactly.
    if len(q_words) < 5:
        return None
    matcher = SequenceMatcher(None, q_words, p_words, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return None
    matched_q = {i for b in blocks for i in range(b.a, b.a + b.size)}
    matched_p = {j for b in blocks for j in range(b.b, b.b + b.size)}
    span_start, span_end = blocks[0].b, blocks[-1].b + blocks[-1].size
    differences = [w for i, w in enumerate(q_words) if i not in matched_q] + [
        p_words[j] for j in range(span_start, span_end) if j not in matched_p
    ]
    if (
        len(matched_q) / len(q_words) >= _FUZZY_THRESHOLD
        and span_end - span_start <= 2 * len(q_words)
        and all(w in _FILLER_WORDS for w in differences)
    ):
        return "fuzzy", None
    return None


def split_sentences(answer: str) -> list[str]:
    """Split after sentence punctuation, keeping trailing [n] markers with their sentence."""
    parts: list[str] = []
    last = 0
    for m in _SENTENCE_BREAK.finditer(answer):
        end = m.start() + len(m.group(1))
        parts.append(answer[last:end].strip())
        last = m.end()
    tail = answer[last:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _keep_markers(good: set[int]):
    def replace(m: re.Match[str]) -> str:
        return m.group(0) if int(m.group(1)) in good else ""

    return replace


def verify(model: ModelAnswer, passages: dict[int, str]) -> VerificationResult:
    """Check every citation against the passages the model was actually given."""
    verified: dict[int, VerifiedCitation] = {}
    rejected: list[RejectedCitation] = []
    for c in model.citations:
        if c.passage not in passages:
            rejected.append(RejectedCitation(c.passage, c.quote, "unknown_passage"))
            continue
        if len(normalize_words(c.quote)) < _MIN_QUOTE_WORDS:
            rejected.append(RejectedCitation(c.passage, c.quote, "quote_too_short"))
            continue
        found = _find_quote(c.quote, passages[c.passage])
        if found is None:
            rejected.append(RejectedCitation(c.passage, c.quote, "quote_not_found"))
            continue
        kind, offset = found
        # Keep the best evidence per passage: exact beats fuzzy, longer beats shorter.
        best = verified.get(c.passage)
        if (
            best is None
            or (kind == "exact" and best.match == "fuzzy")
            or (kind == best.match and len(c.quote) > len(best.quote))
        ):
            verified[c.passage] = VerifiedCitation(c.passage, c.quote, kind, offset)

    sentences = split_sentences(model.answer) if not model.insufficient_evidence else []
    kept: list[KeptSentence] = []
    for sentence in sentences:
        markers = {int(n) for n in _MARKER.findall(sentence)}
        good = markers & verified.keys()
        if not good:
            continue  # no verified support: never present it as fact
        # Drop markers that point to unverified passages; keep the sentence.
        cleaned = _MARKER.sub(_keep_markers(good), sentence)
        text = re.sub(r"\s+([.!?,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()
        kept.append(KeptSentence(text, tuple(sorted(good))))

    result = VerificationResult(
        answer="",
        citations=[],
        rejected=rejected,
        sentences_total=len(sentences),
        sentences_removed=len(sentences) - len(kept),
    )
    return keep_only(result, kept, verified)


def keep_only(
    result: VerificationResult,
    sentences: list[KeptSentence],
    verified: dict[int, VerifiedCitation] | None = None,
) -> VerificationResult:
    """The result restricted to `sentences`; citations follow the markers still used."""
    by_passage = verified if verified is not None else {c.passage: c for c in result.citations}
    used = {n for s in sentences for n in s.passages}
    return VerificationResult(
        answer=" ".join(s.text for s in sentences),
        citations=[by_passage[n] for n in sorted(used)],
        rejected=result.rejected,
        sentences_total=result.sentences_total,
        sentences_removed=result.sentences_removed,
        sentences=sentences,
    )
