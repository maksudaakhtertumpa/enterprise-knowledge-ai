"""Sentence-level entailment: does the cited evidence actually support each claim?

Citation verification (verify.py) proves a quote exists in the passage the model
saw. It cannot tell whether the *sentence* around that quote follows from it:
the Phase 5 benchmark found an answer that cited a genuine quote about annual
leave but claimed it for a different leave type. A natural-language-inference
(NLI) model reads (premise, hypothesis) and predicts entailment / neutral /
contradiction, which is exactly that question:

    premise    = the passage(s) the sentence cites
    hypothesis = the sentence, without its [n] markers

Long premises: NLI models are trained on short premises and grow unsure as the
premise gets longer, even when it contains the claim word for word (measured in
Phase 7: a verbatim sentence from a 400-token passage scored 0.26). So, following
SummaC (Laban et al., 2022, "SummaC: Re-Visiting NLI-based Models for Inconsistency
Detection in Summarization"), each claim is scored against the whole cited text
*and* against every window of 1..N consecutive sentences in it, and keeps the best
score (`EKA_ENTAILMENT_GRANULARITY=sentences`, the default). A claim that no part of
its sources supports still scores low.

Backends:
- `nli`: a sentence-transformers CrossEncoder trained for NLI. Default
  `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT, ~280M
  parameters, multilingual, so Malay answers over English documents can be
  checked). Runs locally on MPS / CUDA / CPU, $0.
- `lexical`: offline stand-in for tests and CI. It checks that the sentence's
  content words and numbers appear in the premise. NOT semantic.
- `none`: no entailment check.
"""

import logging
import re
import threading
import unicodedata
from typing import Any, Protocol

from app.core.config import Settings
from app.services.generation.verify import split_sentences
from app.services.retrieval.embeddings import resolve_device

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\s*\[\d{1,3}\]")
_WORD = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)?$")
# Words that don't carry the claim (English + Malay); negations deliberately excluded.
_STOP = frozenset(
    "a an the of to in on at for by with from as and or is are was were be been being "
    "that this these those it its their our your which who will shall may can must "
    "has have had do does did per each any all some more most than then also only "
    "you we they he she yang dan di ke untuk dengan pada "
    "ini itu adalah ialah boleh akan".split()
)
_NEGATIONS = frozenset("not no never none without cannot tidak bukan tiada jangan".split())


class EntailmentUnavailableError(RuntimeError):
    pass


class EntailmentChecker(Protocol):
    model_name: str

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(entailment) in [0, 1] for each (premise, hypothesis) pair."""
        ...

    def describe(self) -> dict[str, Any]: ...


def strip_markers(sentence: str) -> str:
    return re.sub(r"\s+([.!?,;:])", r"\1", _MARKER.sub("", sentence)).strip()


def premise_windows(passage: str, size: int) -> list[str]:
    """Every run of 1..size consecutive sentences of the passage (headings count as lines)."""
    sentences = [
        sentence
        for line in re.split(r"\n+", passage)
        for sentence in split_sentences(line.strip().lstrip("#").strip())
        if sentence
    ]
    windows = []
    for n in range(1, size + 1):
        windows.extend(" ".join(sentences[i : i + n]) for i in range(len(sentences) - n + 1))
    return windows


def score_claims(
    checker: "EntailmentChecker",
    claims: list[tuple[list[str], str]],
    *,
    granularity: str = "sentences",
    window: int = 3,
) -> list[float]:
    """Best P(entailment) per (cited passages, claim), over the whole text and its windows.

    All pairs go to the model in one call, so batching covers every claim at once.
    """
    pairs: list[tuple[str, str]] = []
    owners: list[int] = []
    for i, (passages, hypothesis) in enumerate(claims):
        candidates = ["\n\n".join(passages)]
        if granularity == "sentences":
            for passage in passages:
                candidates.extend(premise_windows(passage, window))
        for premise in dict.fromkeys(c for c in candidates if c.strip()):
            pairs.append((premise, hypothesis))
            owners.append(i)
    best = [0.0] * len(claims)
    for i, score in zip(owners, checker.entailment(pairs), strict=True):
        best[i] = max(best[i], score)
    return best


class NLIEntailmentChecker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device_preference = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.device: str | None = None
        self.entailment_index: int | None = None
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - depends on install
                raise EntailmentUnavailableError(
                    "sentence-transformers is not installed. Install the ML extra or set "
                    "EKA_ENTAILMENT_BACKEND=none (or lexical for tests)."
                ) from exc
            self.device = resolve_device(self.device_preference)
            logger.info(
                "entailment_loading", extra={"model": self.model_name, "device": self.device}
            )
            # float32 on purpose: DeBERTa-v3 is known to overflow in float16.
            model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            labels = {
                int(i): str(name).lower() for i, name in (model.config.id2label or {}).items()
            }
            index = next((i for i, name in labels.items() if name.startswith("entail")), None)
            if index is None:
                raise EntailmentUnavailableError(
                    f"{self.model_name} is not an NLI model: no 'entailment' label in {labels}"
                )
            self.entailment_index = index
            self._model = model
            logger.info("entailment_loaded", extra=self.describe())

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "nli",
            "model": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
        }

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        if self._model is None:
            self.load()
        with self._infer_lock:
            probs = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                apply_softmax=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [float(row[self.entailment_index]) for row in probs]


def _content_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return [w for w in _WORD.findall(text) if w not in _STOP]


class LexicalEntailmentChecker:
    """Offline stand-in: share of the claim's content words found in the premise.

    Numbers must all appear, and a negation present on only one side fails.
    Deterministic and dependency-free, NOT semantic.
    """

    model_name = "lexical-v1"

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": "lexical", "model": self.model_name}

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for premise, hypothesis in pairs:
            p_words = set(_content_words(premise))
            h_words = _content_words(hypothesis)
            if not h_words:
                scores.append(0.0)
                continue
            numbers = [w for w in h_words if _NUMBER.match(w)]
            if any(n not in p_words for n in numbers):
                scores.append(0.0)
                continue
            if (_NEGATIONS & set(h_words)) and not (_NEGATIONS & p_words):
                scores.append(0.0)
                continue
            scores.append(round(sum(w in p_words for w in h_words) / len(h_words), 4))
        return scores


def build_entailment_checker(settings: Settings) -> EntailmentChecker | None:
    if settings.entailment_backend == "none" or settings.answer_entailment_mode == "off":
        return None
    if settings.entailment_backend == "lexical":
        return LexicalEntailmentChecker()
    return NLIEntailmentChecker(
        settings.entailment_model,
        device=settings.entailment_device,
        batch_size=settings.entailment_batch_size,
        max_length=settings.entailment_max_length,
    )
