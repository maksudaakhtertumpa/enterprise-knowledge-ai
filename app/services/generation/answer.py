"""Grounded answering: retrieve -> gate -> prompt -> generate -> verify.

    question
      │  dense retrieval + cross-encoder rerank (the measured Phase 4 pipeline)
      ▼
    top passages ──► score gate ──(too weak)──► "insufficient evidence", no LLM call
      │
      ▼  nonce-fenced prompt, JSON-only reply
    local LLM (Ollama)
      │
      ▼  every citation checked against the passages it was given
    verified answer  ──(nothing survives)──► "insufficient evidence"

The response never contains a source the system didn't retrieve: citation
metadata (file, section, pages, offsets) is looked up by passage number.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import AnswerStatus, Chunk, Document
from app.services.generation.llm import LLMClient
from app.services.generation.prompt import (
    SYSTEM_PROMPT,
    PromptPassage,
    build_prompt,
    looks_like_injection,
)
from app.services.generation.verify import (
    ModelOutputError,
    VerificationResult,
    parse_model_output,
    verify,
)
from app.services.retrieval.search import SearchHit, SearchMode, SearchService

logger = logging.getLogger(__name__)

# Tokens kept free beyond the answer budget: chat template, fences, the question.
_PROMPT_MARGIN = 384


class Reason:
    NO_PASSAGES = "no_relevant_passages"
    LOW_SCORE = "low_retrieval_score"
    MODEL_DECLINED = "model_declined"
    NO_VERIFIED_CITATIONS = "no_verified_citations"
    INVALID_OUTPUT = "model_output_invalid"


@dataclass(slots=True)
class EvidencePassage:
    number: int
    hit: SearchHit
    suspicious: bool


@dataclass(slots=True)
class AnswerOutcome:
    status: AnswerStatus
    answer: str | None
    reason: str | None
    passages: list[EvidencePassage]
    verification: VerificationResult | None
    citations_proposed: int = 0
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    embedding_model: str | None = None
    reranker_model: str | None = None


class AnswerService:
    def __init__(
        self,
        db: Session,
        settings: Settings,
        search: SearchService,
        llm: LLMClient,
        count_tokens,
    ) -> None:
        self.db = db
        self.settings = settings
        self.search = search
        self.llm = llm
        self.count_tokens = count_tokens

    def answer(
        self,
        workspace_id: str,
        question: str,
        *,
        passages: int | None = None,
        document_ids: Sequence[str] | None = None,
        mode: SearchMode | None = None,
        rerank: bool | None = None,
    ) -> AnswerOutcome:
        s = self.settings
        t0 = time.perf_counter()
        n_passages = passages or s.answer_passages
        rerank = s.rerank_by_default if rerank is None else rerank

        # ---- 1. retrieve --------------------------------------------------------
        found = self.search.search(
            workspace_id,
            question,
            mode=mode or SearchMode(s.search_default_mode),
            top_k=n_passages,
            document_ids=document_ids,
            rerank=rerank and self.search.reranker is not None,
        )
        timings = {f"retrieval_{k}": v for k, v in found.timings_ms.items()}
        evidence, warnings = self._fit_budget(question, found.hits)

        base = {
            "passages": evidence,
            "embedding_model": found.embedding_model,
            "reranker_model": found.reranker_model,
            "warnings": warnings,
        }

        # ---- 2. gate: don't ask the model when retrieval found nothing useful ----
        if not evidence:
            return self._insufficient(Reason.NO_PASSAGES, t0, timings, None, **base)
        top_score = evidence[0].hit.rerank_score
        if top_score is not None and top_score < s.answer_min_rerank_score:
            return self._insufficient(Reason.LOW_SCORE, t0, timings, None, **base)

        # ---- 3. generate ----------------------------------------------------------
        prompt_passages = [
            PromptPassage(
                number=e.number,
                text=e.hit.chunk.text,
                title=e.hit.document.title or e.hit.document.filename,
                heading=e.hit.chunk.heading,
            )
            for e in evidence
        ]
        prompt = build_prompt(question, prompt_passages)
        t_gen = time.perf_counter()
        result = self.llm.generate(prompt, question, prompt_passages)
        timings["generate_ms"] = _ms(t_gen)
        if result.load_ms:
            timings["model_load_ms"] = result.load_ms
        usage = {
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }
        if result.prompt_tokens and result.prompt_tokens >= s.llm_num_ctx - s.llm_max_answer_tokens:
            warnings.append("prompt may have been truncated by the model's context window")
        if result.meta.get("done_reason") == "length":
            warnings.append("the model hit the answer length limit; the reply may be cut off")

        # ---- 4. verify --------------------------------------------------------------
        t_ver = time.perf_counter()
        try:
            parsed = parse_model_output(result.text)
        except ModelOutputError as exc:
            logger.warning("llm_output_invalid", extra={"error": str(exc)})
            timings["verify_ms"] = _ms(t_ver)
            return self._insufficient(Reason.INVALID_OUTPUT, t0, timings, None, **base, **usage)
        checked = verify(parsed, {p.number: p.text for p in prompt_passages})
        timings["verify_ms"] = _ms(t_ver)
        proposed = len(parsed.citations)

        if parsed.insufficient_evidence:
            return self._insufficient(
                Reason.MODEL_DECLINED, t0, timings, checked, proposed=proposed, **base, **usage
            )
        if not checked.supported:
            return self._insufficient(
                Reason.NO_VERIFIED_CITATIONS,
                t0,
                timings,
                checked,
                proposed=proposed,
                **base,
                **usage,
            )
        timings["total_ms"] = _ms(t0)
        return AnswerOutcome(
            status=AnswerStatus.ANSWERED,
            answer=checked.answer,
            reason=None,
            verification=checked,
            citations_proposed=proposed,
            timings_ms=timings,
            **base,
            **usage,
        )

    # ------------------------------------------------------------------ helpers
    def _fit_budget(
        self, question: str, hits: list[SearchHit]
    ) -> tuple[list[EvidencePassage], list[str]]:
        """Keep passages in rank order while they fit the model's context window."""
        s = self.settings
        budget = s.llm_num_ctx - s.llm_max_answer_tokens - _PROMPT_MARGIN
        budget -= self.count_tokens(SYSTEM_PROMPT) + self.count_tokens(question)
        evidence: list[EvidencePassage] = []
        warnings: list[str] = []
        used = 0
        for hit in hits:
            cost = self.count_tokens(hit.chunk.text) + 40  # fence + label
            if evidence and used + cost > budget:
                warnings.append(
                    f"{len(hits) - len(evidence)} passage(s) left out to fit the context window"
                )
                break
            used += cost
            evidence.append(
                EvidencePassage(
                    number=len(evidence) + 1,
                    hit=hit,
                    suspicious=looks_like_injection(hit.chunk.text),
                )
            )
        if any(e.suspicious for e in evidence):
            warnings.append("some evidence contains instruction-like text; it was treated as data")
        return evidence, warnings

    def _insufficient(
        self,
        reason: str,
        t0: float,
        timings: dict[str, float],
        checked: VerificationResult | None,
        *,
        proposed: int = 0,
        **kwargs: Any,
    ) -> AnswerOutcome:
        timings["total_ms"] = _ms(t0)
        return AnswerOutcome(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            answer=None,
            reason=reason,
            verification=checked,
            citations_proposed=proposed,
            timings_ms=timings,
            **kwargs,
        )


def citation_location(chunk: Chunk, quote_offset: int | None, quote: str) -> tuple[int, int] | None:
    """Absolute character span of a verified quote in the document's extracted text."""
    if quote_offset is None:
        return None
    start = chunk.char_start + quote_offset
    return start, start + len(quote)


def source_label(document: Document) -> str:
    return document.title or document.filename


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
