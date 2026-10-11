"""Question answering with verified citations, its history, and evidence lookup.

All endpoints are scoped to the caller's workspace. POST is used for questions
so their text never appears in URLs or proxy logs; the access log records only
the question's length.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError, NotFoundError
from app.core.security import WorkspaceContext, get_workspace, rate_limited
from app.db.models import Chunk, Document, DocumentStatus, QuestionLog
from app.db.session import get_db
from app.schemas.ask import (
    AskRequest,
    AskResponse,
    Citation,
    EntailmentOut,
    EvidenceContext,
    EvidenceOut,
    HistoryItem,
    HistoryList,
    RejectedCitationOut,
    SentenceCheckOut,
    SourceRef,
    Usage,
    VerificationOut,
)
from app.schemas.common import ErrorResponse
from app.services.generation.answer import AnswerOutcome, AnswerService, citation_location
from app.services.generation.llm import LLMError
from app.services.retrieval.search import SearchService
from app.services.retrieval.vector_store import IndexMismatchError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"], responses={401: {"model": ErrorResponse}})

Workspace = Annotated[WorkspaceContext, Depends(get_workspace)]
DB = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


class AnsweringDisabledError(AppError):
    status_code = 503
    code = "answering_disabled"


class GenerationError(AppError):
    def __init__(self, exc: LLMError) -> None:
        super().__init__(str(exc))
        self.status_code = exc.status_code
        self.code = exc.code


class IndexOutOfDateError(AppError):
    status_code = 409
    code = "index_out_of_date"


def _source(doc: Document, chunk: Chunk) -> SourceRef:
    return SourceRef(
        document_id=doc.id,
        filename=doc.filename,
        title=doc.title,
        file_type=doc.file_type,
        heading=chunk.heading,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
    )


def _to_response(question: str, outcome: AnswerOutcome, backend: str | None) -> AskResponse:
    by_number = {e.number: e for e in outcome.passages}
    citations = []
    if outcome.verification and outcome.answer:
        for c in outcome.verification.citations:
            e = by_number[c.passage]
            span = citation_location(e.hit.chunk, c.offset, c.quote)
            citations.append(
                Citation(
                    marker=c.passage,
                    chunk_id=e.hit.chunk.id,
                    quote=c.quote,
                    match=c.match,
                    quote_char_start=span[0] if span else None,
                    quote_char_end=span[1] if span else None,
                    source=_source(e.hit.document, e.hit.chunk),
                )
            )
    entailment = None
    if outcome.entailment is not None:
        ent = outcome.entailment
        entailment = EntailmentOut(
            model=ent.model,
            mode=ent.mode,
            threshold=ent.threshold,
            sentences_removed=ent.removed,
            sentences_flagged=ent.flagged,
            sentences=[
                SentenceCheckOut(text=c.text, score=c.score, supported=c.supported)
                for c in ent.sentences
            ],
        )
    verification = None
    if outcome.verification is not None:
        v = outcome.verification
        verification = VerificationOut(
            citations_proposed=outcome.citations_proposed,
            citations_verified=len(v.citations) if outcome.answer else 0,
            sentences_total=v.sentences_total,
            sentences_removed=v.sentences_removed,
            rejected=[
                RejectedCitationOut(marker=r.passage, quote=r.quote[:300], reason=r.reason)
                for r in v.rejected
            ],
            entailment=entailment,
        )
    return AskResponse(
        question=question,
        status=outcome.status,
        answer=outcome.answer,
        reason=outcome.reason,
        citations=citations,
        evidence=[
            EvidenceOut(
                marker=e.number,
                chunk_id=e.hit.chunk.id,
                text=e.hit.chunk.text,
                score=round(e.hit.score, 6),
                rerank_score=round(e.hit.rerank_score, 6)
                if e.hit.rerank_score is not None
                else None,
                suspicious=e.suspicious,
                source=_source(e.hit.document, e.hit.chunk),
            )
            for e in outcome.passages
        ],
        verification=verification,
        model=outcome.model,
        llm_backend=backend,
        embedding_model=outcome.embedding_model,
        reranker_model=outcome.reranker_model,
        usage=Usage(
            prompt_tokens=outcome.prompt_tokens, completion_tokens=outcome.completion_tokens
        ),
        warnings=outcome.warnings,
        timings_ms=outcome.timings_ms,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a question from this workspace's documents, with verified citations",
    description=(
        "Retrieves evidence, asks the local LLM to answer only from it, and verifies every "
        "citation against the retrieved passages. Unsupported sentences are removed; if "
        "nothing is supported the status is `insufficient_evidence`."
    ),
    responses={
        409: {"model": ErrorResponse, "description": "Vectors built with a different model"},
        503: {"model": ErrorResponse, "description": "LLM not running, model missing, or disabled"},
        504: {"model": ErrorResponse, "description": "The LLM timed out"},
        429: {"model": ErrorResponse, "description": "Rate limit reached; see Retry-After"},
    },
)
def ask(
    body: AskRequest,
    request: Request,
    workspace: Annotated[WorkspaceContext, Depends(rate_limited("ask"))],
    db: DB,
    settings: AppSettings,
) -> AskResponse:
    services = request.app.state.services
    if services.llm is None:
        raise AnsweringDisabledError("Question answering is disabled on this server.")
    search = SearchService(
        db,
        services.embedder,
        services.vectors,
        settings.fusion,
        reranker=services.reranker,
        rerank_candidates=settings.rerank_candidates,
    )
    service = AnswerService(
        db,
        settings,
        search,
        services.llm,
        services.embedder.count_tokens,
        entailment=services.entailment,
    )
    try:
        outcome = service.answer(
            workspace.id,
            body.question,
            passages=body.passages,
            document_ids=[str(d) for d in body.document_ids] if body.document_ids else None,
            mode=body.mode,
            rerank=body.rerank,
        )
    except LLMError as exc:
        logger.warning("ask_llm_error", extra={"code": exc.code})
        raise GenerationError(exc) from exc
    except IndexMismatchError as exc:
        raise IndexOutOfDateError(str(exc)) from exc

    response = _to_response(body.question, outcome, services.llm.backend)
    log = QuestionLog(
        workspace_id=workspace.id,
        question=body.question if settings.ask_history else None,
        answer=outcome.answer if settings.ask_history else None,
        status=outcome.status,
        reason=outcome.reason,
        model=outcome.model,
        latency_ms=outcome.timings_ms.get("total_ms", 0.0),
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        citations_proposed=outcome.citations_proposed,
        citations_verified=len(response.citations),
        detail=response.model_dump(mode="json") if settings.ask_history else {},
    )
    db.add(log)
    db.commit()
    response.id = log.id if settings.ask_history else None

    logger.info(
        "ask",
        extra={
            "question_chars": len(body.question),
            "status": outcome.status.value,
            "reason": outcome.reason,
            "citations_verified": len(response.citations),
            "passages": len(outcome.passages),
            **outcome.timings_ms,
        },
    )
    return response


@router.get(
    "/ask/history", response_model=HistoryList, summary="Recent questions in this workspace"
)
def history(
    workspace: Workspace,
    db: DB,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HistoryList:
    where = QuestionLog.workspace_id == workspace.id
    total = db.scalar(select(func.count()).select_from(QuestionLog).where(where)) or 0
    rows = db.scalars(
        select(QuestionLog)
        .where(where)
        .order_by(QuestionLog.created_at.desc(), QuestionLog.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return HistoryList(
        items=[
            HistoryItem(
                id=r.id,
                created_at=r.created_at,
                question=r.question,
                status=r.status,
                reason=r.reason,
                answer=r.answer,
                model=r.model,
                latency_ms=r.latency_ms,
                citations_verified=r.citations_verified,
            )
            for r in rows
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/ask/history/{entry_id}",
    response_model=AskResponse,
    summary="Replay a past answer with its citations",
    responses={404: {"model": ErrorResponse}},
)
def history_entry(entry_id: uuid.UUID, workspace: Workspace, db: DB) -> AskResponse:
    row = db.scalar(
        select(QuestionLog).where(
            QuestionLog.id == str(entry_id), QuestionLog.workspace_id == workspace.id
        )
    )
    if row is None or not row.detail:
        raise NotFoundError("History entry not found")
    return AskResponse.model_validate({**row.detail, "id": row.id})


@router.get(
    "/evidence/{chunk_id}",
    response_model=EvidenceContext,
    summary="Inspect a cited passage in its surrounding document text",
    responses={404: {"model": ErrorResponse}},
)
def evidence(
    chunk_id: uuid.UUID,
    workspace: Workspace,
    db: DB,
    context: Annotated[int, Query(ge=0, le=2000, description="Characters of context")] = 300,
) -> EvidenceContext:
    row = db.execute(
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.id == str(chunk_id),
            Chunk.workspace_id == workspace.id,
            Document.workspace_id == workspace.id,
            Document.status == DocumentStatus.READY,
        )
    ).first()
    if row is None:
        raise NotFoundError("Evidence not found")
    chunk, doc = row
    text = doc.extracted_text or ""
    return EvidenceContext(
        chunk_id=chunk.id,
        text=chunk.text,
        before=text[max(0, chunk.char_start - context) : chunk.char_start],
        after=text[chunk.char_end : chunk.char_end + context],
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        source=_source(doc, chunk),
    )
