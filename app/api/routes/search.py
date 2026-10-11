"""Search endpoint, scoped to the caller's workspace.

POST (not GET) so the query text never lands in URLs, proxy logs or browser
history — search queries over internal documents can be sensitive. For the
same reason the access log records the query's length, not its content.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.security import WorkspaceContext, rate_limited
from app.db.session import get_db
from app.schemas.common import ErrorResponse
from app.schemas.search import SearchRequest, SearchResponse, SearchResult, SourceInfo
from app.services.retrieval.reranker import RerankerUnavailableError
from app.services.retrieval.search import SearchMode, SearchService
from app.services.retrieval.vector_store import IndexMismatchError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


class IndexOutOfDateError(AppError):
    status_code = 409
    code = "index_out_of_date"


class RerankingDisabledError(AppError):
    status_code = 503
    code = "reranking_disabled"


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search this workspace's documents (BM25, dense or hybrid; optional reranking)",
    responses={
        401: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "Vectors built with a different model"},
        503: {"model": ErrorResponse, "description": "Reranking requested but disabled"},
        429: {"model": ErrorResponse, "description": "Rate limit reached; see Retry-After"},
    },
)
def search(
    body: SearchRequest,
    request: Request,
    workspace: Annotated[WorkspaceContext, Depends(rate_limited("search"))],
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SearchResponse:
    services = request.app.state.services
    top_k = min(body.top_k, settings.search_max_top_k)
    mode = body.mode or SearchMode(settings.search_default_mode)
    rerank = settings.rerank_by_default if body.rerank is None else body.rerank
    service = SearchService(
        db,
        services.embedder,
        services.vectors,
        settings.fusion,
        reranker=services.reranker,
        rerank_candidates=settings.rerank_candidates,
    )
    try:
        outcome = service.search(
            workspace.id,
            body.query,
            mode=mode,
            top_k=top_k,
            document_ids=[str(d) for d in body.document_ids] if body.document_ids else None,
            rerank=rerank,
        )
    except IndexMismatchError as exc:
        raise IndexOutOfDateError(str(exc)) from exc
    except RerankerUnavailableError as exc:
        raise RerankingDisabledError(str(exc)) from exc

    logger.info(
        "search",
        extra={
            "mode": mode.value,
            "reranked": rerank,
            "query_chars": len(body.query),
            "top_k": top_k,
            "results": len(outcome.hits),
            **outcome.timings_ms,
        },
    )
    return SearchResponse(
        query=body.query,
        mode=mode,
        top_k=top_k,
        reranked=rerank,
        embedding_model=outcome.embedding_model,
        reranker_model=outcome.reranker_model,
        timings_ms=outcome.timings_ms,
        results=[
            SearchResult(
                rank=hit.rank,
                score=round(hit.score, 6),
                chunk_id=hit.chunk.id,
                text=hit.chunk.text,
                matched_terms=list(hit.matched_terms),
                retriever_ranks=hit.retriever_ranks,
                retriever_scores=(
                    {k: round(v, 6) for k, v in hit.retriever_scores.items()}
                    if hit.retriever_scores
                    else None
                ),
                retrieval_rank=hit.retrieval_rank,
                source=SourceInfo(
                    document_id=hit.document.id,
                    filename=hit.document.filename,
                    title=hit.document.title,
                    file_type=hit.document.file_type,
                    heading=hit.chunk.heading,
                    page_start=hit.chunk.page_start,
                    page_end=hit.chunk.page_end,
                    char_start=hit.chunk.char_start,
                    char_end=hit.chunk.char_end,
                ),
            )
            for hit in outcome.hits
        ],
    )
