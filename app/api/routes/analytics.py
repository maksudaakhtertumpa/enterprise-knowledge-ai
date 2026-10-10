"""Workspace usage analytics, computed on the server from the question log.

Only metrics columns are read (status, reason, latency, token and citation
counts), never question or answer text, so this works the same when
EKA_ASK_HISTORY is off.
"""

import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import WorkspaceContext, get_workspace
from app.db.models import AnswerStatus, Document, DocumentStatus, QuestionLog
from app.db.session import get_db
from app.schemas.analytics import (
    AnalyticsResponse,
    CitationStats,
    DayCount,
    DocumentStats,
    LatencyStats,
    QuestionStats,
    TokenStats,
)
from app.schemas.common import ErrorResponse

router = APIRouter(tags=["analytics"], responses={401: {"model": ErrorResponse}})


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (same definition as numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Usage analytics for this workspace (questions, refusals, latency, documents)",
)
def analytics(
    workspace: Annotated[WorkspaceContext, Depends(get_workspace)],
    db: Annotated[Session, Depends(get_db)],
    days: Annotated[int, Query(ge=1, le=365, description="Window size in days")] = 30,
) -> AnalyticsResponse:
    now = datetime.now(UTC)
    first_day = (now - timedelta(days=days - 1)).date()
    since = datetime.combine(first_day, datetime.min.time(), tzinfo=UTC)

    rows = db.execute(
        select(
            QuestionLog.created_at,
            QuestionLog.status,
            QuestionLog.reason,
            QuestionLog.model,
            QuestionLog.latency_ms,
            QuestionLog.prompt_tokens,
            QuestionLog.completion_tokens,
            QuestionLog.citations_proposed,
            QuestionLog.citations_verified,
        ).where(QuestionLog.workspace_id == workspace.id, QuestionLog.created_at >= since)
    ).all()

    answered = [r for r in rows if r.status is AnswerStatus.ANSWERED]
    reasons = Counter(r.reason or "unknown" for r in rows if r.status is not AnswerStatus.ANSWERED)
    latencies = [float(r.latency_ms) for r in rows]
    proposed = sum(r.citations_proposed for r in rows)
    verified = sum(r.citations_verified for r in rows)

    per_day = {first_day + timedelta(days=i): [0, 0] for i in range(days)}
    for r in rows:
        bucket = per_day.get(r.created_at.astimezone(UTC).date())
        if bucket is not None:
            bucket[0 if r.status is AnswerStatus.ANSWERED else 1] += 1

    doc_rows = db.execute(
        select(
            Document.status,
            func.count(),
            func.coalesce(func.sum(Document.chunk_count), 0),
            func.coalesce(func.sum(Document.size_bytes), 0),
        )
        .where(Document.workspace_id == workspace.id)
        .group_by(Document.status)
    ).all()
    by_status = {s.value: 0 for s in DocumentStatus}
    chunks = size = 0
    for status, count, chunk_sum, size_sum in doc_rows:
        by_status[status.value] = count
        size += size_sum
        if status is DocumentStatus.READY:
            chunks += chunk_sum

    return AnalyticsResponse(
        window_days=days,
        since=since,
        questions=QuestionStats(
            total=len(rows),
            answered=len(answered),
            insufficient_evidence=len(rows) - len(answered),
            answer_rate=_ratio(len(answered), len(rows)),
            refusal_reasons=dict(reasons.most_common()),
        ),
        latency=LatencyStats(
            p50_ms=round(percentile(latencies, 0.50), 1),
            p95_ms=round(percentile(latencies, 0.95), 1),
            mean_ms=round(sum(latencies) / len(latencies), 1),
            max_ms=round(max(latencies), 1),
        )
        if latencies
        else None,
        citations=CitationStats(
            proposed=proposed,
            verified=verified,
            acceptance_rate=_ratio(verified, proposed),
            avg_per_answer=round(verified / len(answered), 2) if answered else None,
        ),
        tokens=TokenStats(
            prompt=sum(r.prompt_tokens or 0 for r in rows),
            completion=sum(r.completion_tokens or 0 for r in rows),
        ),
        models=dict(Counter(r.model for r in rows if r.model).most_common()),
        per_day=[
            DayCount(date=d, answered=a, insufficient_evidence=i)
            for d, (a, i) in sorted(per_day.items())
        ],
        documents=DocumentStats(
            total=sum(by_status.values()), by_status=by_status, chunks=chunks, size_bytes=size
        ),
    )
