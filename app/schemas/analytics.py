from datetime import date, datetime

from pydantic import BaseModel, Field


class QuestionStats(BaseModel):
    total: int
    answered: int
    insufficient_evidence: int
    answer_rate: float | None = Field(None, description="answered / total; null when no questions")
    refusal_reasons: dict[str, int] = Field(
        default_factory=dict, description="Why questions got no answer, e.g. model_declined"
    )


class LatencyStats(BaseModel):
    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_ms: float


class CitationStats(BaseModel):
    proposed: int = Field(..., description="Citations the model proposed")
    verified: int = Field(..., description="Citations that passed verification and were shown")
    acceptance_rate: float | None = Field(
        None, description="verified / proposed; null when nothing was proposed"
    )
    avg_per_answer: float | None = Field(
        None, description="Verified citations per answered question"
    )


class TokenStats(BaseModel):
    prompt: int
    completion: int
    cost_usd: float = Field(0.0, description="Always 0: models run locally")


class DayCount(BaseModel):
    date: date
    answered: int
    insufficient_evidence: int


class DocumentStats(BaseModel):
    total: int
    by_status: dict[str, int]
    chunks: int = Field(..., description="Searchable chunks in ready documents")
    size_bytes: int


class AnalyticsResponse(BaseModel):
    """Usage of this workspace over a time window.

    These are operational numbers from real traffic, not quality metrics: a high
    answer rate is not the same as correct answers. Quality is measured offline
    with labelled questions (eval/README.md).
    """

    window_days: int
    since: datetime
    questions: QuestionStats
    latency: LatencyStats | None = Field(None, description="Null when no questions")
    citations: CitationStats
    tokens: TokenStats
    models: dict[str, int] = Field(default_factory=dict, description="Questions per LLM model")
    per_day: list[DayCount] = Field(
        default_factory=list, description="UTC days in the window, oldest first, including zeros"
    )
    documents: DocumentStats
