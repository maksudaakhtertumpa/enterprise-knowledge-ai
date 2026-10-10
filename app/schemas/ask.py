import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.db.models import AnswerStatus, FileType
from app.services.retrieval.search import SearchMode


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=1000, examples=["How many days of annual leave do I get?"]
    )
    passages: int | None = Field(
        None, ge=1, le=10, description="Evidence passages given to the model (default: server)"
    )
    document_ids: list[uuid.UUID] | None = Field(
        None, max_length=100, description="Answer only from these documents"
    )
    mode: SearchMode | None = Field(None, description="Retrieval mode (default: server)")
    rerank: bool | None = Field(None, description="Rerank evidence (default: server)")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class SourceRef(BaseModel):
    document_id: str
    filename: str
    title: str | None
    file_type: FileType
    heading: str | None = None
    page_start: int | None = Field(None, description="1-based; null for DOCX/TXT")
    page_end: int | None = None


class Citation(BaseModel):
    marker: int = Field(..., description="The [n] used in the answer text")
    chunk_id: str
    quote: str = Field(..., description="Verified excerpt from the source passage")
    match: Literal["exact", "fuzzy"] = Field(
        ...,
        description="exact: found word for word; fuzzy: differs only in filler words "
        "such as 'the' or 'of' (numbers, negations and content words always match)",
    )
    quote_char_start: int | None = Field(
        None, description="Offset of the quote in the document's extracted text (exact only)"
    )
    quote_char_end: int | None = None
    source: SourceRef


class EvidenceOut(BaseModel):
    marker: int
    chunk_id: str
    text: str
    score: float
    rerank_score: float | None = None
    suspicious: bool = Field(
        False, description="Contains instruction-like text (treated as data, flagged for review)"
    )
    source: SourceRef


class RejectedCitationOut(BaseModel):
    marker: int
    quote: str
    reason: Literal["unknown_passage", "quote_not_found", "quote_too_short"]


class VerificationOut(BaseModel):
    citations_proposed: int
    citations_verified: int
    sentences_total: int
    sentences_removed: int = Field(..., description="Answer sentences dropped for lack of support")
    rejected: list[RejectedCitationOut] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float = Field(0.0, description="Always 0: models run locally")


class AskResponse(BaseModel):
    id: str | None = Field(None, description="History entry id (null if history is off)")
    question: str
    status: AnswerStatus
    answer: str | None = Field(None, description="Only verified sentences, with [n] markers")
    reason: str | None = Field(
        None,
        description="Why there is no answer: no_relevant_passages, low_retrieval_score, "
        "model_declined, no_verified_citations, model_output_invalid",
    )
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(
        default_factory=list, description="Every passage the model was shown"
    )
    verification: VerificationOut | None = None
    model: str | None = None
    llm_backend: str | None = None
    embedding_model: str | None = None
    reranker_model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class HistoryItem(BaseModel):
    id: str
    created_at: datetime
    question: str | None
    status: AnswerStatus
    reason: str | None
    answer: str | None
    model: str | None
    latency_ms: float
    citations_verified: int


class HistoryList(BaseModel):
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


class EvidenceContext(BaseModel):
    chunk_id: str
    text: str
    before: str = Field(..., description="Text just before the chunk in the document")
    after: str = Field(..., description="Text just after the chunk in the document")
    char_start: int
    char_end: int
    source: SourceRef
