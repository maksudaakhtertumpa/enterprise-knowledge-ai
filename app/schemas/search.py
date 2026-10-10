import uuid

from pydantic import BaseModel, Field, field_validator

from app.db.models import FileType
from app.services.retrieval.search import SearchMode


class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=1000, examples=["How many days of annual leave?"]
    )
    mode: SearchMode | None = Field(
        None,
        description="`bm25` = keyword (SQLite FTS5); `dense` = semantic (embeddings + ChromaDB); "
        "`hybrid` = both, fused with Reciprocal Rank Fusion. Omit to use the server default "
        "(EKA_SEARCH_DEFAULT_MODE, `dense` out of the box).",
    )
    rerank: bool | None = Field(
        None,
        description="Reorder the top candidates with the cross-encoder reranker. "
        "Omit to use the server default (EKA_RERANK_BY_DEFAULT).",
    )
    top_k: int = Field(10, ge=1, description="Results to return (capped by server setting)")
    document_ids: list[uuid.UUID] | None = Field(
        None, max_length=100, description="Restrict the search to these documents"
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SourceInfo(BaseModel):
    document_id: str
    filename: str
    title: str | None
    file_type: FileType
    heading: str | None = Field(None, description="Section the chunk belongs to")
    page_start: int | None = Field(None, description="1-based; null for DOCX/TXT")
    page_end: int | None = None
    char_start: int = Field(..., description="Offset into the document's extracted text")
    char_end: int


class SearchResult(BaseModel):
    rank: int
    score: float = Field(
        ...,
        description="Higher is better. BM25: unbounded; dense: cosine similarity; "
        "hybrid: RRF score (only comparable within one result list); reranked: "
        "cross-encoder relevance in (0, 1).",
    )
    chunk_id: str
    text: str
    source: SourceInfo
    matched_terms: list[str] = Field(
        default_factory=list, description="BM25/hybrid: query terms found in this chunk"
    )
    retriever_ranks: dict[str, int] | None = Field(
        None,
        description="Hybrid only: rank in each retriever's list, e.g. {'bm25': 3, 'dense': 1}. "
        "A missing key means that retriever did not return the chunk.",
    )
    retriever_scores: dict[str, float] | None = Field(
        None, description="Hybrid only: raw score from each retriever"
    )
    retrieval_rank: int | None = Field(
        None, description="Reranked only: this result's rank before reranking"
    )


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    top_k: int
    reranked: bool = False
    embedding_model: str | None = None
    reranker_model: str | None = None
    timings_ms: dict[str, float]
    results: list[SearchResult]
