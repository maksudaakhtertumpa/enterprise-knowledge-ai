"""Search orchestration: retrieve -> hydrate -> (fuse) -> (rerank).

Retrievers (FTS5, Chroma) only return chunk IDs and scores. Text, source
metadata and permissions come from SQLite in a single query that filters on
the workspace *again* and on `status = ready` — so even a stale or
misbehaving index entry can't surface another workspace's text or a
half-processed document.

Hybrid mode runs both retrievers, removes anything hydration rejects, and
then fuses the two *visible* rankings with Reciprocal Rank Fusion. Filtering
before fusing matters: a stale ID must not occupy a rank position and push
real results down.

Reranking is an optional second stage on top of any mode: the first stage
returns `rerank_candidates` results and a cross-encoder reorders them. Each
reranked hit keeps its first-stage rank (`retrieval_rank`) so the effect of
reranking is visible per result.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentStatus
from app.services.retrieval.context import passage_with_context
from app.services.retrieval.embeddings import Embedder
from app.services.retrieval.fusion import reciprocal_rank_fusion
from app.services.retrieval.keyword import KeywordIndex
from app.services.retrieval.reranker import Reranker, RerankerUnavailableError
from app.services.retrieval.vector_store import VectorStore

# Fetch a few extra candidates so hits dropped during hydration (e.g. a
# document that is being re-processed) don't leave the page short.
_OVERFETCH = 10


class SearchMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class FusionConfig:
    rrf_k: int = 60
    # How deep each retriever's list goes before fusion. Deeper lists let a
    # chunk that is, say, 30th for BM25 but 2nd for dense still be credited.
    candidates: int = 50
    bm25_weight: float = 1.0
    dense_weight: float = 1.0


@dataclass(slots=True)
class SearchHit:
    rank: int
    score: float
    chunk: Chunk
    document: Document
    matched_terms: tuple[str, ...] = ()
    # Hybrid only: this chunk's rank and raw score in each retriever's list.
    retriever_ranks: dict[str, int] | None = None
    retriever_scores: dict[str, float] | None = None
    # Reranked searches only: cross-encoder score and the pre-rerank position.
    rerank_score: float | None = None
    retrieval_rank: int | None = None


@dataclass(slots=True)
class SearchOutcome:
    hits: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    embedding_model: str | None = None
    reranker_model: str | None = None


class SearchService:
    def __init__(
        self,
        db: Session,
        embedder: Embedder,
        vectors: VectorStore,
        fusion: FusionConfig | None = None,
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.vectors = vectors
        self.keywords = KeywordIndex(db)
        self.fusion = fusion or FusionConfig()
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    def search(
        self,
        workspace_id: str,
        query: str,
        *,
        mode: SearchMode,
        top_k: int,
        document_ids: Sequence[str] | None = None,
        rerank: bool = False,
    ) -> SearchOutcome:
        if rerank and self.reranker is None:
            raise RerankerUnavailableError("Reranking is disabled on this server.")
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        # How many first-stage results we need before any reranking.
        first_stage = max(top_k, self.rerank_candidates) if rerank else top_k
        depth = (
            max(self.fusion.candidates, first_stage) if mode is SearchMode.HYBRID else first_stage
        ) + _OVERFETCH

        # ---- retrieve -----------------------------------------------------------
        rankings: dict[str, list[tuple[str, float]]] = {}
        matched: dict[str, tuple[str, ...]] = {}
        model: str | None = None
        if mode in (SearchMode.BM25, SearchMode.HYBRID):
            t = time.perf_counter()
            keyword_hits = self.keywords.search(workspace_id, query, depth, document_ids)
            rankings["bm25"] = [(h.chunk_id, h.score) for h in keyword_hits]
            matched = {h.chunk_id: h.matched_terms for h in keyword_hits}
            timings["bm25_ms"] = _ms(t)
        if mode in (SearchMode.DENSE, SearchMode.HYBRID):
            t = time.perf_counter()
            vector = self.embedder.embed_query(query)
            timings["embed_ms"] = _ms(t)
            t = time.perf_counter()
            model = self.embedder.model_name
            vector_hits = self.vectors.query(
                workspace_id, model=model, embedding=vector, limit=depth, document_ids=document_ids
            )
            rankings["dense"] = [(h.chunk_id, h.score) for h in vector_hits]
            timings["dense_ms"] = _ms(t)

        # ---- hydrate (and drop anything not visible) ----------------------------
        t = time.perf_counter()
        all_ids = list(dict.fromkeys(cid for ranking in rankings.values() for cid, _ in ranking))
        rows = self._hydrate(workspace_id, all_ids)
        visible = {name: [(c, s) for c, s in r if c in rows] for name, r in rankings.items()}
        timings["hydrate_ms"] = _ms(t)

        # ---- rank or fuse --------------------------------------------------------
        hits: list[SearchHit] = []
        if mode is SearchMode.HYBRID:
            t = time.perf_counter()
            raw = {name: dict(r) for name, r in visible.items()}
            fused = reciprocal_rank_fusion(
                {name: [c for c, _ in r] for name, r in visible.items()},
                k=self.fusion.rrf_k,
                weights={"bm25": self.fusion.bm25_weight, "dense": self.fusion.dense_weight},
            )
            for item in fused[:first_stage]:
                chunk, document = rows[item.id]
                hits.append(
                    SearchHit(
                        rank=len(hits) + 1,
                        score=item.score,
                        chunk=chunk,
                        document=document,
                        matched_terms=matched.get(item.id, ()),
                        retriever_ranks=item.ranks,
                        retriever_scores={n: raw[n][item.id] for n in item.ranks},
                    )
                )
            timings["fuse_ms"] = _ms(t)
        else:
            (ranking,) = visible.values()
            for chunk_id, score in ranking[:first_stage]:
                chunk, document = rows[chunk_id]
                hits.append(
                    SearchHit(
                        rank=len(hits) + 1,
                        score=score,
                        chunk=chunk,
                        document=document,
                        matched_terms=matched.get(chunk_id, ()),
                    )
                )

        # ---- rerank ---------------------------------------------------------------
        reranker_model = None
        if rerank and hits:
            t = time.perf_counter()
            scores = self.reranker.score(
                query, [passage_with_context(h.chunk, h.document) for h in hits]
            )
            for hit, score in zip(hits, scores, strict=True):
                hit.retrieval_rank = hit.rank
                hit.rerank_score = score
                hit.score = score
            # Ties keep first-stage order, so reranking is deterministic.
            hits.sort(key=lambda h: (-h.score, h.retrieval_rank))
            for position, hit in enumerate(hits, start=1):
                hit.rank = position
            reranker_model = self.reranker.model_name
            timings["rerank_ms"] = _ms(t)

        timings["total_ms"] = _ms(t0)
        return SearchOutcome(
            hits=hits[:top_k],
            timings_ms=timings,
            embedding_model=model,
            reranker_model=reranker_model,
        )

    def _hydrate(
        self, workspace_id: str, chunk_ids: list[str]
    ) -> dict[str, tuple[Chunk, Document]]:
        if not chunk_ids:
            return {}
        rows = self.db.execute(
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.id.in_(chunk_ids),
                Chunk.workspace_id == workspace_id,
                Document.workspace_id == workspace_id,
                Document.status == DocumentStatus.READY,
            )
        ).all()
        return {chunk.id: (chunk, doc) for chunk, doc in rows}


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
