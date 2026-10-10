"""Vector indexing for a document's chunks.

Contextual embedding: each chunk is embedded together with a short header
(document title and section heading). A chunk reading "This applies to all
grades" means little alone; "Travel Policy — Allowances: This applies to all
grades" embeds near questions about travel allowances. The stored chunk text
is unchanged; only the embedding input gets the header.
"""

import logging
import time
from collections.abc import Sequence

from sqlalchemy import select

from app.db.models import Chunk, Document, DocumentStatus, Workspace
from app.services.container import Services
from app.services.retrieval.context import passage_with_context

logger = logging.getLogger(__name__)


def embedding_text(chunk: Chunk, document: Document) -> str:
    return passage_with_context(chunk, document)


def index_vectors(services: Services, document: Document, chunks: Sequence[Chunk]) -> float:
    """Replace the document's vectors. Returns elapsed milliseconds."""
    start = time.perf_counter()
    vectors = services.vectors
    vectors.delete_document(document.workspace_id, document.id)
    batch = max(services.settings.embedding_batch_size, 1) * 4
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        embeddings = services.embedder.embed_documents([embedding_text(c, document) for c in part])
        vectors.upsert(
            document.workspace_id,
            model=services.embedder.model_name,
            ids=[c.id for c in part],
            embeddings=embeddings,
            metadatas=[{"document_id": document.id, "ordinal": c.ordinal} for c in part],
        )
    return (time.perf_counter() - start) * 1000


def reindex_vectors(services: Services, workspace_id: str | None = None) -> tuple[int, int]:
    """Rebuild vectors for every ready document (e.g. after changing the model).

    Re-embeds the existing chunks; it does not re-extract or re-chunk. Use the
    document reprocess endpoint when chunking settings change too.
    Returns (documents, chunks) re-indexed.
    """
    n_docs = n_chunks = 0
    with services.session_factory() as db:
        query = select(Document).where(Document.status == DocumentStatus.READY)
        if workspace_id:
            query = query.where(Document.workspace_id == workspace_id)
        documents = db.scalars(query).all()
        # Drop whole collections first: they may hold vectors from the old model
        # for documents that are no longer ready.
        ws_ids = [workspace_id] if workspace_id else db.scalars(select(Workspace.id)).all()
        for ws in ws_ids:
            services.vectors.drop_workspace(ws)
        for document in documents:
            index_vectors(services, document, document.chunks)
            n_docs += 1
            n_chunks += len(document.chunks)
            meta = dict(document.meta or {})
            meta["embedding"] = services.embedder.describe()
            document.meta = meta
        db.commit()
    return n_docs, n_chunks
