"""The text a model sees for a chunk: document title + section heading + body.

Used for both embedding (Phase 3) and cross-encoder reranking (Phase 4), so
the two stages judge the same passage. A chunk saying "This applies to all
grades" means little alone; "Travel Policy — Allowances" in front of it tells
the model what it is about. The stored chunk text is never changed.
"""

from app.db.models import Chunk, Document


def passage_with_context(chunk: Chunk, document: Document) -> str:
    title = document.title or document.filename
    header = f"{title} — {chunk.heading}" if chunk.heading else title
    return f"{header}\n\n{chunk.text}"
