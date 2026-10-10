"""Workspace-scoped data access for documents and chunks.

Every method takes `workspace_id` as a required argument and filters on it.
There is deliberately no "get by id" without a workspace: a document in
another workspace is indistinguishable from one that doesn't exist.
"""

from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentStatus


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, workspace_id: str, document_id: str) -> Document | None:
        return self.db.scalar(
            select(Document).where(
                Document.id == document_id, Document.workspace_id == workspace_id
            )
        )

    def find_by_hash(self, workspace_id: str, sha256: str) -> Document | None:
        return self.db.scalar(
            select(Document).where(Document.workspace_id == workspace_id, Document.sha256 == sha256)
        )

    def list_documents(
        self,
        workspace_id: str,
        *,
        limit: int,
        offset: int,
        status: DocumentStatus | None = None,
    ) -> tuple[Sequence[Document], int]:
        conditions = [Document.workspace_id == workspace_id]
        if status is not None:
            conditions.append(Document.status == status)
        total = self.db.scalar(select(func.count()).select_from(Document).where(*conditions))
        items = self.db.scalars(
            select(Document)
            .where(*conditions)
            .order_by(Document.created_at.desc(), Document.id)
            .limit(limit)
            .offset(offset)
        ).all()
        return items, total or 0

    def list_chunks(
        self, workspace_id: str, document_id: str, *, limit: int, offset: int
    ) -> tuple[Sequence[Chunk], int]:
        conditions = [Chunk.workspace_id == workspace_id, Chunk.document_id == document_id]
        total = self.db.scalar(select(func.count()).select_from(Chunk).where(*conditions))
        items = self.db.scalars(
            select(Chunk).where(*conditions).order_by(Chunk.ordinal).limit(limit).offset(offset)
        ).all()
        return items, total or 0

    def replace_chunks(self, document: Document, chunks: list[Chunk]) -> None:
        """Idempotent: reprocessing a document swaps its whole chunk set."""
        self.db.execute(delete(Chunk).where(Chunk.document_id == document.id))
        self.db.add_all(chunks)

    def mark_interrupted(self) -> int:
        """At startup, any job still pending/processing died with the previous process."""
        stuck = self.db.scalars(
            select(Document).where(
                Document.status.in_([DocumentStatus.PENDING, DocumentStatus.PROCESSING])
            )
        ).all()
        for doc in stuck:
            doc.status = DocumentStatus.FAILED
            doc.error = "Processing was interrupted by a restart. Reprocess the document."
        return len(stuck)
