"""SQLAlchemy ORM models.

Every content row carries `workspace_id`. Repositories always filter on it,
which is what keeps one workspace's documents invisible to another.
"""

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, deferred, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Stores UTC; always returns timezone-aware datetimes (SQLite drops tzinfo)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime not allowed; use UTC-aware datetimes")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        return value.replace(tzinfo=UTC) if value is not None else None


class Base(DeclarativeBase):
    type_annotation_map = {datetime: UTCDateTime(), dict[str, Any]: JSON}


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )


class ApiKey(Base):
    """Only a SHA-256 hash of the key is stored. The plaintext is shown once, at creation."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    # First characters of the key, safe to display, so humans can tell keys apart.
    key_prefix: Mapped[str] = mapped_column(String(16))
    label: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(default=None)

    workspace: Mapped[Workspace] = relationship(back_populates="api_keys")


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class FileType(enum.StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # Duplicate detection is per workspace: two teams may hold the same file.
        UniqueConstraint("workspace_id", "sha256", name="uq_document_workspace_sha256"),
        Index("ix_documents_workspace_created", "workspace_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[FileType] = mapped_column(Enum(FileType, native_enum=False, length=8))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(255))

    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, native_enum=False, length=16), default=DocumentStatus.PENDING
    )
    error: Mapped[str | None] = mapped_column(String(500), default=None)

    title: Mapped[str | None] = mapped_column(String(500), default=None)
    page_count: Mapped[int | None] = mapped_column(default=None)
    char_count: Mapped[int | None] = mapped_column(default=None)
    chunk_count: Mapped[int] = mapped_column(default=0)
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)

    # Normalised full text; chunk offsets index into it. Deferred so list
    # queries don't load whole documents.
    extracted_text: Mapped[str | None] = deferred(mapped_column(Text, default=None))

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(default=None)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.ordinal",
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunk_document_ordinal"),
        Index("ix_chunks_workspace", "workspace_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    # Denormalised so retrieval can filter by workspace without a join.
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(default=None)
    page_end: Mapped[int | None] = mapped_column(default=None)
    heading: Mapped[str | None] = mapped_column(String(500), default=None)
    token_count: Mapped[int] = mapped_column(Integer)

    document: Mapped[Document] = relationship(back_populates="chunks")
