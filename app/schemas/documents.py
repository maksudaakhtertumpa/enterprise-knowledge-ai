from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentStatus, FileType


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: FileType
    size_bytes: int
    sha256: str
    status: DocumentStatus
    error: str | None = None
    title: str | None = None
    page_count: int | None = Field(None, description="Null for formats without fixed pages")
    char_count: int | None = None
    chunk_count: int
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="meta")
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    ordinal: int
    text: str
    heading: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int
    char_end: int
    token_count: int


class ChunkList(BaseModel):
    document_id: str
    items: list[ChunkOut]
    total: int
    limit: int
    offset: int


class WorkspaceOut(BaseModel):
    id: str
    name: str
