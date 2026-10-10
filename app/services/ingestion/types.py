"""Data shapes passed between ingestion stages.

extract  ->  ExtractedDocument (list of Blocks with page/heading)
assemble ->  AssembledText (one normalised string + block spans into it)
chunk    ->  list[ChunkSpec] (spans into that same string)

Keeping offsets into a single canonical string means every chunk can be
traced back to its exact position in the document — the basis for citations.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"


@dataclass(slots=True)
class Block:
    text: str
    kind: BlockKind = BlockKind.PARAGRAPH
    page: int | None = None  # 1-based; None when the format has no fixed pages (DOCX, TXT)
    heading: str | None = None  # nearest enclosing section heading


@dataclass(slots=True)
class ExtractedDocument:
    blocks: list[Block]
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class BlockSpan:
    start: int
    end: int
    kind: BlockKind
    page: int | None
    heading: str | None


@dataclass(slots=True)
class AssembledText:
    text: str
    spans: list[BlockSpan]


@dataclass(slots=True, frozen=True)
class ChunkSpec:
    ordinal: int
    char_start: int
    char_end: int
    text: str
    token_count: int
    page_start: int | None
    page_end: int | None
    heading: str | None


class ExtractionError(Exception):
    """Expected, user-explainable failure (encrypted PDF, no text, corrupt file)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
