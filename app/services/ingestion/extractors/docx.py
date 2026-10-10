"""DOCX extraction: paragraphs and tables in reading order, with heading tracking.

DOCX has no fixed pages (pagination happens when Word renders), so blocks get
page=None and citations use the section heading instead. Headers/footers are
skipped: they repeat on every page and add noise to retrieval.
"""

import zipfile
from pathlib import Path
from typing import Any

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.services.ingestion.normalize import normalize_lines, normalize_text
from app.services.ingestion.types import Block, BlockKind, ExtractedDocument, ExtractionError
from app.services.ingestion.validation import check_zip_safety


def extract_docx(path: Path, *, max_uncompressed_bytes: int) -> ExtractedDocument:
    check_zip_safety(path, max_uncompressed_bytes=max_uncompressed_bytes)
    try:
        # python-docx parses XML with resolve_entities=False, so XXE payloads are inert.
        document = docx.Document(str(path))
    except (PackageNotFoundError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ExtractionError("corrupt_file", "The DOCX file could not be parsed.") from exc

    blocks: list[Block] = []
    current_heading: str | None = None

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = normalize_text(item.text)
            if not text:
                continue
            if _is_heading(item):
                current_heading = text[:500]
                blocks.append(Block(text=text, kind=BlockKind.HEADING, heading=current_heading))
            else:
                blocks.append(Block(text=text, heading=current_heading))
        elif isinstance(item, Table):
            text = _table_text(item)
            if text:
                blocks.append(Block(text=text, kind=BlockKind.TABLE, heading=current_heading))

    if not blocks:
        raise ExtractionError("no_text", "The DOCX file contains no text.")

    return ExtractedDocument(blocks=blocks, page_count=None, metadata=_metadata(document))


def _is_heading(paragraph: Paragraph) -> bool:
    try:
        name = (paragraph.style.name or "") if paragraph.style is not None else ""
    except (KeyError, AttributeError):
        return False
    return name == "Title" or name.startswith("Heading")


def _table_text(table: Table) -> str:
    """One line per row, cells separated by ' | '. Merged cells appear once."""
    lines: list[str] = []
    for row in table.rows:
        seen: set[int] = set()
        cells: list[str] = []
        for cell in row.cells:
            # A merged cell is returned once per grid column it spans.
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(normalize_text(cell.text))
        if any(cells):
            lines.append(" | ".join(cells))
    return normalize_lines("\n".join(lines))


def _metadata(document: Any) -> dict[str, Any]:
    props = document.core_properties
    out: dict[str, Any] = {}
    if props.title:
        out["title"] = normalize_text(props.title)[:500]
    if props.author:
        out["author"] = normalize_text(props.author)[:200]
    if props.created:
        out["created"] = props.created.isoformat()
    if props.modified:
        out["modified"] = props.modified.isoformat()
    return out
