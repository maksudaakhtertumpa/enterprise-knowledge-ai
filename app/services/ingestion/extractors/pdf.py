"""PDF text extraction with page numbers and outline-derived section headings."""

import logging
import re
import statistics
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import DependencyError, FileNotDecryptedError, PdfReadError

from app.services.ingestion.normalize import normalize_text
from app.services.ingestion.types import Block, ExtractedDocument, ExtractionError

# pypdf logs a warning for every slightly-malformed object; that's noise for us.
logging.getLogger("pypdf").setLevel(logging.ERROR)

_TERMINAL = re.compile(r"[.!?:;][\"')\]]?$")
# Below this many characters per page on average, the PDF is probably scanned images.
_SCANNED_CHARS_PER_PAGE = 40


def extract_pdf(path: Path, *, max_pages: int) -> ExtractedDocument:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            _try_empty_password(reader)
        page_count = len(reader.pages)
    except ExtractionError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as exc:
        raise ExtractionError("corrupt_file", "The PDF could not be parsed.") from exc

    if page_count == 0:
        raise ExtractionError("empty_document", "The PDF has no pages.")
    if page_count > max_pages:
        raise ExtractionError(
            "too_many_pages", f"The PDF has {page_count} pages; the limit is {max_pages}."
        )

    headings = _outline_headings(reader)
    blocks: list[Block] = []
    warnings: list[str] = []
    current_heading: str | None = None
    total_chars = 0

    for index, page in enumerate(reader.pages):
        page_no = index + 1
        current_heading = headings.get(page_no, current_heading)
        try:
            raw = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page must not sink the document
            warnings.append(f"page {page_no}: text extraction failed")
            continue
        for paragraph in _paragraphs(raw):
            text = normalize_text(paragraph)
            if text:
                total_chars += len(text)
                blocks.append(Block(text=text, page=page_no, heading=current_heading))

    if total_chars == 0:
        raise ExtractionError(
            "no_text",
            "No extractable text found. The PDF may be scanned images; OCR is not supported yet.",
        )
    if total_chars / page_count < _SCANNED_CHARS_PER_PAGE:
        warnings.append("very little text per page; parts of this PDF may be scanned images")

    return ExtractedDocument(
        blocks=blocks,
        page_count=page_count,
        metadata=_metadata(reader),
        warnings=warnings,
    )


def _try_empty_password(reader: PdfReader) -> None:
    # Many "encrypted" PDFs only restrict printing/copying and open with an empty password.
    try:
        if reader.decrypt(""):
            return
    except (DependencyError, FileNotDecryptedError, NotImplementedError):
        pass
    raise ExtractionError("encrypted", "The PDF is password-protected.")


def _paragraphs(page_text: str) -> list[str]:
    """Split one page's text into paragraphs.

    pypdf separates lines with newlines but rarely marks paragraph breaks.
    We break on blank lines; after a line that ends a sentence and is clearly
    shorter than the page's full-width lines (the last line of a paragraph);
    and around short, unpunctuated, capitalised lines (titles, headings).
    """
    lines = [line.rstrip() for line in page_text.splitlines()]
    non_empty = [len(line) for line in lines if line.strip()]
    if not non_empty:
        return []
    # 90th-percentile line length ~ a full-width line (robust to one odd long line).
    full_width = (
        statistics.quantiles(non_empty, n=10)[-1] if len(non_empty) >= 4 else max(non_empty)
    )

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                paragraphs.append("\n".join(current))
                current = []
            continue
        if full_width and _looks_like_title(line, full_width):
            if current:
                paragraphs.append("\n".join(current))
                current = []
            paragraphs.append(line)
            continue
        current.append(line)
        if full_width and _TERMINAL.search(line) and len(line) < 0.8 * full_width:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


def _looks_like_title(line: str, full_width: float) -> bool:
    stripped = line.strip()
    return (
        len(stripped) < 0.6 * full_width
        and not _TERMINAL.search(stripped)
        and not stripped.endswith(",")
        and stripped[:1].isupper()
        and len(stripped.split()) <= 12
    )


def _outline_headings(reader: PdfReader) -> dict[int, str]:
    """Map 1-based page number -> last bookmark title starting on that page."""
    result: dict[int, str] = {}

    def walk(items: list[Any]) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            try:
                page_no = reader.get_destination_page_number(item) + 1
                title = normalize_text(str(item.title))
            except Exception:  # noqa: BLE001, S112 - malformed bookmarks are common; skip them
                continue
            if title and page_no > 0:
                result[page_no] = title[:500]

    try:
        walk(reader.outline)
    except Exception:  # noqa: BLE001 - outline is optional metadata
        return {}
    return result


def _metadata(reader: PdfReader) -> dict[str, Any]:
    try:
        info = reader.metadata
    except Exception:  # noqa: BLE001
        return {}
    if not info:
        return {}
    out: dict[str, Any] = {}
    for key, attr in (("title", "title"), ("author", "author"), ("subject", "subject")):
        value = getattr(info, attr, None)
        if value:
            out[key] = normalize_text(str(value))[:500]
    try:
        created = info.creation_date  # parses a PDF date string; malformed ones raise
    except Exception:  # noqa: BLE001
        created = None
    if created:
        out["created"] = created.isoformat()
    return out
