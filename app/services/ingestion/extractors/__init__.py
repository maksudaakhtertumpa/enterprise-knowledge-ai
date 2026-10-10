"""Dispatch to the right extractor for a file type."""

from pathlib import Path

from app.core.config import Settings
from app.db.models import FileType
from app.services.ingestion.extractors.docx import extract_docx
from app.services.ingestion.extractors.pdf import extract_pdf
from app.services.ingestion.extractors.txt import extract_txt
from app.services.ingestion.types import ExtractedDocument


def extract(path: Path, file_type: FileType, settings: Settings) -> ExtractedDocument:
    match file_type:
        case FileType.PDF:
            return extract_pdf(path, max_pages=settings.max_pdf_pages)
        case FileType.DOCX:
            return extract_docx(
                path, max_uncompressed_bytes=settings.max_docx_uncompressed_mb * 1024 * 1024
            )
        case FileType.TXT:
            return extract_txt(path)
    raise ValueError(f"no extractor for {file_type}")  # pragma: no cover - enum is exhaustive
