"""Plain-text extraction with encoding detection and Markdown-style headings."""

import re
from pathlib import Path

from app.services.ingestion.normalize import normalize_text
from app.services.ingestion.types import Block, BlockKind, ExtractedDocument, ExtractionError

_BLANK_LINES = re.compile(r"\n\s*\n")
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


def decode_text(raw: bytes) -> tuple[str, str]:
    """Return (text, encoding). UTF-8 first, then Windows-1252, then Latin-1 as last resort."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding), encoding.removesuffix("-sig")
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"  # never fails; flagged as a warning


def extract_txt(path: Path) -> ExtractedDocument:
    text, encoding = decode_text(path.read_bytes())
    warnings = [] if encoding == "utf-8" else [f"decoded as {encoding}, not UTF-8"]

    blocks: list[Block] = []
    heading: str | None = None
    for paragraph in _BLANK_LINES.split(text.replace("\r\n", "\n")):
        lines = paragraph.strip("\n").split("\n")
        # A heading line may be glued to the paragraph below it.
        while lines and (match := _MD_HEADING.match(lines[0].strip())):
            heading = normalize_text(match.group(1))[:500]
            if heading:
                blocks.append(Block(text=heading, kind=BlockKind.HEADING, heading=heading))
            lines = lines[1:]
        body = normalize_text("\n".join(lines))
        if body:
            blocks.append(Block(text=body, heading=heading))

    if not blocks:
        raise ExtractionError("no_text", "The text file is empty.")
    return ExtractedDocument(blocks=blocks, metadata={"encoding": encoding}, warnings=warnings)
