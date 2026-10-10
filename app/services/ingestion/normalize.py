"""Text normalisation and assembly.

Applied once, at ingestion, so that BM25, embeddings and citation matching
all see identical text.
"""

import re
import unicodedata

from app.services.ingestion.types import AssembledText, Block, BlockSpan

BLOCK_SEPARATOR = "\n\n"

# Control characters except tab/newline (NUL, form feed, etc.).
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Word broken across a line: "infor-\nmation" -> "information" (lowercase both sides only,
# so real hyphenated compounds like "Kuala-\nLumpur" are left alone).
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_SPACES = re.compile(r"[ \t  -   　]+")
_ZERO_WIDTH = re.compile(r"[­​-‍⁠﻿]")


def normalize_text(text: str) -> str:
    """Normalise one block's text into a single clean paragraph-ish string.

    - NFKC: folds ligatures (ﬁ -> fi) and full-width forms, which PDFs are full of.
    - Removes soft hyphens / zero-width characters and control characters.
    - Rejoins words hyphenated across line breaks.
    - Joins the remaining single line breaks into spaces (PDF line wrapping).
      Use `normalize_lines` where line structure matters (tables).
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub(" ", text)
    text = _HYPHEN_BREAK.sub("", text)
    text = text.replace("\n", " ")
    return _SPACES.sub(" ", text).strip()


def normalize_lines(text: str) -> str:
    """Like normalize_text but keeps line breaks (tables, lists)."""
    lines = [normalize_text(line) for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def assemble(blocks: list[Block]) -> AssembledText:
    """Join blocks into one canonical string, recording each block's span."""
    parts: list[str] = []
    spans: list[BlockSpan] = []
    cursor = 0
    for block in blocks:
        if not block.text:
            continue
        if parts:
            parts.append(BLOCK_SEPARATOR)
            cursor += len(BLOCK_SEPARATOR)
        parts.append(block.text)
        spans.append(
            BlockSpan(
                start=cursor,
                end=cursor + len(block.text),
                kind=block.kind,
                page=block.page,
                heading=block.heading,
            )
        )
        cursor += len(block.text)
    return AssembledText(text="".join(parts), spans=spans)
