"""Structure-aware chunking.

Strategy (in priority order):
1. A new section (a heading block, or a change of enclosing heading) always
   starts a new chunk, so a chunk never mixes sections and its `heading` is
   meaningful for citations.
2. Within a section, whole sentences (or table rows) are packed greedily up
   to `chunk_size` tokens. We split *between* sentences, never inside one,
   unless a single sentence is longer than a chunk.
3. Over-long sentences fall back to word windows; a single "word" longer than
   a chunk (e.g. a base64 blob) falls back to character windows.
4. Consecutive chunks in a section share up to `overlap` tokens of whole
   sentences, so a fact that straddles a boundary is retrievable from either side.

Invariant (tested): every chunk's text is exactly `doc.text[char_start:char_end]`.
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from app.services.ingestion.types import AssembledText, BlockKind, ChunkSpec

TokenCounter = Callable[[str], int]

_TOKEN = re.compile(r"\w+|[^\w\s]")


def approx_token_count(text: str) -> int:
    """Approximate sub-word token count; Phase 3 swaps in the embedding model's tokenizer.

    Each word or punctuation mark counts as one token, plus one per extra 10
    characters, because real tokenizers split long strings (IDs, URLs, base64)
    into many pieces. Close to real counts for English prose.
    """
    return sum(1 + (len(tok) - 1) // 10 for tok in _TOKEN.findall(text))


_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*\s+")
_NEXT_STARTS_SENTENCE = re.compile(r"[A-Z0-9\"'“‘(\[]")
_ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "no",
        "fig",
        "figs",
        "eq",
        "al",
        "approx",
        "dept",
        "inc",
        "ltd",
        "co",
        "corp",
        "sdn",
        "bhd",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    ]
)
_INITIAL = re.compile(r"(?:[A-Za-z]\.)+")


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    tokens: int
    page: int | None
    heading: str | None
    is_heading: bool


class Chunker:
    def __init__(
        self,
        chunk_size: int,
        overlap: int,
        count_tokens: TokenCounter = approx_token_count,
    ) -> None:
        if chunk_size < 16:
            raise ValueError("chunk_size must be at least 16 tokens")
        if not 0 <= overlap < chunk_size // 2:
            raise ValueError("overlap must be in [0, chunk_size / 2)")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.count_tokens = count_tokens

    # ------------------------------------------------------------------ public
    def split(self, doc: AssembledText) -> list[ChunkSpec]:
        chunks: list[ChunkSpec] = []
        current: list[_Unit] = []
        current_tokens = 0

        for unit in self._units(doc):
            # A section starts at a heading block, or wherever the enclosing heading
            # changes (PDF bookmarks label blocks without adding heading blocks).
            starts_section = unit.is_heading or (
                bool(current) and unit.heading != current[-1].heading
            )
            too_big = current_tokens + unit.tokens > self.chunk_size
            only_headings = bool(current) and all(u.is_heading for u in current)

            # Keep stacked headings ("2 Policy" / "2.1 Leave") together with their body.
            if current and (too_big or (starts_section and not only_headings)):
                chunks.append(self._emit(doc, current, len(chunks)))
                current = [] if starts_section else self._overlap_tail(current)
                current_tokens = sum(u.tokens for u in current)
                while current and current_tokens + unit.tokens > self.chunk_size:
                    current_tokens -= current.pop(0).tokens

            current.append(unit)
            current_tokens += unit.tokens

        if current:
            chunks.append(self._emit(doc, current, len(chunks)))
        return chunks

    # --------------------------------------------------------------- packing
    def _overlap_tail(self, units: list[_Unit]) -> list[_Unit]:
        tail: list[_Unit] = []
        tokens = 0
        for unit in reversed(units):
            if unit.is_heading or tokens + unit.tokens > self.overlap:
                break
            tail.insert(0, unit)
            tokens += unit.tokens
        # Never carry the whole previous chunk forward — that would duplicate it.
        return tail if len(tail) < len(units) else tail[1:]

    def _emit(self, doc: AssembledText, units: list[_Unit], ordinal: int) -> ChunkSpec:
        start, end = units[0].start, units[-1].end
        text = doc.text[start:end]
        pages = [u.page for u in units if u.page is not None]
        headings = [u.heading for u in units if u.is_heading]
        return ChunkSpec(
            ordinal=ordinal,
            char_start=start,
            char_end=end,
            text=text,
            token_count=self.count_tokens(text),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            heading=headings[-1] if headings else units[0].heading,
        )

    # ------------------------------------------------------------- splitting
    def _units(self, doc: AssembledText) -> Iterator[_Unit]:
        for span in doc.spans:
            if span.kind is BlockKind.HEADING:
                pieces = [(span.start, span.end)]
            elif span.kind is BlockKind.TABLE:
                pieces = list(_line_spans(doc.text, span.start, span.end))
            else:
                pieces = list(_sentence_spans(doc.text, span.start, span.end))

            for start, end in pieces:
                for s, e in self._fit(doc.text, start, end):
                    yield _Unit(
                        start=s,
                        end=e,
                        tokens=self.count_tokens(doc.text[s:e]),
                        page=span.page,
                        heading=span.heading,
                        is_heading=span.kind is BlockKind.HEADING,
                    )

    def _fit(self, text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
        """Yield (start, end) pieces of text[start:end], each within chunk_size tokens."""
        if self.count_tokens(text[start:end]) <= self.chunk_size:
            yield start, end
            return

        window_start: int | None = None
        window_end = start
        window_tokens = 0
        for match in re.finditer(r"\S+", text[start:end]):
            w_start, w_end = start + match.start(), start + match.end()
            w_tokens = self.count_tokens(match.group())

            if w_tokens > self.chunk_size:  # pathological "word": cut by characters
                if window_start is not None:
                    yield window_start, window_end
                    window_start, window_tokens = None, 0
                yield from self._char_windows(text, w_start, w_end)
                continue

            if window_start is not None and window_tokens + w_tokens > self.chunk_size:
                yield window_start, window_end
                window_start, window_tokens = None, 0
            if window_start is None:
                window_start = w_start
            window_end = w_end
            window_tokens += w_tokens

        if window_start is not None:
            yield window_start, window_end

    def _char_windows(self, text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
        pos = start
        step = max(self.chunk_size, 16)
        while pos < end:
            stop = min(pos + step, end)
            # Shrink until the piece fits (dense punctuation can mean >1 token per char).
            while stop - pos > 1 and self.count_tokens(text[pos:stop]) > self.chunk_size:
                stop = pos + (stop - pos) // 2
            yield pos, stop
            pos = stop


def _sentence_spans(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    block = text[start:end]
    sent_start = 0
    for match in _SENTENCE_END.finditer(block):
        if match.end() >= len(block):
            break
        if not _NEXT_STARTS_SENTENCE.match(block, match.end()):
            continue
        word = block[: match.start() + 1].rsplit(None, 1)[-1]  # token ending in the '.'
        bare = word.rstrip(".").lower()
        if word.endswith(".") and (bare in _ABBREVIATIONS or _INITIAL.fullmatch(word)):
            continue
        sent_end = match.start() + len(match.group().rstrip())
        yield start + sent_start, start + sent_end
        sent_start = match.end()
    tail = block[sent_start:].rstrip()
    if tail:
        yield start + sent_start, start + sent_start + len(tail)


def _line_spans(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    pos = start
    for line in text[start:end].split("\n"):
        if line.strip():
            yield pos, pos + len(line)
        pos += len(line) + 1
