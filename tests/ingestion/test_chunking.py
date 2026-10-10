import random

import pytest

from app.services.ingestion.chunking import Chunker, approx_token_count
from app.services.ingestion.normalize import assemble
from app.services.ingestion.types import Block, BlockKind


def sentences(n: int, prefix: str = "Fact") -> str:
    return " ".join(f"{prefix} number {i} is stated in this sentence." for i in range(n))


def check_invariants(doc, chunks, size):
    assert chunks, "expected at least one chunk"
    for i, c in enumerate(chunks):
        assert c.ordinal == i
        assert c.text == doc.text[c.char_start : c.char_end], "chunk must be an exact slice"
        assert c.text == c.text.strip(), "no leading/trailing whitespace"
        assert c.token_count <= size, f"chunk {i} has {c.token_count} tokens > {size}"
    # Every non-whitespace character of the document is covered by some chunk.
    covered = [False] * len(doc.text)
    for c in chunks:
        for j in range(c.char_start, c.char_end):
            covered[j] = True
    missing = [ch for ch, ok in zip(doc.text, covered, strict=True) if not ok and not ch.isspace()]
    assert not missing, f"uncovered text: {''.join(missing)[:80]!r}"


def test_short_document_is_one_chunk():
    doc = assemble([Block("One short paragraph.")])
    chunks = Chunker(100, 10).split(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "One short paragraph."


def test_long_text_respects_size_and_covers_everything():
    doc = assemble([Block(sentences(200))])
    chunks = Chunker(64, 16).split(doc)
    assert len(chunks) > 5
    check_invariants(doc, chunks, 64)


def test_chunks_break_between_sentences_not_inside():
    doc = assemble([Block(sentences(60))])
    for c in Chunker(64, 0).split(doc):
        assert c.text.startswith("Fact number") and c.text.endswith("sentence.")


def test_overlap_repeats_trailing_sentences():
    doc = assemble([Block(sentences(40))])
    chunks = Chunker(64, 20).split(doc)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.char_start < prev.char_end, "consecutive chunks should overlap"
        overlap = doc.text[nxt.char_start : prev.char_end]
        assert approx_token_count(overlap) <= 20


def test_zero_overlap_means_disjoint_chunks():
    doc = assemble([Block(sentences(40))])
    chunks = Chunker(64, 0).split(doc)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.char_start >= prev.char_end


def test_new_heading_starts_new_chunk_and_sets_heading():
    blocks = [
        Block("Leave", kind=BlockKind.HEADING, heading="Leave"),
        Block("Annual leave is fourteen days.", heading="Leave"),
        Block("Claims", kind=BlockKind.HEADING, heading="Claims"),
        Block("Claims are paid monthly.", heading="Claims"),
    ]
    chunks = Chunker(200, 20).split(assemble(blocks))
    assert [c.heading for c in chunks] == ["Leave", "Claims"]
    assert chunks[1].text.startswith("Claims")
    assert "fourteen" not in chunks[1].text, "overlap must not cross a section boundary"


def test_heading_change_without_heading_block_starts_new_chunk():
    # PDFs label blocks with bookmark headings but have no heading blocks.
    blocks = [
        Block("Leave text one.", page=1, heading="Leave"),
        Block("Claims text two.", page=2, heading="Claims"),
    ]
    chunks = Chunker(200, 20).split(assemble(blocks))
    assert [(c.heading, c.page_start) for c in chunks] == [("Leave", 1), ("Claims", 2)]


def test_stacked_headings_stay_with_their_body():
    blocks = [
        Block("2 Policy", kind=BlockKind.HEADING, heading="2 Policy"),
        Block("2.1 Leave", kind=BlockKind.HEADING, heading="2.1 Leave"),
        Block("Body text.", heading="2.1 Leave"),
    ]
    chunks = Chunker(200, 20).split(assemble(blocks))
    assert len(chunks) == 1
    assert chunks[0].heading == "2.1 Leave"


def test_page_range_tracks_blocks():
    blocks = [Block(sentences(30, f"P{p}"), page=p) for p in (1, 2, 3)]
    doc = assemble(blocks)
    chunks = Chunker(100, 0).split(doc)
    check_invariants(doc, chunks, 100)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    for c in chunks:
        assert c.page_start <= c.page_end
        # Text from page p is labelled "P{p} number"; check labels agree with page range.
        for p in (1, 2, 3):
            if f"P{p} number" in c.text:
                assert c.page_start <= p <= c.page_end


def test_pages_are_none_for_unpaginated_formats():
    chunks = Chunker(100, 0).split(assemble([Block("No pages here.")]))
    assert chunks[0].page_start is None and chunks[0].page_end is None


def test_abbreviations_do_not_split_sentences():
    text = "Dr. Tan approved it, e.g. for Sdn. Bhd. entities. Next sentence here."
    doc = assemble([Block(text)])
    # First sentence is 17 approx tokens; 20 fits it alone but not both sentences.
    chunks = Chunker(20, 0).split(doc)
    assert len(chunks) == 2
    assert chunks[0].text == "Dr. Tan approved it, e.g. for Sdn. Bhd. entities."


def test_sentence_longer_than_chunk_falls_back_to_word_windows():
    long_sentence = " ".join(f"word{i}" for i in range(300)) + "."
    doc = assemble([Block(long_sentence)])
    chunks = Chunker(50, 0).split(doc)
    check_invariants(doc, chunks, 50)
    assert len(chunks) >= 6


@pytest.mark.parametrize("alphabet", ["abcdef0123456789+/=", "abcdef0123456789"])
def test_giant_unbroken_token_is_cut_by_characters(alphabet):
    rng = random.Random(0)
    blob = "".join(rng.choice(alphabet) for _ in range(5000))
    doc = assemble([Block(f"Header text. {blob} Footer text.")])
    chunks = Chunker(64, 8).split(doc)
    check_invariants(doc, chunks, 64)
    assert len(chunks) > 5, "a 5,000-character blob must not hide inside one small chunk"


def test_approx_token_count_penalises_long_strings():
    assert approx_token_count("Annual leave is fourteen days.") == 6
    assert approx_token_count("x" * 100) == 10


def test_table_rows_are_units():
    rows = "\n".join(f"Grade {i} | {i * 2} days | Notes for grade {i}" for i in range(40))
    doc = assemble([Block(rows, kind=BlockKind.TABLE)])
    chunks = Chunker(60, 0).split(doc)
    check_invariants(doc, chunks, 60)
    for c in chunks:
        assert c.text.startswith("Grade ") and c.text.split("\n")[-1].count("|") == 2


def test_empty_document_has_no_chunks():
    assert Chunker(100, 10).split(assemble([])) == []


@pytest.mark.parametrize("seed", range(5))
def test_randomised_documents_keep_invariants(seed):
    rng = random.Random(seed)
    blocks = []
    for page in range(1, rng.randint(2, 6)):
        if rng.random() < 0.4:
            blocks.append(Block(f"Section {page}", kind=BlockKind.HEADING, page=page))
        for _ in range(rng.randint(1, 5)):
            words = [
                rng.choice(["alpha", "beta", "gamma", "delta", "RM1,200.50", "e.g."])
                for _ in range(rng.randint(3, 120))
            ]
            blocks.append(Block(" ".join(words) + ".", page=page))
    doc = assemble(blocks)
    size = rng.choice([32, 64, 128])
    check_invariants(doc, Chunker(size, size // 4).split(doc), size)


@pytest.mark.parametrize(("size", "overlap"), [(8, 0), (100, 50), (100, 60), (100, -1)])
def test_invalid_configuration_rejected(size, overlap):
    with pytest.raises(ValueError):
        Chunker(size, overlap)
