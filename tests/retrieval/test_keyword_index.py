import uuid

import pytest
from sqlalchemy import text

from app.db.models import Chunk
from app.db.session import build_engine, build_session_factory, init_db
from app.services.retrieval.keyword import KeywordIndex, query_terms, table_name


@pytest.fixture
def db(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'k.db'}")
    init_db(engine)
    with build_session_factory(engine)() as session:
        yield session
    engine.dispose()


def ws() -> str:
    return str(uuid.uuid4())


def chunk(text_: str, heading: str | None = None) -> Chunk:
    return Chunk(id=str(uuid.uuid4()), text=text_, heading=heading)


def index(db, workspace, doc, *chunks):
    KeywordIndex(db).replace_document(workspace, doc, list(chunks))
    db.commit()
    return [c.id for c in chunks]


# ------------------------------------------------------------ query handling
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Annual LEAVE policy", ["annual", "leave", "policy"]),
        ("how many days of leave?", ["many", "days", "leave"]),
        ("leave leave LEAVE", ["leave"]),
        ("to be or not to be", ["not"]),
        ("the of and", ["the", "of", "and"]),  # only stopwords: keep them
        ('"quoted" NEAR(a b) text:secret OR *', ["quoted", "near", "b", "text", "secret"]),
        ("résumé café", ["résumé", "café"]),
        ("!!! ???", []),
    ],
)
def test_query_terms(query, expected):
    assert query_terms(query) == expected


def test_query_terms_are_capped():
    assert len(query_terms(" ".join(f"word{i}" for i in range(100)))) == 32


@pytest.mark.parametrize("bad", ["", "x", "abc; DROP TABLE documents", "../" + "0" * 32])
def test_table_name_rejects_non_uuid(bad):
    with pytest.raises(ValueError):
        table_name(bad)


@pytest.mark.parametrize(
    "hostile",
    ['") OR 1=1 --', "NEAR(", "text:", '"', "a AND", "*", "^leave", "{text heading}: x"],
)
def test_hostile_queries_never_error(db, hostile):
    w = ws()
    index(db, w, "d", chunk("annual leave"))
    KeywordIndex(db).search(w, hostile, 10)  # must not raise an FTS5 syntax error


# ------------------------------------------------------------------- ranking
def test_more_relevant_chunk_ranks_first(db):
    w = ws()
    weak, strong = index(
        db,
        w,
        "d",
        chunk("The cafeteria opens at eight. Leave your trays at the counter."),
        chunk("Annual leave: employees receive fourteen days of annual leave per year."),
    )
    hits = KeywordIndex(db).search(w, "annual leave days", 10)
    assert [h.chunk_id for h in hits] == [strong, weak]
    assert hits[0].score > hits[1].score > 0
    assert set(hits[0].matched_terms) == {"annual", "leave", "days"}


def test_heading_match_is_boosted(db):
    w = ws()
    # Same term frequency (once each); only the field differs.
    body_only, in_heading, *_ = index(
        db,
        w,
        "d",
        chunk("Overtime is paid at the end of each month.", heading="Payroll"),
        chunk("It is paid at 1.5 times the hourly rate.", heading="Overtime"),
        *[chunk(f"Unrelated filler note number {i}.", heading="Misc") for i in range(5)],
    )
    hits = KeywordIndex(db).search(w, "overtime", 10)
    assert hits[0].chunk_id == in_heading


def test_stemming_and_diacritics(db):
    w = ws()
    (cid,) = index(db, w, "d", chunk("Leaves were approved for the café résumé review."))
    for query in ["leave approve", "cafe", "resume"]:
        assert [h.chunk_id for h in KeywordIndex(db).search(w, query, 5)] == [cid], query


def test_no_match_returns_empty(db):
    w = ws()
    index(db, w, "d", chunk("annual leave"))
    assert KeywordIndex(db).search(w, "quantum chromodynamics", 10) == []


def test_unknown_workspace_returns_empty_without_creating_table(db):
    w = ws()
    assert KeywordIndex(db).search(w, "anything", 10) == []
    assert not db.execute(
        text("SELECT 1 FROM sqlite_master WHERE name=:n"), {"n": table_name(w)}
    ).first()


# --------------------------------------------------------- writes & isolation
def test_replace_document_is_idempotent(db):
    w = ws()
    index(db, w, "d", chunk("old text about leave"))
    (new_id,) = index(db, w, "d", chunk("new text about leave"))
    hits = KeywordIndex(db).search(w, "leave", 10)
    assert [h.chunk_id for h in hits] == [new_id]


def test_delete_document_and_document_filter(db):
    w = ws()
    (a,) = index(db, w, "doc-a", chunk("leave policy A"))
    (b,) = index(db, w, "doc-b", chunk("leave policy B"))
    ki = KeywordIndex(db)
    assert {h.chunk_id for h in ki.search(w, "leave", 10, document_ids=["doc-b"])} == {b}
    ki.delete_document(w, "doc-a")
    db.commit()
    assert {h.chunk_id for h in ki.search(w, "leave", 10)} == {b}
    ki.delete_document(ws(), "doc-a")  # unknown workspace: no-op, no error
    assert a not in {h.chunk_id for h in ki.search(w, "policy", 10)}


def test_workspaces_cannot_see_each_other(db):
    a, b = ws(), ws()
    index(db, a, "d1", chunk("confidential salary bands for grade E5"))
    assert KeywordIndex(db).search(b, "salary bands", 10) == []
    index(db, b, "d2", chunk("public holiday calendar"))
    assert KeywordIndex(db).search(b, "salary bands", 10) == []


def test_scores_are_independent_of_other_workspaces(db):
    """Per-workspace tables mean BM25 statistics can't leak across tenants."""
    a, b = ws(), ws()
    index(db, a, "d1", chunk("annual leave is fourteen days"), chunk("claims are monthly"))
    before = KeywordIndex(db).search(a, "leave", 10)[0].score
    # Flood workspace B with the same term; A's IDF must not move.
    index(db, b, "d2", *[chunk(f"leave leave leave note {i}") for i in range(50)])
    after = KeywordIndex(db).search(a, "leave", 10)[0].score
    assert after == pytest.approx(before)
