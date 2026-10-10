"""BM25 keyword retrieval with SQLite FTS5 — one FTS table per workspace.

Why one table per workspace rather than one shared table with a
`workspace_id` column:

1. Isolation by construction: a query physically cannot read another
   workspace's rows.
2. Correct BM25 statistics. BM25's IDF and average-length terms are computed
   over the whole table. In a shared table, workspace A's scores would shift
   with workspace B's documents — a ranking-quality problem *and* a subtle side
   channel (A could probe whether a term is common in other tenants' data).

Writes happen in the caller's SQLAlchemy session, so chunks and their keyword
index commit or roll back together.

Tokenizer: `porter unicode61 remove_diacritics 2` — Unicode-aware word
splitting, accent folding (résumé = resume) and English stemming (leaves ->
leav, approved -> approv). Stemming slightly mangles non-English words; for
Malay-heavy corpora `unicode61` alone may be better (configurable later).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import Chunk

_WORD = re.compile(r"\w+", re.UNICODE)
_MAX_TERMS = 32
# Function words that carry no retrieval signal. Removed only when the query
# also has real terms, so "to be or not to be" still searches for something.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)
# Heading matches count double: a query term in a section title is a strong signal.
_TEXT_WEIGHT, _HEADING_WEIGHT = 1.0, 2.0


@dataclass(frozen=True, slots=True)
class KeywordHit:
    chunk_id: str
    score: float  # higher is better (negated FTS5 bm25)
    matched_terms: tuple[str, ...]


def table_name(workspace_id: str) -> str:
    hex_id = workspace_id.replace("-", "")
    if not re.fullmatch(r"[0-9a-f]{32}", hex_id):
        raise ValueError("invalid workspace id")
    return f"fts_{hex_id}"  # safe to interpolate: validated hex only


def query_terms(query: str) -> list[str]:
    """Turn free text into safe FTS5 terms (no operators can be injected)."""
    terms: list[str] = []
    for word in _WORD.findall(query.lower()):
        if word not in terms:
            terms.append(word)
    content = [t for t in terms if t not in _STOPWORDS]
    return (content or terms)[:_MAX_TERMS]


class KeywordIndex:
    def __init__(self, db: Session) -> None:
        self.db = db

    def _ensure_table(self, workspace_id: str) -> str:
        name = table_name(workspace_id)
        self.db.execute(
            text(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING fts5("
                "text, heading, chunk_id UNINDEXED, document_id UNINDEXED, "
                "tokenize = 'porter unicode61 remove_diacritics 2')"
            )
        )
        return name

    def _exists(self, workspace_id: str) -> bool:
        return (
            self.db.execute(
                text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n"),
                {"n": table_name(workspace_id)},
            ).first()
            is not None
        )

    def replace_document(
        self, workspace_id: str, document_id: str, chunks: Sequence[Chunk]
    ) -> None:
        name = self._ensure_table(workspace_id)
        self.db.execute(text(f"DELETE FROM {name} WHERE document_id = :d"), {"d": document_id})
        if chunks:
            self.db.execute(
                text(
                    f"INSERT INTO {name} (text, heading, chunk_id, document_id) "
                    "VALUES (:text, :heading, :chunk_id, :document_id)"
                ),
                [
                    {
                        "text": c.text,
                        "heading": c.heading or "",
                        "chunk_id": c.id,
                        "document_id": document_id,
                    }
                    for c in chunks
                ],
            )

    def delete_document(self, workspace_id: str, document_id: str) -> None:
        if self._exists(workspace_id):
            name = table_name(workspace_id)
            self.db.execute(text(f"DELETE FROM {name} WHERE document_id = :d"), {"d": document_id})

    def search(
        self,
        workspace_id: str,
        query: str,
        limit: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[KeywordHit]:
        terms = query_terms(query)
        if not terms or not self._exists(workspace_id):
            return []
        name = table_name(workspace_id)
        # Each term is double-quoted, so FTS5 treats it as a literal token —
        # user input can never become NEAR/AND/NOT/column operators.
        match = " OR ".join(f'"{t}"' for t in terms)
        params: dict[str, object] = {"match": match, "limit": limit}
        doc_filter = ""
        if document_ids:
            keys = [f"d{i}" for i in range(len(document_ids))]
            doc_filter = f"AND document_id IN ({', '.join(':' + k for k in keys)})"
            params.update(dict(zip(keys, document_ids, strict=True)))
        rows = self.db.execute(
            text(  # table name is validated hex; values are bound
                f"SELECT chunk_id, bm25({name}, {_TEXT_WEIGHT}, {_HEADING_WEIGHT}) AS rank, "
                f"text, heading FROM {name} WHERE {name} MATCH :match {doc_filter} "
                "ORDER BY rank LIMIT :limit"
            ),
            params,
        ).all()
        return [
            KeywordHit(
                chunk_id=row.chunk_id,
                score=-float(row.rank),
                matched_terms=_matched(terms, f"{row.heading} {row.text}"),
            )
            for row in rows
        ]


def _matched(terms: list[str], content: str) -> tuple[str, ...]:
    words = set(_WORD.findall(content.lower()))
    # Prefix match approximates the stemmer, e.g. "approval" vs "approved".
    return tuple(
        t for t in terms if t in words or (len(t) >= 5 and any(w.startswith(t[:5]) for w in words))
    )
