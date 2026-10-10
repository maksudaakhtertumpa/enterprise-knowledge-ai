"""Ingestion job: extract -> normalise -> chunk -> keyword index -> embed -> ready.

Runs after the upload response is sent (FastAPI BackgroundTasks) in its own
DB session. The document only becomes `ready` once *both* indexes hold it;
search only returns chunks of `ready` documents, so a half-indexed document
is never visible.

Expected failures (encrypted PDF, no text, corrupt file) become `failed` with
a message the user can act on; unexpected ones are logged with a stack trace
and shown to the user only as a generic error.
"""

import logging
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentStatus, utcnow
from app.repositories.documents import DocumentRepository
from app.services.container import Services
from app.services.ingestion.chunking import Chunker
from app.services.ingestion.extractors import extract
from app.services.ingestion.normalize import assemble
from app.services.ingestion.types import ExtractionError
from app.services.retrieval.indexing import index_vectors
from app.services.retrieval.keyword import KeywordIndex
from app.services.storage import LocalFileStorage

logger = logging.getLogger(__name__)


def ingest_document(services: Services, document_id: str) -> None:
    settings = services.settings
    storage = LocalFileStorage(settings.uploads_dir)

    with services.session_factory() as db:
        doc = db.scalar(select(Document).where(Document.id == document_id))
        if doc is None:
            logger.warning("ingest_skipped_missing", extra={"document_id": document_id})
            return
        doc.status = DocumentStatus.PROCESSING
        doc.error = None
        db.commit()

        # ---- 1. extract + chunk (pure CPU, no side effects) -------------------
        t0 = time.perf_counter()
        try:
            chunker = Chunker(
                settings.chunk_size_tokens,
                settings.chunk_overlap_tokens,
                count_tokens=services.embedder.count_tokens,
            )
            extracted = extract(storage.path_for(doc.storage_key), doc.file_type, settings)
            t_extract = time.perf_counter()
            assembled = assemble(extracted.blocks)
            specs = chunker.split(assembled)
            t_chunk = time.perf_counter()
        except ExtractionError as exc:
            _fail(db, doc, exc.message, code=exc.code)
            return
        except Exception:
            logger.exception("ingest_crashed", extra={"document_id": document_id})
            _fail(db, doc, "An internal error occurred while processing this document.")
            return

        # ---- 2. chunks + BM25 index in one transaction ------------------------
        chunks = [
            Chunk(
                document_id=doc.id,
                workspace_id=doc.workspace_id,
                ordinal=s.ordinal,
                text=s.text,
                char_start=s.char_start,
                char_end=s.char_end,
                page_start=s.page_start,
                page_end=s.page_end,
                heading=s.heading,
                token_count=s.token_count,
            )
            for s in specs
        ]
        DocumentRepository(db).replace_chunks(doc, chunks)
        db.flush()  # assigns chunk IDs
        KeywordIndex(db).replace_document(doc.workspace_id, doc.id, chunks)
        doc.extracted_text = assembled.text
        doc.char_count = len(assembled.text)
        doc.page_count = extracted.page_count
        doc.chunk_count = len(chunks)
        doc.title = extracted.metadata.get("title") or None
        doc.meta = {
            **extracted.metadata,
            "warnings": extracted.warnings,
            "chunking": {
                "chunk_size_tokens": chunker.chunk_size,
                "chunk_overlap_tokens": chunker.overlap,
                "tokenizer": services.embedder.model_name,
            },
        }
        db.commit()  # still PROCESSING: invisible to search until step 3 succeeds

        # ---- 3. embeddings ------------------------------------------------------
        try:
            embed_ms = index_vectors(services, doc, chunks)
        except Exception:
            logger.exception("embedding_failed", extra={"document_id": document_id})
            try:
                services.vectors.delete_document(doc.workspace_id, doc.id)
            except Exception:  # noqa: BLE001 - best effort; hydration ignores orphans
                logger.warning("vector_cleanup_failed", extra={"document_id": document_id})
            _fail(
                db, doc, "The document could not be indexed for semantic search.", "indexing_failed"
            )
            return

        doc.meta = {**doc.meta, "embedding": services.embedder.describe()}
        doc.status = DocumentStatus.READY
        doc.processed_at = utcnow()
        db.commit()

        logger.info(
            "ingest_completed",
            extra={
                "document_id": doc.id,
                "file_type": doc.file_type.value,
                "pages": doc.page_count,
                "chunks": doc.chunk_count,
                "chars": doc.char_count,
                "extract_ms": round((t_extract - t0) * 1000, 1),
                "chunk_ms": round((t_chunk - t_extract) * 1000, 1),
                "embed_ms": round(embed_ms, 1),
            },
        )


def _fail(db: Session, doc: Document, message: str, code: str = "internal_error") -> None:
    db.rollback()
    doc.status = DocumentStatus.FAILED
    doc.error = message[:500]
    doc.meta = {**(doc.meta or {}), "error_code": code}
    db.commit()
    # Log the code, not the filename: filenames can themselves be sensitive.
    logger.warning("ingest_failed", extra={"document_id": doc.id, "error_code": code})
