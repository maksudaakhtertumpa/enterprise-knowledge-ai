"""Dense vector storage in ChromaDB — one collection per workspace.

ChromaDB holds only vectors plus the minimum metadata needed to filter
(`document_id`, `ordinal`). SQLite stays the source of truth for chunk text
and metadata, so the two can never disagree about what a chunk says; search
results are always "hydrated" from SQLite.

Each collection records which embedding model and dimension produced its
vectors. Querying with a different model (e.g. after changing
EKA_EMBEDDING_MODEL) is refused with a clear "re-index" error instead of
returning meaningless similarities.
"""

import logging
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class IndexMismatchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VectorHit:
    chunk_id: str
    score: float  # cosine similarity in [-1, 1]; higher is better


def collection_name(workspace_id: str) -> str:
    hex_id = workspace_id.replace("-", "")
    if not re.fullmatch(r"[0-9a-f]{32}", hex_id):
        raise ValueError("invalid workspace id")
    return f"ws-{hex_id}"


class VectorStore:
    """Opened lazily, so constructing the app never touches the disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._client: Any = None
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self.path.mkdir(parents=True, exist_ok=True)
                    self._client = chromadb.PersistentClient(
                        path=str(self.path),
                        # No usage data leaves the machine.
                        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
                    )
        return self._client

    def close(self) -> None:
        """Release Chroma's file handles and background resources. Idempotent.

        Without this, every app instance leaks ~7 open files. macOS allows only
        256 per process by default, so a long test run (or a server reloading
        repeatedly) eventually fails with confusing "file not found" errors.
        """
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def heartbeat(self) -> None:
        self.client.heartbeat()

    # ------------------------------------------------------------ collections
    def _get(self, workspace_id: str, model: str, dimension: int, create: bool) -> Any | None:
        name = collection_name(workspace_id)
        if create:
            collection = self.client.get_or_create_collection(
                name,
                # Never let Chroma embed for us: its default model would be
                # downloaded silently and differ from ours.
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"embedding_model": model, "dimension": dimension},
            )
        else:
            try:
                collection = self.client.get_collection(name, embedding_function=None)
            except Exception:  # noqa: BLE001 - chroma raises NotFound variants by version
                return None
        meta = collection.metadata or {}
        if meta.get("embedding_model") != model or meta.get("dimension") != dimension:
            raise IndexMismatchError(
                f"This workspace's vectors were built with {meta.get('embedding_model')!r} "
                f"(dim {meta.get('dimension')}), but the server now uses {model!r} "
                f"(dim {dimension}). Run: python -m app.cli reindex"
            )
        return collection

    def drop_workspace(self, workspace_id: str) -> None:
        try:
            self.client.delete_collection(collection_name(workspace_id))
        except Exception:  # noqa: BLE001 - absent is fine
            logger.debug("drop_workspace_noop", extra={"workspace_id": workspace_id})

    # ------------------------------------------------------------------- data
    def upsert(
        self,
        workspace_id: str,
        *,
        model: str,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, str | int]],
    ) -> None:
        if not ids:
            return
        collection = self._get(workspace_id, model, len(embeddings[0]), create=True)
        collection.upsert(ids=list(ids), embeddings=list(embeddings), metadatas=list(metadatas))

    def delete_document(self, workspace_id: str, document_id: str) -> None:
        try:
            collection = self.client.get_collection(
                collection_name(workspace_id), embedding_function=None
            )
        except Exception:  # noqa: BLE001 - nothing indexed yet
            return
        collection.delete(where={"document_id": document_id})

    def query(
        self,
        workspace_id: str,
        *,
        model: str,
        embedding: Sequence[float],
        limit: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[VectorHit]:
        collection = self._get(workspace_id, model, len(embedding), create=False)
        if collection is None:
            return []
        count = collection.count()
        if count == 0:
            return []
        where = {"document_id": {"$in": list(document_ids)}} if document_ids else None
        result = collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(limit, count),
            where=where,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma's cosine distance is 1 - cosine similarity.
        return [VectorHit(chunk_id=i, score=1.0 - d) for i, d in zip(ids, distances, strict=True)]

    def count(self, workspace_id: str) -> int:
        try:
            return self.client.get_collection(
                collection_name(workspace_id), embedding_function=None
            ).count()
        except Exception:  # noqa: BLE001
            return 0
