"""Long-lived services shared by request handlers, background jobs and the CLI.

Built once per app (or CLI run) and passed explicitly — no module-level
globals — so tests can construct fully isolated instances.
"""

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.session import build_engine, build_session_factory
from app.services.retrieval.embeddings import Embedder, build_embedder
from app.services.retrieval.reranker import Reranker, build_reranker
from app.services.retrieval.vector_store import VectorStore


@dataclass
class Services:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    embedder: Embedder
    vectors: VectorStore
    reranker: Reranker | None = None

    def close(self) -> None:
        """Release file handles and connections (called on shutdown)."""
        self.vectors.close()
        self.engine.dispose()


def build_services(settings: Settings) -> Services:
    engine = build_engine(settings.db_url)
    return Services(
        settings=settings,
        engine=engine,
        session_factory=build_session_factory(engine),
        embedder=build_embedder(settings),
        vectors=VectorStore(settings.chroma_dir),
        reranker=build_reranker(settings),
    )
