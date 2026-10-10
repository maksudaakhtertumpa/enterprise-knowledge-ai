"""Engine and session factory.

One engine per app instance (stored on app.state), so tests can run many
isolated apps side by side. Request handlers get a session through
`get_db`; background jobs open their own session from the same factory.
"""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


def build_engine(url: str) -> Engine:
    is_sqlite = url.startswith("sqlite")
    engine = create_engine(
        url,
        # FastAPI runs sync handlers and background jobs in a thread pool.
        connect_args={"check_same_thread": False, "timeout": 30} if is_sqlite else {},
        pool_pre_ping=not is_sqlite,
    )

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")  # SQLite ignores FKs unless asked
            cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


def init_db(engine: Engine) -> None:
    """Create tables if missing. Alembic migrations replace this with the Postgres move."""
    Base.metadata.create_all(engine)


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()
