from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.security import create_workspace_with_key
from app.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Each test gets its own data directory and database, so tests never share state."""
    # The hashing embedder is offline and instant; real-model tests live in tests/ml/.
    return Settings(
        environment="test",
        data_dir=tmp_path / "data",
        log_json=False,
        embedding_backend="hashing",
        reranker_backend="hashing",
        llm_backend="extractive",
        entailment_backend="lexical",
        # Plain retrieval by default in tests, so each test controls reranking explicitly.
        rerank_by_default=False,
        _env_file=None,
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # Context manager runs the lifespan (startup/shutdown) like a real server.
    # raise_server_exceptions=False lets us assert on the 500 envelope.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _new_workspace(app: FastAPI, name: str) -> dict[str, str]:
    with app.state.session_factory() as db:
        workspace, key = create_workspace_with_key(db, name)
        db.commit()
        return {"id": workspace.id, "key": key}


@pytest.fixture
def workspace(client: TestClient, app: FastAPI) -> dict[str, str]:
    return _new_workspace(app, "Acme")


@pytest.fixture
def other_workspace(client: TestClient, app: FastAPI) -> dict[str, str]:
    return _new_workspace(app, "Globex")


@pytest.fixture
def auth(workspace: dict[str, str]) -> dict[str, str]:
    return {"X-API-Key": workspace["key"]}


@pytest.fixture
def other_auth(other_workspace: dict[str, str]) -> dict[str, str]:
    return {"X-API-Key": other_workspace["key"]}
