"""Regression test: app shutdown must release every file handle it opened.

macOS allows 256 open files per process by default. Before ChromaDB clients
were closed on shutdown, each app instance leaked ~11 handles and the test
suite crashed on a Mac after ~45 tests. Linux's higher limit hid the bug.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.security import create_workspace_with_key
from app.main import create_app

FD_DIR = "/proc/self/fd" if Path("/proc/self/fd").is_dir() else "/dev/fd"


def open_files() -> int:
    return len(os.listdir(FD_DIR))


def run_app_once(data_dir: Path) -> None:
    settings = Settings(
        data_dir=data_dir,
        embedding_backend="hashing",
        reranker_backend="hashing",
        log_json=False,
        _env_file=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        with app.state.session_factory() as db:
            _, key = create_workspace_with_key(db, "w")
            db.commit()
        headers = {"X-API-Key": key}
        files = {"file": ("a.txt", b"Annual leave policy.", "text/plain")}
        assert client.post("/api/v1/documents", headers=headers, files=files).status_code == 202
        body = {"query": "leave", "mode": "hybrid"}
        assert client.post("/api/v1/search", headers=headers, json=body).status_code == 200


@pytest.mark.skipif(not Path(FD_DIR).is_dir(), reason="no fd listing on this OS")
def test_repeated_app_lifecycles_do_not_leak_file_handles(tmp_path):
    run_app_once(tmp_path / "warmup")  # first run loads libraries, thread pools, etc.
    before = open_files()
    for i in range(8):
        run_app_once(tmp_path / f"run{i}")
    leaked = open_files() - before
    # A leak of even one Chroma client per run would show up as ~80 here.
    assert leaked <= 4, f"{leaked} file handles leaked over 8 app lifecycles"
