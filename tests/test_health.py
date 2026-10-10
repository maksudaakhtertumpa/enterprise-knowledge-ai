from fastapi.testclient import TestClient

from app import __version__
from app.core.config import Settings
from app.main import create_app


def test_liveness_returns_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "status": "ok",
        "app": "Enterprise Knowledge AI",
        "version": __version__,
        "environment": "test",
    }


def test_readiness_ok_when_data_dir_writable(client: TestClient, settings: Settings) -> None:
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert settings.uploads_dir.is_dir(), "startup should create the uploads directory"


def test_readiness_503_when_data_dir_unusable(tmp_path) -> None:
    # A regular *file* where the data directory should be makes mkdir fail.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    settings = Settings(
        environment="test",
        data_dir=blocker / "data",
        embedding_backend="hashing",
        reranker_backend="hashing",
        _env_file=None,
    )
    app = create_app(settings)
    # No `with`: skip lifespan, which would also fail to create the directory.
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/health/ready")
    app.state.services.close()
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"][0] == {"name": "data_dir", "ok": False, "detail": "NotADirectoryError"}


def test_openapi_schema_is_served(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/health" in paths and "/health/ready" in paths
