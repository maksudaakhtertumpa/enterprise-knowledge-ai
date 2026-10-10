"""Liveness and readiness probes.

- /health        → "is the process up?"  (cheap; used by Docker/load balancers)
- /health/ready  → "can it serve traffic?" (checks dependencies; grows each phase:
                    data dir, database, vector store, embedding model; the LLM in Phase 5)
"""

import os
import tempfile

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app import __version__
from app.core.config import Settings, get_settings
from app.schemas.health import ComponentCheck, LivenessResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=LivenessResponse, summary="Liveness probe")
def health(settings: Settings = Depends(get_settings)) -> LivenessResponse:
    return LivenessResponse(
        app=settings.app_name, version=__version__, environment=settings.environment
    )


def _check_data_dir(settings: Settings) -> ComponentCheck:
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=settings.data_dir, prefix=".probe-")
        os.close(fd)
        os.remove(path)
        return ComponentCheck(name="data_dir", ok=True)
    except OSError as exc:
        return ComponentCheck(name="data_dir", ok=False, detail=type(exc).__name__)


def _check_database(request: Request) -> ComponentCheck:
    try:
        with request.app.state.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return ComponentCheck(name="database", ok=True)
    except Exception as exc:  # noqa: BLE001 - any failure means "not ready"
        return ComponentCheck(name="database", ok=False, detail=type(exc).__name__)


def _check_vector_store(request: Request) -> ComponentCheck:
    try:
        request.app.state.services.vectors.heartbeat()
        return ComponentCheck(name="vector_store", ok=True)
    except Exception as exc:  # noqa: BLE001
        return ComponentCheck(name="vector_store", ok=False, detail=type(exc).__name__)


def _check_embedder(request: Request, settings: Settings) -> ComponentCheck:
    embedder = request.app.state.services.embedder
    if embedder.is_loaded:
        return ComponentCheck(name="embedding_model", ok=True, detail=embedder.model_name)
    # With preload off, the model loads on first use; that's a valid, ready state.
    ok = not settings.embedding_preload
    return ComponentCheck(name="embedding_model", ok=ok, detail="not loaded")


def _check_reranker(request: Request, settings: Settings) -> ComponentCheck:
    reranker = request.app.state.services.reranker
    if reranker is None:
        return ComponentCheck(name="reranker", ok=True, detail="disabled")
    if reranker.is_loaded:
        return ComponentCheck(name="reranker", ok=True, detail=reranker.model_name)
    return ComponentCheck(name="reranker", ok=not settings.reranker_preload, detail="not loaded")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"model": ReadinessResponse, "description": "A dependency is unavailable"}},
)
def ready(request: Request, settings: Settings = Depends(get_settings)) -> JSONResponse:
    checks = [
        _check_data_dir(settings),
        _check_database(request),
        _check_vector_store(request),
        _check_embedder(request, settings),
        _check_reranker(request, settings),
    ]
    is_ready = all(c.ok for c in checks)
    body = ReadinessResponse(status="ready" if is_ready else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if is_ready else 503, content=body.model_dump())
