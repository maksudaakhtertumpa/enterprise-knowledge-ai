"""FastAPI application factory.

`create_app()` builds a fresh app from a Settings object, which lets tests
construct isolated apps (e.g. with a temp data_dir) without touching globals.
Run locally with:  uvicorn app.main:app --reload
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import ask, documents, health, search, workspace
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.db.session import init_db
from app.repositories.documents import DocumentRepository
from app.schemas.common import ErrorResponse
from app.services.container import build_services
from app.services.retrieval.embeddings import check_compatibility

# Multipart framing (boundaries, part headers) on top of the file itself.
_MULTIPART_OVERHEAD = 64 * 1024

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        init_db(app.state.engine)
        with app.state.session_factory() as db:
            interrupted = DocumentRepository(db).mark_interrupted()
            db.commit()
        embedder = app.state.services.embedder
        if settings.embedding_preload:
            # First run downloads the model (~1.2 GB for Qwen3-Embedding-0.6B).
            # Off the event loop so the process stays responsive to signals.
            await asyncio.to_thread(embedder.load)
            check_compatibility(embedder, settings)
        reranker = app.state.services.reranker
        if reranker is not None and settings.reranker_preload:
            await asyncio.to_thread(reranker.load)
        llm = app.state.services.llm
        if llm is not None:
            # Not required for startup: search works without the LLM. Warn early instead.
            llm_ok, llm_detail = await asyncio.to_thread(llm.health)
            if not llm_ok:
                logger.warning("llm_not_ready", extra={"detail": llm_detail})
        logger.info(
            "startup",
            extra={
                "environment": settings.environment,
                "version": __version__,
                "interrupted_jobs_marked_failed": interrupted,
                "embedding": embedder.describe(),
                "reranker": reranker.describe() if reranker else None,
                "llm": llm.describe() if llm else None,
            },
        )
        try:
            yield
        finally:
            app.state.services.close()
            logger.info("shutdown")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="Hybrid-search RAG platform with grounded, citation-verified answers.",
        debug=settings.debug,
        lifespan=lifespan,
        responses={
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Internal error"},
        },
    )
    app.dependency_overrides[get_settings] = lambda: settings
    # Built here without touching disk or network (the model loads at startup),
    # so probes work even before startup completes.
    services = build_services(settings)
    app.state.services = services
    app.state.engine = services.engine
    app.state.session_factory = services.session_factory

    # Middleware: last added runs outermost. Order, outside -> in:
    # RequestContext (request ID, access log) -> BodySizeLimit -> CORS -> app.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=settings.max_upload_bytes + _MULTIPART_OVERHEAD
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    # Probes live at the root so load balancers needn't know the API version.
    app.include_router(health.router)
    app.include_router(workspace.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(search.router, prefix=settings.api_prefix)
    app.include_router(ask.router, prefix=settings.api_prefix)

    return app


app = create_app()
