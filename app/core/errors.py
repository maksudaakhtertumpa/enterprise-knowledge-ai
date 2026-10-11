"""One error envelope for the whole API.

Every failure — validation, 404, domain error, or crash — returns:
    {"error": {"code": "...", "message": "...", "details": ..., "request_id": "..."}}
Unhandled exceptions never leak stack traces or internals to the client.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_ctx

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for expected, client-facing domain errors."""

    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


def _envelope(
    status: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        headers=headers,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
                "request_id": request_id_ctx.get(),
            }
        },
    )


async def _app_error(_: Request, exc: AppError) -> JSONResponse:
    return _envelope(
        exc.status_code, exc.code, exc.message, exc.details, getattr(exc, "headers", None)
    )


async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = {
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        413: "payload_too_large",
        429: "rate_limited",
    }
    return _envelope(exc.status_code, code.get(exc.status_code, "http_error"), str(exc.detail))


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Drop the raw 'input' and 'ctx' values so we never echo submitted data back.
    details = [
        {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
    ]
    return _envelope(422, "validation_error", "Request validation failed", details)


async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", exc_info=exc)
    return _envelope(500, "internal_error", "An unexpected error occurred")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)
