"""Request-ID, access-log and body-size middleware, written as pure ASGI.

Pure ASGI (rather than Starlette's BaseHTTPMiddleware) keeps the request_id
context variable visible to the outermost 500 handler, so even crash responses
carry the ID the client can quote in a bug report.
"""

import json
import logging
import re
import time
import uuid

from fastapi import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_ctx

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "x-request-id"
# Accept a client-supplied ID only if it is short and boring; otherwise mint one.
# This prevents log injection through the header.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1")
        request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex
        request_id_ctx.set(request_id)

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    # Set by the auth dependency; lets us audit who did what.
                    "workspace_id": scope.get("state", {}).get("workspace_id"),
                },
            )


class BodyTooLargeError(HTTPException):
    """Subclass of HTTPException so FastAPI's body parser re-raises it unchanged."""

    def __init__(self) -> None:
        super().__init__(status_code=413, detail="Request body is too large")


class BodySizeLimitMiddleware:
    """Reject oversized request bodies *before* they are buffered.

    Without this, Starlette would spool an entire multi-gigabyte upload to a
    temp file before our handler could check its size. Checks Content-Length
    up front and also counts streamed bytes (for chunked uploads that omit it).
    In production, the reverse proxy should enforce the same limit (Phase 8).
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await _send_413(send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLargeError
            return message

        await self.app(scope, limited_receive, send)


async def _send_413(send: Send) -> None:
    body = json.dumps(
        {
            "error": {
                "code": "payload_too_large",
                "message": "Request body is too large",
                "details": None,
                "request_id": request_id_ctx.get(),
            }
        }
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                # The client is still sending; tell it we're not reading the rest.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
