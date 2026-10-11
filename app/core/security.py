"""Workspace API-key authentication.

Keys look like `eka_<43 url-safe chars>` (256 bits of randomness). Because they
are random rather than human-chosen, a fast SHA-256 hash is the right storage
choice — slow password hashes (bcrypt/argon2) defend against guessing
low-entropy secrets, which doesn't apply here.

Every authenticated request resolves to exactly one workspace. Route handlers
receive a `WorkspaceContext`, never a raw ID from the client.
"""

import hashlib
import secrets
from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.core.ratelimit import RateLimits, get_rate_limits
from app.db.models import ApiKey, Workspace
from app.db.session import get_db

KEY_PREFIX = "eka_"
_header = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="Workspace API key (eka_...)"
)


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    id: str
    name: str


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_workspace_with_key(
    db: Session, name: str, label: str | None = None
) -> tuple[Workspace, str]:
    """Create a workspace and its first key. Returns the plaintext key — show it once."""
    workspace = Workspace(name=name)
    db.add(workspace)
    db.flush()
    key = issue_api_key(db, workspace.id, label=label)
    return workspace, key


def issue_api_key(db: Session, workspace_id: str, label: str | None = None) -> str:
    key = generate_api_key()
    db.add(
        ApiKey(
            workspace_id=workspace_id,
            key_hash=hash_api_key(key),
            key_prefix=key[:12],
            label=label,
        )
    )
    return key


def get_workspace(
    request: Request,
    api_key: str | None = Depends(_header),
    db: Session = Depends(get_db),
    limits: RateLimits = Depends(get_rate_limits),
) -> WorkspaceContext:
    # An address that keeps presenting bad keys waits before any key is checked.
    client = limits.client_address(request)
    limits.check_auth_allowed(client)

    # One generic message for every failure, so callers can't probe which part was wrong.
    if not api_key or not api_key.startswith(KEY_PREFIX) or len(api_key) > 128:
        limits.record_auth_failure(client)
        raise UnauthorizedError("Missing or invalid API key")

    row = db.execute(
        select(Workspace.id, Workspace.name)
        .join(ApiKey, ApiKey.workspace_id == Workspace.id)
        .where(ApiKey.key_hash == hash_api_key(api_key), ApiKey.revoked_at.is_(None))
    ).first()
    if row is None:
        limits.record_auth_failure(client)
        raise UnauthorizedError("Missing or invalid API key")

    ctx = WorkspaceContext(id=row.id, name=row.name)
    request.state.workspace_id = ctx.id  # picked up by the access log
    return ctx


def rate_limited(name: str):
    """Dependency: authenticate, then spend one token from this workspace's `name` bucket."""

    def dependency(
        workspace: WorkspaceContext = Depends(get_workspace),
        limits: RateLimits = Depends(get_rate_limits),
    ) -> WorkspaceContext:
        limits.check(name, workspace.id)
        return workspace

    return dependency
