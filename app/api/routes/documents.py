"""Document management endpoints. All are scoped to the caller's workspace."""

import logging
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ConflictError, NotFoundError
from app.core.security import WorkspaceContext, get_workspace, rate_limited
from app.db.models import Document, DocumentStatus
from app.db.session import get_db
from app.repositories.documents import DocumentRepository
from app.schemas.common import ErrorResponse
from app.schemas.documents import ChunkList, ChunkOut, DocumentList, DocumentOut
from app.services.ingestion.pipeline import ingest_document
from app.services.ingestion.validation import (
    UploadRejected,
    file_type_for,
    sanitize_filename,
    verify_content,
)
from app.services.retrieval.keyword import KeywordIndex
from app.services.storage import FileTooLarge, LocalFileStorage

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={401: {"model": ErrorResponse, "description": "Missing or invalid API key"}},
)

Workspace = Annotated[WorkspaceContext, Depends(get_workspace)]
IngestWorkspace = Annotated[WorkspaceContext, Depends(rate_limited("ingest"))]
DB = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


class UploadError(AppError):
    def __init__(self, exc: UploadRejected) -> None:
        super().__init__(exc.message)
        self.status_code = exc.status_code
        self.code = exc.code


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


def _get_or_404(repo: DocumentRepository, workspace_id: str, document_id: uuid.UUID) -> Document:
    doc = repo.get(workspace_id, str(document_id))
    if doc is None:
        raise NotFoundError("Document not found")
    return doc


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentOut,
    summary="Upload a document for ingestion",
    description=(
        "Accepts PDF, DOCX or TXT. The file is validated and stored, then processed "
        "in the background; poll `GET /documents/{id}` until `status` is `ready` or `failed`."
    ),
    responses={
        409: {"model": ErrorResponse, "description": "Identical file already in this workspace"},
        413: {"model": ErrorResponse, "description": "File exceeds the upload limit"},
        415: {"model": ErrorResponse, "description": "Unsupported or mismatched file type"},
        429: {"model": ErrorResponse, "description": "Rate limit reached; see Retry-After"},
    },
)
def upload_document(
    request: Request,
    background: BackgroundTasks,
    workspace: IngestWorkspace,
    db: DB,
    settings: AppSettings,
    file: UploadFile = File(..., description="PDF, DOCX or TXT"),
) -> Document:
    try:
        filename = sanitize_filename(file.filename)
        file_type = file_type_for(filename)
    except UploadRejected as exc:
        raise UploadError(exc) from exc

    storage = LocalFileStorage(settings.uploads_dir)
    document_id = str(uuid.uuid4())
    try:
        stored = storage.save(
            file.file,
            workspace_id=workspace.id,
            document_id=document_id,
            ext=file_type.value,
            max_bytes=settings.max_upload_bytes,
        )
    except FileTooLarge as exc:
        raise PayloadTooLargeError(
            f"File exceeds the {settings.max_upload_mb} MB upload limit."
        ) from exc

    def discard() -> None:
        storage.delete(stored.key)

    if stored.size_bytes == 0:
        discard()
        raise AppError("The file is empty.")  # 400 app_error
    try:
        verify_content(
            storage.path_for(stored.key),
            file_type,
            max_docx_uncompressed=settings.max_docx_uncompressed_mb * 1024 * 1024,
        )
    except UploadRejected as exc:
        discard()
        raise UploadError(exc) from exc

    repo = DocumentRepository(db)
    existing = repo.find_by_hash(workspace.id, stored.sha256)
    if existing is not None:
        discard()
        raise ConflictError(
            "This file has already been uploaded to this workspace.",
            details={"existing_document_id": existing.id},
        )

    doc = Document(
        id=document_id,
        workspace_id=workspace.id,
        filename=filename,
        file_type=file_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=stored.key,
        status=DocumentStatus.PENDING,
        meta={"client_content_type": (file.content_type or "")[:100]},
    )
    db.add(doc)
    try:
        db.commit()
    except IntegrityError as exc:  # two identical uploads raced past find_by_hash
        db.rollback()
        discard()
        raise ConflictError("This file has already been uploaded to this workspace.") from exc

    background.add_task(ingest_document, request.app.state.services, doc.id)
    logger.info(
        "document_uploaded",
        extra={"document_id": doc.id, "file_type": file_type.value, "size": doc.size_bytes},
    )
    return doc


@router.get("", response_model=DocumentList, summary="List documents in this workspace")
def list_documents(
    workspace: Workspace,
    db: DB,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
) -> DocumentList:
    items, total = DocumentRepository(db).list_documents(
        workspace.id, limit=limit, offset=offset, status=status_filter
    )
    return DocumentList(
        items=[DocumentOut.model_validate(d) for d in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentOut,
    summary="Get a document and its ingestion status",
    responses={404: {"model": ErrorResponse}},
)
def get_document(document_id: uuid.UUID, workspace: Workspace, db: DB) -> Document:
    return _get_or_404(DocumentRepository(db), workspace.id, document_id)


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkList,
    summary="Inspect a document's chunks",
    responses={404: {"model": ErrorResponse}},
)
def list_chunks(
    document_id: uuid.UUID,
    workspace: Workspace,
    db: DB,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ChunkList:
    repo = DocumentRepository(db)
    doc = _get_or_404(repo, workspace.id, document_id)
    items, total = repo.list_chunks(workspace.id, doc.id, limit=limit, offset=offset)
    return ChunkList(
        document_id=doc.id,
        items=[ChunkOut.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{document_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentOut,
    summary="Re-run ingestion (e.g. after changing chunk settings)",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        429: {"model": ErrorResponse, "description": "Rate limit reached; see Retry-After"},
    },
)
def reprocess_document(
    document_id: uuid.UUID,
    request: Request,
    background: BackgroundTasks,
    workspace: IngestWorkspace,
    db: DB,
    settings: AppSettings,
) -> Document:
    doc = _get_or_404(DocumentRepository(db), workspace.id, document_id)
    if doc.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING):
        raise ConflictError("The document is already queued or being processed.")
    doc.status = DocumentStatus.PENDING
    doc.error = None
    db.commit()
    background.add_task(ingest_document, request.app.state.services, doc.id)
    return doc


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document, its chunks and the stored file",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
def delete_document(
    document_id: uuid.UUID, request: Request, workspace: Workspace, db: DB, settings: AppSettings
) -> Response:
    doc = _get_or_404(DocumentRepository(db), workspace.id, document_id)
    if doc.status is DocumentStatus.PROCESSING:
        raise ConflictError("The document is being processed; try again shortly.")
    key = doc.storage_key
    KeywordIndex(db).delete_document(workspace.id, doc.id)  # same transaction as the row
    db.delete(doc)  # chunks go with it (ON DELETE CASCADE)
    db.commit()
    # Vectors and file are removed after the commit. A crash in between leaves
    # orphans that are harmless: search hydrates from SQLite and drops IDs it
    # can't find, and orphan files are sweepable. Never the reverse (a DB row
    # pointing at missing data).
    try:
        request.app.state.services.vectors.delete_document(workspace.id, doc.id)
    except Exception:  # noqa: BLE001
        logger.warning("vector_delete_failed", extra={"document_id": str(document_id)})
    LocalFileStorage(settings.uploads_dir).delete(key)
    logger.info("document_deleted", extra={"document_id": str(document_id)})
    return Response(status_code=status.HTTP_204_NO_CONTENT)
