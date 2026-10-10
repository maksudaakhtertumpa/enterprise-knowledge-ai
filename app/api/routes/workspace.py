from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.security import WorkspaceContext, get_workspace
from app.schemas.common import ErrorResponse
from app.schemas.documents import WorkspaceOut

router = APIRouter(tags=["workspace"])


@router.get(
    "/workspace",
    response_model=WorkspaceOut,
    summary="Identify the workspace this API key belongs to",
    responses={401: {"model": ErrorResponse}},
)
def current_workspace(
    workspace: Annotated[WorkspaceContext, Depends(get_workspace)],
) -> WorkspaceOut:
    return WorkspaceOut(id=workspace.id, name=workspace.name)
