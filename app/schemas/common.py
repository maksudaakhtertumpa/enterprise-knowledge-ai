from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str
    message: str
    details: object | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Documented shape of every error response (used in OpenAPI `responses=`)."""

    error: ErrorBody
