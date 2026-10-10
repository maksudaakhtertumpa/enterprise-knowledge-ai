from typing import Literal

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    version: str
    environment: str


class ComponentCheck(BaseModel):
    name: str
    ok: bool
    detail: str | None = None
    # Optional components (the LLM) are reported but don't make the service "not ready":
    # upload and search still work while Ollama is stopped.
    required: bool = True


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: list[ComponentCheck] = Field(default_factory=list)
