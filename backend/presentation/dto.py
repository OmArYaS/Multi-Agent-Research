"""HTTP request/response DTOs for the presentation layer."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from domain.models import FinalReport, ReviewDecision, RunStatus


class StartRunRequest(BaseModel):
    topic: str = Field(..., description="Research topic.")
    model: Optional[str] = Field(
        default=None, description="Optional Ollama model override."
    )
    max_iterations: Optional[int] = Field(
        default=None, ge=1, description="Max revision iterations (default 2)."
    )


class StartRunResponse(BaseModel):
    run_id: str
    status: RunStatus = "running"


class AgentStepDTO(BaseModel):
    step: str
    agent: str
    status: str
    detail: str
    timestamp: str


class RunResponse(BaseModel):
    run_id: str
    topic: str
    model: str
    max_iterations: int
    status: RunStatus
    created_at: str
    updated_at: str
    report: Optional[FinalReport] = None
    review: Optional[ReviewDecision] = None
    steps: List[AgentStepDTO] = []
    error: Optional[str] = None


class RunListResponse(BaseModel):
    runs: List[RunResponse]
    count: int


class ModelResponse(BaseModel):
    models: List[str]
