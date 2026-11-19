"""FastAPI routers and HTTP error mapping.

Calls application use cases only. Infrastructure adapters are accessed via the
DI container stored on FastAPI application state.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request, status

from application.use_cases import ResearchUseCases
from domain.interfaces import LLMProvider
from presentation.dto import (
    AgentStepDTO,
    ModelResponse,
    RunListResponse,
    RunResponse,
    StartRunRequest,
    StartRunResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_use_cases(request: Request) -> ResearchUseCases:
    container = request.app.state.container
    return container.use_cases


def _get_llm_provider(request: Request) -> LLMProvider:
    container = request.app.state.container
    return container.llm_provider


def _record_to_response(record) -> RunResponse:
    return RunResponse(
        run_id=record.run_id,
        topic=record.topic,
        model=record.model,
        max_iterations=record.max_iterations,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        report=record.report,
        review=record.review,
        steps=[
            AgentStepDTO(
                step=s.step,
                agent=s.agent,
                status=s.status,
                detail=s.detail,
                timestamp=s.timestamp,
            )
            for s in record.steps
        ],
        error=record.error,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/runs", response_model=StartRunResponse, status_code=status.HTTP_201_CREATED)
async def start_run(request: Request, body: StartRunRequest) -> StartRunResponse:
    """Start a new multi-agent research run."""
    use_cases: ResearchUseCases = _get_use_cases(request)
    try:
        record = await use_cases.start_research(
            topic=body.topic,
            model=body.model,
            max_iterations=body.max_iterations,
        )
        return StartRunResponse(run_id=record.run_id, status=record.status)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to start research run")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(request: Request, run_id: str) -> RunResponse:
    """Inspect a run: status, agent steps, reviewer logs, final report."""
    use_cases: ResearchUseCases = _get_use_cases(request)
    record = await use_cases.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _record_to_response(record)


@router.get("/runs", response_model=RunListResponse)
async def list_runs(request: Request) -> RunListResponse:
    """List past research runs, most recent first."""
    use_cases: ResearchUseCases = _get_use_cases(request)
    records = await use_cases.list_runs()
    return RunListResponse(runs=[_record_to_response(r) for r in records], count=len(records))


@router.get("/models", response_model=ModelResponse)
async def list_models(request: Request) -> ModelResponse:
    """List locally available Ollama models; empty list if Ollama is offline."""
    llm: LLMProvider = _get_llm_provider(request)
    try:
        models = await llm.list_models()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to list Ollama models")
        models = []
    return ModelResponse(models=models)
