"""FastAPI application entrypoint.

Wires the DI container and exposes the research API. Ollama being offline is
handled gracefully (empty model list, clear errors on generation).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from infrastructure.di import build_container, get_container
from presentation.api import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = build_container()
    app.state.container = container
    logger.info("Multi-Agent Research Orchestrator started")
    yield
    await container.shutdown()
    logger.info("Multi-Agent Research Orchestrator shut down")


app = FastAPI(
    title="Multi-Agent Research & Workflow Orchestrator",
    version="1.0.0",
    description="Clean-architecture, multi-agent research workflow with LangGraph and Ollama.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
