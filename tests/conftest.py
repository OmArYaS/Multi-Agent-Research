"""Pytest configuration and shared fixtures for offline tests."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient

from domain.models import (
    AgentStep,
    AnalysisInsights,
    FinalReport,
    ResearchPlan,
    ReviewDecision,
    RunRecord,
    SearchResult,
    SubQuestion,
)
from domain.interfaces import LLMProvider, RunRepository, SearchProvider
from presentation.api import router


class FakeLLMProvider:
    """Deterministic fake LLM provider for offline tests.

    Returns structured instances based on the requested Pydantic schema.
    """

    def __init__(self, always_revise: bool = False):
        self.always_revise = always_revise
        self.calls: List[str] = []

    async def generate(self, prompt, system=None, model=None, temperature=0.3, json_mode=False):
        self.calls.append(prompt[:80])
        return "fake raw text"

    async def generate_structured(self, prompt, system=None, model=None, schema=None, temperature=0.3, max_retries=2):
        self.calls.append(prompt[:80])
        if schema is ReviewDecision:
            return ReviewDecision(
                verdict="revise" if self.always_revise else "pass",
                score=40 if self.always_revise else 90,
                feedback="needs more sources" if self.always_revise else "looks good",
                missing_elements=["citations"] if self.always_revise else [],
                required_changes=["add citations"] if self.always_revise else [],
            )
        if schema is FinalReport:
            return FinalReport(
                title="Fake Report",
                executive_summary="summary",
                body="# Report\n\nFake body.",
                key_takeaways=["tk1"],
                sources=[SearchResult(title="S", url="http://example.com", snippet="s")],
                citations=["[1]"],
            )
        if schema is AnalysisInsights:
            return AnalysisInsights(
                insights=["i1"],
                key_takeaways=["tk1"],
                comparisons=["c1"],
                knowledge_gaps=["g1"],
            )
        if schema is ResearchPlan:
            return ResearchPlan(
                topic="",
                sub_questions=[SubQuestion(question="q1", rationale="r1")],
                search_queries=["q1"],
            )
        raise ValueError(f"unexpected schema {schema}")

    async def list_models(self):
        return ["qwen2.5:7b-instruct-q4_k_m", "gemma2:2b"]


class FakeSearchProvider:
    def __init__(self, results=None):
        self.results = results or [
            SearchResult(title="Result A", url="http://example.com/a", snippet="snippet a"),
            SearchResult(title="Result B", url="http://example.com/b", snippet="snippet b"),
        ]
        self.call_count = 0

    async def search(self, query: str):
        self.call_count += 1
        return list(self.results)


class FakeRunRepository:
    def __init__(self):
        self._runs = {}

    async def save(self, run: RunRecord):
        self._runs[run.run_id] = run

    async def get(self, run_id: str):
        return self._runs.get(run_id)

    async def list(self, limit: int = 50):
        return list(self._runs.values())[:limit]

    async def append_step(self, run_id: str, step: AgentStep):
        pass

    async def update_state(self, run_id: str, **fields):
        pass


class FakeUseCases:
    def __init__(self, repo: FakeRunRepository, always_revise: bool = False):
        self.repo = repo
        self.always_revise = always_revise
        self.started = []

    async def start_research(self, topic, model=None, max_iterations=None):
        run_id = "run-123"
        record = RunRecord(
            run_id=run_id,
            topic=topic,
            model=model or "qwen2.5:7b-instruct-q4_k_m",
            max_iterations=max_iterations or 2,
            status="completed",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            report=FinalReport(
                title="Report",
                executive_summary="sum",
                body="# Report",
                key_takeaways=["tk"],
                sources=[SearchResult(title="S", url="http://example.com", snippet="s")],
                citations=["[1]"],
            ),
            steps=[AgentStep(step="plan", agent="Planner", status="completed", detail="done")],
        )
        self.started.append((topic, model, max_iterations))
        await self.repo.save(record)
        return record

    async def get_run(self, run_id: str):
        return await self.repo.get(run_id)

    async def list_runs(self, limit: int = 50):
        return await self.repo.list(limit=limit)


class FakeContainer:
    def __init__(self, use_cases: FakeUseCases, llm: FakeLLMProvider):
        self.use_cases = use_cases
        self.llm_provider = llm


@asynccontextmanager
async def dummy_lifespan(app):
    yield


def make_app(container: FakeContainer) -> TestClient:
    from fastapi import FastAPI

    app = FastAPI()
    app.state.container = container
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(scope="session")
def repo():
    return FakeRunRepository()


@pytest.fixture(scope="session")
def llm():
    return FakeLLMProvider()


@pytest.fixture(scope="session")
def search():
    return FakeSearchProvider()


@pytest.fixture(scope="session")
def always_revise_llm():
    return FakeLLMProvider(always_revise=True)
