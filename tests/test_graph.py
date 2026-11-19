"""Tests for LangGraph state transitions, conditional routing, and the max_iterations loop guard."""

import pytest

from application.orchestrator import ResearchOrchestrator
from tests.conftest import FakeLLMProvider, FakeSearchProvider, FakeRunRepository


@pytest.fixture
def orchestrator(llm, search, repo):
    return ResearchOrchestrator(
        llm_provider=llm,
        search_provider=search,
        repository=repo,
        default_model="qwen2.5:7b-instruct-q4_k_m",
        default_max_iterations=2,
    )


@pytest.mark.asyncio
async def test_happy_path_completes(orchestrator, llm, search, repo):
    search.call_count = 0
    record = await orchestrator.run(
        run_id="run-happy", topic="renewable energy", max_iterations=2
    )

    assert record.status == "completed"
    assert record.report is not None
    assert record.report.title == "Fake Report"
    assert len(record.steps) >= 5  # plan, research, analyze, review, finalize
    step_names = [s.step for s in record.steps]
    assert "plan" in step_names
    assert "research" in step_names
    assert "analyze" in step_names
    assert "review" in step_names
    assert "finalize" in step_names
    # Single research pass because reviewer passed.
    assert search.call_count == 1
    assert record.iteration_count == 0


@pytest.mark.asyncio
async def test_revise_loops_back_under_cap(orchestrator, always_revise_llm, search, repo):
    orchestrator.llm = always_revise_llm
    search.call_count = 0

    # max_iterations = 2 -> initial research, then one revision pass loops back.
    record = await orchestrator.run(
        run_id="run-revise-1", topic="quantum computing", max_iterations=2
    )
    assert record.status == "completed"
    assert record.report is not None
    assert record.iteration_count == 2
    assert search.call_count == 2  # initial + one re-research pass


@pytest.mark.asyncio
async def test_max_iteration_cap_prevents_infinite_loop(
    orchestrator, always_revise_llm, search, repo
):
    orchestrator.llm = always_revise_llm
    search.call_count = 0

    # max_iterations = 1 -> only the initial research pass; the cap stops any
    # further re-research even though the reviewer keeps asking to revise.
    record = await orchestrator.run(
        run_id="run-cap-1", topic="deepseek r1", max_iterations=1
    )
    assert record.status == "completed"
    assert record.report is not None
    assert record.iteration_count == 1
    assert search.call_count == 1  # no re-research beyond the cap
