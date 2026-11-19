"""Domain models: pure Pydantic schemas and state definitions.

This module contains only Pydantic models. It has zero dependencies on
frameworks (FastAPI, LangGraph) or vendors (Ollama, DuckDuckGo, SQLite).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return str(uuid4())[:8]


# ---------------------------------------------------------------------------
# Research artifacts
# ---------------------------------------------------------------------------


class SubQuestion(BaseModel):
    """A single decomposed sub-question produced by the planner."""

    id: str = Field(default_factory=_short_id)
    question: str = Field(..., description="The sub-question to investigate.")
    rationale: str = Field(default="", description="Why this sub-question matters.")


class SearchResult(BaseModel):
    """A single web search result with grounding metadata."""

    title: str = Field(..., description="Page title.")
    url: str = Field(..., description="Canonical URL of the source.")
    snippet: str = Field(default="", description="Search snippet / excerpt.")
    source: str = Field(default="duckduckgo", description="Search provider name.")


class ResearchPlan(BaseModel):
    """Structured plan produced by the Planner agent."""

    topic: str
    sub_questions: List[SubQuestion] = Field(default_factory=list)
    search_queries: List[str] = Field(default_factory=list)


class ResearchFindings(BaseModel):
    """Raw findings gathered by the Researcher agent."""

    topic: str
    results: List[SearchResult] = Field(default_factory=list)
    summary: str = Field(default="", description="Synthesis of retrieved snippets.")
    key_sources: List[str] = Field(default_factory=list, description="Source URLs judged most relevant.")


class AnalysisInsights(BaseModel):
    """Structured insights produced by the Analyst agent."""

    insights: List[str] = Field(default_factory=list)
    key_takeaways: List[str] = Field(default_factory=list)
    comparisons: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list, description="Open questions / missing info.")


class ReviewDecision(BaseModel):
    """Structured review verdict produced by the Reviewer/Critic agent."""

    verdict: Literal["pass", "revise"] = Field(
        ..., description="'pass' to finalize, 'revise' to loop back."
    )
    score: int = Field(default=0, ge=0, le=100, description="Quality score 0-100.")
    feedback: str = Field(..., description="Human-readable review feedback.")
    missing_elements: List[str] = Field(default_factory=list)
    required_changes: List[str] = Field(default_factory=list, description="Actionable revision steps.")


class FinalReport(BaseModel):
    """Final research report with citations."""

    title: str
    executive_summary: str = Field(default="")
    body: str = Field(default="", description="Markdown report body.")
    key_takeaways: List[str] = Field(default_factory=list)
    sources: List[SearchResult] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list, description="Citation markers used in body.")


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------


class AgentStep(BaseModel):
    """A single logged step of agent execution."""

    step: str = Field(..., description="Node / step name.")
    agent: str = Field(..., description="Agent that produced the step.")
    status: Literal["pending", "in_progress", "completed", "failed"] = "completed"
    detail: str = Field(default="")
    timestamp: str = Field(default_factory=_utc_now)


RunStatus = Literal["pending", "running", "completed", "failed"]


class RunState(BaseModel):
    """Mutable LangGraph state for a single research run.

    This is the source of truth passed between graph nodes. It is serialized
    to/from plain dicts when interacting with LangGraph.
    """

    run_id: str
    topic: str
    model: str
    max_iterations: int = Field(default=2, ge=1)
    iteration_count: int = Field(default=0, ge=0)
    status: RunStatus = "pending"
    plan: Optional[ResearchPlan] = None
    findings: Optional[ResearchFindings] = None
    analysis: Optional[AnalysisInsights] = None
    review: Optional[ReviewDecision] = None
    report: Optional[FinalReport] = None
    steps: List[AgentStep] = Field(default_factory=list)
    error: Optional[str] = None
    messages: List[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "RunState":
        return cls.model_validate(data)

    def add_step(self, step: str, agent: str, detail: str, status: str = "completed") -> None:
        self.steps.append(AgentStep(step=step, agent=agent, detail=detail, status=status))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Persisted record
# ---------------------------------------------------------------------------


class RunRecord(BaseModel):
    """A persisted research run, suitable for storage and API responses."""

    run_id: str
    topic: str
    model: str
    max_iterations: int
    status: RunStatus
    created_at: str
    updated_at: str
    report: Optional[FinalReport] = None
    review: Optional[ReviewDecision] = None
    steps: List[AgentStep] = Field(default_factory=list)
    error: Optional[str] = None
