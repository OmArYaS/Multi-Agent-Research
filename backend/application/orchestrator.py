"""Application layer: LangGraph orchestration and agent nodes.

Depends ONLY on domain models/interfaces. Never imports Ollama, DuckDuckGo,
or SQLite directly.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from domain.models import (
    AgentStep,
    AnalysisInsights,
    FinalReport,
    ReviewDecision,
    ResearchFindings,
    ResearchPlan,
    RunState,
)
from domain.interfaces import LLMProvider, RunRepository, SearchProvider
from application import prompts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state schema (typed dict, converted to/from RunState at the edges)
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    run_id: str
    topic: str
    model: str
    max_iterations: int
    iteration_count: int
    status: str
    plan: dict
    findings: dict
    analysis: dict
    review: dict
    report: dict
    steps: list
    error: Optional[str]
    messages: list


def _run_state_to_graph(state: RunState) -> GraphState:
    return GraphState(
        run_id=state.run_id,
        topic=state.topic,
        model=state.model,
        max_iterations=state.max_iterations,
        iteration_count=state.iteration_count,
        status=state.status,
        plan=state.plan.model_dump() if state.plan else None,
        findings=state.findings.model_dump() if state.findings else None,
        analysis=state.analysis.model_dump() if state.analysis else None,
        review=state.review.model_dump() if state.review else None,
        report=state.report.model_dump() if state.report else None,
        steps=[s.model_dump() for s in state.steps],
        error=state.error,
        messages=list(state.messages),
    )


def _graph_to_run_state(graph: GraphState) -> RunState:
    def _model(cls, data):
        return cls.model_validate(data) if data else None

    return RunState(
        run_id=graph["run_id"],
        topic=graph["topic"],
        model=graph.get("model", ""),
        max_iterations=graph.get("max_iterations", 2),
        iteration_count=graph.get("iteration_count", 0),
        status=graph.get("status", "pending"),
        plan=_model(ResearchPlan, graph.get("plan")),
        findings=_model(ResearchFindings, graph.get("findings")),
        analysis=_model(AnalysisInsights, graph.get("analysis")),
        review=_model(ReviewDecision, graph.get("review")),
        report=_model(FinalReport, graph.get("report")),
        steps=[AgentStep.model_validate(s) for s in graph.get("steps", [])],
        error=graph.get("error"),
        messages=list(graph.get("messages", [])),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ResearchOrchestrator:
    """Builds and runs the multi-agent LangGraph research workflow."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        search_provider: SearchProvider,
        repository: RunRepository,
        default_model: str = "qwen2.5:7b-instruct-q4_k_m",
        default_max_iterations: int = 2,
    ):
        self.llm = llm_provider
        self.search = search_provider
        self.repo = repository
        self.default_model = default_model
        self.default_max_iterations = default_max_iterations
        self.graph = self._build_graph()

    # -- graph construction ---------------------------------------------

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("research", self._research_node)
        builder.add_node("analyze", self._analyze_node)
        builder.add_node("review", self._review_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "research")
        builder.add_edge("research", "analyze")
        builder.add_edge("analyze", "review")
        builder.add_conditional_edges(
            "review",
            self._route_after_review,
            {"research": "research", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    # -- conditional routing + loop guard -------------------------------

    def _route_after_review(self, state: GraphState) -> str:
        """Decide whether to revise (loop) or finalize.

        Loop guard: only revisit research when the reviewer requests a
        revision AND the iteration cap has not been reached.
        """
        review = state.get("review") or {}
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 2)
        if (
            review.get("verdict") == "revise"
            and iteration_count < max_iterations
        ):
            logger.info(
                "Review requested revision; iteration %d/%d. Looping back to research.",
                iteration_count + 1,
                max_iterations,
            )
            return "research"
        logger.info("Review passed or iteration cap reached. Finalizing report.")
        return "finalize"

    # -- step helpers ---------------------------------------------------

    def _step(self, state: GraphState, agent: str, name: str, detail: str, status: str = "completed") -> dict:
        steps = list(state.get("steps", []))
        steps.append(AgentStep(step=name, agent=agent, detail=detail, status=status).model_dump())
        return {"steps": steps}

    # -- agent nodes ----------------------------------------------------

    async def _plan_node(self, state: GraphState) -> dict:
        topic = state["topic"]
        model = state.get("model") or self.default_model
        logger.info("[plan] topic=%s model=%s", topic, model)
        system = prompts.PLANNER_SYSTEM
        user = prompts.PLANNER_PROMPT.format(topic=topic)
        try:
            plan = await self.llm.generate_structured(
                prompt=user,
                system=system,
                model=model,
                schema=ResearchPlan,
                temperature=0.3,
            )
            plan.topic = topic
            detail = f"Planned {len(plan.sub_questions)} sub-questions."
            return {
                **self._step(state, "Planner", "plan", detail),
                "plan": plan.model_dump(),
                "status": "running",
                "messages": [*state.get("messages", []), f"Planner: {detail}"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Planner node failed")
            return {
                **self._step(state, "Planner", "plan", str(exc), status="failed"),
                "status": "failed",
                "error": str(exc),
            }

    async def _research_node(self, state: GraphState) -> dict:
        topic = state["topic"]
        plan_data = state.get("plan") or {}
        plan = ResearchPlan.model_validate(plan_data) if plan_data else None
        queries = (
            [sq.question for sq in plan.sub_questions]
            if plan and plan.sub_questions
            else list(plan_data.get("search_queries", []))
        )
        if not queries:
            queries = [topic]

        all_results = []
        summaries = []
        for query in queries:
            logger.info("[research] searching: %s", query)
            results = await self.search.search(query)
            all_results.extend(results)
            summaries.append(f"Query: {query}\n" + "\n".join(
                f"- {r.title} ({r.url}): {r.snippet}" for r in results
            ))

        findings = ResearchFindings(topic=topic, results=all_results)
        findings.summary = "\n\n".join(summaries)
        findings.key_sources = [r.url for r in all_results[:5]]
        detail = f"Gathered {len(all_results)} search results across {len(queries)} queries."
        return {
            **self._step(state, "Researcher", "research", detail),
            "findings": findings.model_dump(),
            "messages": [*state.get("messages", []), f"Researcher: {detail}"],
        }

    async def _analyze_node(self, state: GraphState) -> dict:
        topic = state["topic"]
        findings = state.get("findings") or {}
        plan = state.get("plan") or {}
        logger.info("[analyze] topic=%s", topic)
        user = prompts.ANALYST_PROMPT.format(
            topic=topic,
            plan=plan,
            findings=findings,
        )
        try:
            analysis = await self.llm.generate_structured(
                prompt=user,
                system=prompts.ANALYST_SYSTEM,
                model=state.get("model") or self.default_model,
                schema=AnalysisInsights,
                temperature=0.3,
            )
            detail = f"Produced {len(analysis.insights)} insights, {len(analysis.knowledge_gaps)} gaps."
            return {
                **self._step(state, "Analyst", "analyze", detail),
                "analysis": analysis.model_dump(),
                "messages": [*state.get("messages", []), f"Analyst: {detail}"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analyst node failed")
            return {
                **self._step(state, "Analyst", "analyze", str(exc), status="failed"),
                "status": "failed",
                "error": str(exc),
            }

    async def _review_node(self, state: GraphState) -> dict:
        topic = state["topic"]
        analysis = state.get("analysis") or {}
        findings = state.get("findings") or {}
        # Build a draft from analysis + findings for the reviewer to grade.
        draft = (
            f"# Draft Report: {topic}\n\n"
            f"## Insights\n" + "\n".join(f"- {i}" for i in analysis.get("insights", [])) +
            f"\n\n## Key Takeaways\n" + "\n".join(f"- {t}" for t in analysis.get("key_takeaways", [])) +
            f"\n\n## Knowledge Gaps\n" + "\n".join(f"- {g}" for g in analysis.get("knowledge_gaps", [])) +
            f"\n\n## Sources\n" + "\n".join(f"- {s.get('url')}" for s in findings.get("results", []))
        )
        logger.info("[review] topic=%s", topic)
        try:
            review = await self.llm.generate_structured(
                prompt=prompts.REVIEWER_PROMPT.format(topic=topic, draft=draft),
                system=prompts.REVIEWER_SYSTEM,
                model=state.get("model") or self.default_model,
                schema=ReviewDecision,
                temperature=0.3,
            )
            # Increment iteration count only when a revision is requested,
            # so the loop guard can cap total research loops.
            iteration_count = state.get("iteration_count", 0)
            if review.verdict == "revise":
                iteration_count += 1
            detail = f"Verdict: {review.verdict} (score {review.score})."
            return {
                **self._step(state, "Reviewer", "review", detail),
                "review": review.model_dump(),
                "iteration_count": iteration_count,
                "messages": [*state.get("messages", []), f"Reviewer: {detail}"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reviewer node failed")
            return {
                **self._step(state, "Reviewer", "review", str(exc), status="failed"),
                "status": "failed",
                "error": str(exc),
            }

    async def _finalize_node(self, state: GraphState) -> dict:
        topic = state["topic"]
        analysis = state.get("analysis") or {}
        review = state.get("review") or {}
        findings = state.get("findings") or {}
        sources = findings.get("results", [])
        logger.info("[finalize] topic=%s", topic)
        try:
            report = await self.llm.generate_structured(
                prompt=prompts.FINALIZE_PROMPT.format(
                    topic=topic,
                    analysis=analysis,
                    feedback=review.get("feedback", ""),
                    sources=sources,
                ),
                system=prompts.FINALIZE_SYSTEM,
                model=state.get("model") or self.default_model,
                schema=FinalReport,
                temperature=0.3,
            )
            report.title = report.title or f"Research Report: {topic}"
            detail = f"Final report with {len(report.sources)} sources."
            return {
                **self._step(state, "Editor", "finalize", detail),
                "report": report.model_dump(),
                "status": "completed",
                "messages": [*state.get("messages", []), f"Editor: {detail}"],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Finalize node failed")
            return {
                **self._step(state, "Editor", "finalize", str(exc), status="failed"),
                "status": "failed",
                "error": str(exc),
            }

    # -- execution entrypoint ------------------------------------------

    async def run(self, run_id: str, topic: str, model: str | None = None, max_iterations: int | None = None) -> RunState:
        model = model or self.default_model
        max_iterations = max_iterations if max_iterations is not None else self.default_max_iterations

        created_at = self._now()
        initial = RunState(
            run_id=run_id,
            topic=topic,
            model=model,
            max_iterations=max_iterations,
            iteration_count=0,
            status="running",
        )
        await self.repo.save(self._run_record(initial, created_at=created_at))

        graph_state = _run_state_to_graph(initial)
        final_graph = await self.graph.ainvoke(graph_state)
        final_state = _graph_to_run_state(final_graph)

        await self.repo.save(self._run_record(final_state, created_at=created_at))
        return final_state

    def _run_record(self, state: RunState, created_at: str):
        from domain.models import RunRecord

        return RunRecord(
            run_id=state.run_id,
            topic=state.topic,
            model=state.model,
            max_iterations=state.max_iterations,
            status=state.status,
            created_at=created_at,
            updated_at=self._now(),
            report=state.report,
            review=state.review,
            steps=state.steps,
            error=state.error,
        )

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
