"""Model-agnostic prompt templates.

All templates are plain strings with named placeholders. They contain no
provider-specific formatting (no Qwen/Gemma stop tokens, no special chat
templates) so any Ollama model can be used interchangeably.
"""

from __future__ import annotations

PLANNER_SYSTEM = (
    "You are a careful research planner. Break the user's research topic into "
    "3-4 focused sub-questions and corresponding web search queries. "
    "Return ONLY valid JSON matching the requested schema."
)

PLANNER_PROMPT = (
    "Research topic: {topic}\n\n"
    "Produce a structured research plan with:\n"
    "- sub_questions: 3-4 decomposed sub-questions, each with a 'question' and 'rationale'\n"
    "- search_queries: 3-4 concise web search queries, one per sub-question\n\n"
    "Return ONLY valid JSON."
)

RESEARCHER_SYSTEM = (
    "You are a research assistant. Summarize the provided web search results "
    "into findings, citing sources by URL. Return ONLY valid JSON."
)

RESEARCHER_PROMPT = (
    "Topic: {topic}\n"
    "Sub-question: {sub_question}\n"
    "Search results (title, url, snippet):\n{results}\n\n"
    "Produce:\n"
    "- summary: a concise synthesis of what the results say about this sub-question\n"
    "- key_sources: the most relevant source URLs\n"
    "Return ONLY valid JSON."
)

ANALYST_SYSTEM = (
    "You are a research analyst. Synthesize gathered findings into structured "
    "insights, key takeaways, comparisons, and knowledge gaps. "
    "Return ONLY valid JSON."
)

ANALYST_PROMPT = (
    "Topic: {topic}\n"
    "Plan: {plan}\n"
    "Findings:\n{findings}\n\n"
    "Produce:\n"
    "- insights: bullet insights drawn from the findings\n"
    "- key_takeaways: the most important takeaways\n"
    "- comparisons: any comparisons or trade-offs evident\n"
    "- knowledge_gaps: open questions or missing information\n"
    "Return ONLY valid JSON."
)

REVIEWER_SYSTEM = (
    "You are a rigorous research reviewer/critic. Evaluate the draft report "
    "against completeness, structure, grounding in retrieved sources, and "
    "citations. Return ONLY valid JSON."
)

REVIEWER_PROMPT = (
    "Topic: {topic}\n"
    "Draft report:\n{draft}\n\n"
    "Evaluate and return:\n"
    "- verdict: 'pass' if the report is complete and well-grounded, else 'revise'\n"
    "- score: quality score 0-100\n"
    "- feedback: specific, actionable feedback\n"
    "- missing_elements: anything still missing\n"
    "- required_changes: concrete steps to fix the draft\n"
    "Return ONLY valid JSON."
)

FINALIZE_SYSTEM = (
    "You are a research editor. Produce a polished final Markdown report "
    "with an executive summary, body, key takeaways, and a sources section "
    "with clickable links. Return ONLY valid JSON."
)

FINALIZE_PROMPT = (
    "Topic: {topic}\n"
    "Analysis:\n{analysis}\n"
    "Review feedback:\n{feedback}\n"
    "Sources:\n{sources}\n\n"
    "Produce:\n"
    "- title\n"
    "- executive_summary\n"
    "- body: full Markdown report with inline citations like [Source 1]\n"
    "- key_takeaways\n"
    "- sources: list of {{title, url}}\n"
    "- citations: citation markers used\n"
    "Return ONLY valid JSON."
)


def get_prompt(name: str) -> str:
    """Retrieve a prompt template by name (system or user variant)."""
    system = globals().get(f"{name.upper()}_SYSTEM")
    user = globals().get(f"{name.upper()}_PROMPT")
    return {"system": system, "user": user}
