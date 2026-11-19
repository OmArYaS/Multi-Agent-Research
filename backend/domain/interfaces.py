"""Domain interfaces (abstract protocols).

Zero framework or vendor imports. Adapters in ``infrastructure/`` implement
these protocols, so swapping Ollama<->OpenAI or DuckDuckGo<->Tavily only
requires changing infrastructure + DI wiring.
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from domain.models import (
    AgentStep,
    BaseModel,
    FinalReport,
    RunRecord,
    SearchResult,
    RunState,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Abstraction over a text-generating model (Ollama, OpenAI, ...)."""

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        """Return raw model text for the given prompt."""

    async def generate_structured(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.3,
        max_retries: int = 2,
    ) -> BaseModel:
        """Return a Pydantic-validated model instance from model output."""

    async def list_models(self) -> List[str]:
        """Return available local model names (empty list if offline)."""


@runtime_checkable
class SearchProvider(Protocol):
    """Abstraction over a web search tool (DuckDuckGo, Tavily, ...)."""

    async def search(self, query: str) -> List[SearchResult]:
        """Return search results for a query."""


@runtime_checkable
class RunRepository(Protocol):
    """Abstraction over run persistence (SQLite, Postgres, ...)."""

    async def save(self, run: RunRecord) -> None:
        """Create or update a run record."""

    async def get(self, run_id: str) -> RunRecord | None:
        """Return a run by id, or None if missing."""

    async def list(self, limit: int = 50) -> List[RunRecord]:
        """Return past runs, most recent first."""

    async def append_step(self, run_id: str, step: AgentStep) -> None:
        """Append an execution step to a run."""

    async def update_state(self, run_id: str, **fields) -> None:
        """Update arbitrary fields on a run record."""
