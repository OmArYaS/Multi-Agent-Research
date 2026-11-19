"""Dependency injection wiring.

Assembles infrastructure adapters and application use cases. This is the only
place that knows about concrete implementations; domain and application
depend solely on interfaces.
"""

from __future__ import annotations

from functools import lru_cache

from application.use_cases import ResearchUseCases
from domain.interfaces import LLMProvider, RunRepository, SearchProvider
from infrastructure.config import Settings
from infrastructure.llm.ollama_provider import OllamaLLMProvider
from infrastructure.parser import StructuredOutputParser
from infrastructure.repository.sqlite_repository import SQLiteRunRepository
from infrastructure.search.duckduckgo_provider import DuckDuckGoSearchProvider
from application.orchestrator import ResearchOrchestrator


class Container:
    """Holds wired dependencies."""

    def __init__(
        self,
        settings: Settings,
        llm_provider: LLMProvider,
        search_provider: SearchProvider,
        repository: RunRepository,
        parser: StructuredOutputParser,
        orchestrator: ResearchOrchestrator,
        use_cases: ResearchUseCases,
    ):
        self.settings = settings
        self.llm_provider = llm_provider
        self.search_provider = search_provider
        self.repository = repository
        self.parser = parser
        self.orchestrator = orchestrator
        self.use_cases = use_cases

    async def shutdown(self) -> None:
        if hasattr(self.repository, "close"):
            self.repository.close()


def build_container(settings: Settings | None = None) -> Container:
    """Build the full dependency graph from settings."""
    settings = settings or Settings()

    parser = StructuredOutputParser()
    llm_provider: LLMProvider = OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        parser=parser,
    )
    search_provider: SearchProvider = DuckDuckGoSearchProvider(
        max_results=settings.search_max_results
    )
    repository: RunRepository = SQLiteRunRepository(db_path=settings.db_path)
    orchestrator = ResearchOrchestrator(
        llm_provider=llm_provider,
        search_provider=search_provider,
        repository=repository,
        default_model=settings.ollama_model,
        default_max_iterations=settings.max_iterations,
    )
    use_cases = ResearchUseCases(
        orchestrator=orchestrator,
        repository=repository,
    )
    return Container(
        settings=settings,
        llm_provider=llm_provider,
        search_provider=search_provider,
        repository=repository,
        parser=parser,
        orchestrator=orchestrator,
        use_cases=use_cases,
    )


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Process-wide cached container (used by the FastAPI app)."""
    return build_container()
