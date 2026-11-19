"""DuckDuckGo search adapter implementing the domain ``SearchProvider`` interface."""

from __future__ import annotations

import asyncio
import logging
from typing import List

from domain.interfaces import SearchProvider
from domain.models import SearchResult

logger = logging.getLogger(__name__)


class DuckDuckGoSearchProvider:
    """Free web search via ``duckduckgo-search`` (no API key required).

    Network failures degrade gracefully to an empty result list so the
    orchestrator keeps working offline / in tests.
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    async def search(self, query: str) -> List[SearchResult]:
        """Run a text search and return results."""

        def _search() -> List[SearchResult]:
            # Imported lazily so tests can run without network/DDGS present.
            from duckduckgo_search import DDGS  # type: ignore

            results: List[SearchResult] = []
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=self.max_results):
                    results.append(
                        SearchResult(
                            title=item.get("title", "") or "",
                            url=item.get("href", "") or item.get("url", "") or "",
                            snippet=item.get("body", "") or "",
                            source="duckduckgo",
                        )
                    )
            return results

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _search)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DuckDuckGo search failed for query %r; returning empty results: %s",
                query,
                exc,
            )
            return []
