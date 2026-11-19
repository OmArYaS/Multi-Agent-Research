"""Application use cases.

Coordinates the orchestrator and repository. Depends only on domain
models/interfaces and the orchestrator — never on infrastructure directly.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from domain.interfaces import RunRepository
from domain.models import RunRecord, RunState

logger = logging.getLogger(__name__)


class ResearchUseCases:
    """Application-level use cases for research runs."""

    def __init__(self, orchestrator, repository: RunRepository):
        self.orchestrator = orchestrator
        self.repository = repository

    async def start_research(
        self,
        topic: str,
        model: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> RunRecord:
        """Start a new research run and execute the multi-agent workflow."""
        run_id = str(uuid.uuid4())
        logger.info("Starting research run %s: topic=%r model=%s", run_id, topic, model)
        try:
            await self.orchestrator.run(
                run_id=run_id,
                topic=topic,
                model=model,
                max_iterations=max_iterations,
            )
            record = await self.repository.get(run_id)
            if record is None:
                raise RuntimeError(f"Run {run_id} was not persisted after execution")
            return record
        except Exception:
            logger.exception("Research run %s failed", run_id)
            # Persist a failed record so the client can inspect the error.
            from domain.models import RunRecord
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc).isoformat()
            failed = RunRecord(
                run_id=run_id,
                topic=topic,
                model=model or "",
                max_iterations=max_iterations or 2,
                status="failed",
                created_at=now,
                updated_at=now,
                error="Research workflow failed; see server logs.",
            )
            await self.repository.save(failed)
            return failed

    async def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Inspect a single run (status, steps, review logs, report)."""
        return await self.repository.get(run_id)

    async def list_runs(self, limit: int = 50) -> List[RunRecord]:
        """List past research runs, most recent first."""
        return await self.repository.list(limit=limit)
