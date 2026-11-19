"""SQLite run repository implementing the domain ``RunRepository`` interface."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import List, Optional

from domain.interfaces import RunRepository
from domain.models import AgentStep, RunRecord

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    topic         TEXT NOT NULL,
    model         TEXT NOT NULL,
    max_iterations INTEGER NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    report        TEXT,
    review        TEXT,
    steps         TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_updated_at ON runs(updated_at);
"""


class SQLiteRunRepository:
    """Persists research runs in a local SQLite file."""

    def __init__(self, db_path: str = "research_runs.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced in later versions to pre-existing DB files."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(runs)")}
        if "review" not in columns:
            self._conn.execute("ALTER TABLE runs ADD COLUMN review TEXT")

    # -- serialization helpers ------------------------------------------

    @staticmethod
    def _record_to_row(run: RunRecord) -> tuple:
        return (
            run.run_id,
            run.topic,
            run.model,
            run.max_iterations,
            run.status,
            run.created_at,
            run.updated_at,
            run.report.model_dump_json() if run.report else None,
            run.review.model_dump_json() if run.review else None,
            json.dumps([step.model_dump() for step in run.steps]),
            run.error,
        )

    @classmethod
    def _row_to_record(cls, row: sqlite3.Row) -> RunRecord:
        report = None
        if row["report"]:
            from domain.models import FinalReport

            report = FinalReport.model_validate_json(row["report"])
        review = None
        if row["review"]:
            from domain.models import ReviewDecision

            review = ReviewDecision.model_validate_json(row["review"])
        steps = [AgentStep.model_validate(s) for s in json.loads(row["steps"] or "[]")]
        return RunRecord(
            run_id=row["run_id"],
            topic=row["topic"],
            model=row["model"],
            max_iterations=row["max_iterations"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            report=report,
            review=review,
            steps=steps,
            error=row["error"],
        )

    def close(self) -> None:
        self._conn.close()

    # -- RunRepository interface ----------------------------------------

    async def save(self, run: RunRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, topic, model, max_iterations, status,
                              created_at, updated_at, report, review, steps, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                topic=excluded.topic,
                model=excluded.model,
                max_iterations=excluded.max_iterations,
                status=excluded.status,
                updated_at=excluded.updated_at,
                report=excluded.report,
                review=excluded.review,
                steps=excluded.steps,
                error=excluded.error
            """,
            self._record_to_row(run),
        )
        self._conn.commit()

    async def get(self, run_id: str) -> Optional[RunRecord]:
        cursor = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    async def list(self, limit: int = 50) -> List[RunRecord]:
        cursor = self._conn.execute(
            "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    async def append_step(self, run_id: str, step: AgentStep) -> None:
        self._conn.execute(
            "UPDATE runs SET steps = ?, updated_at = ? WHERE run_id = ?",
            (
                json.dumps([s.model_dump() for s in self._steps_for(run_id)] + [step.model_dump()]),
                step.timestamp,
                run_id,
            ),
        )
        self._conn.commit()

    async def update_state(self, run_id: str, **fields) -> None:
        if not fields:
            return
        if "steps" in fields:
            steps = fields.pop("steps")
            for s in steps:
                await self.append_step(run_id, AgentStep.model_validate(s))
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [run_id]
        self._conn.execute(
            f"UPDATE runs SET {set_clause}, updated_at = ? WHERE run_id = ?",
            values + [self._now()],
        )
        self._conn.commit()

    # -- internal -------------------------------------------------------

    def _steps_for(self, run_id: str) -> List[dict]:
        cursor = self._conn.execute("SELECT steps FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row or not row["steps"]:
            return []
        return json.loads(row["steps"])

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
