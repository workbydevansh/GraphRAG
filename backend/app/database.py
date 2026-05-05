from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.app.schemas import PipelineResult, QueryRunResponse


class BenchmarkDatabase:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init()

    def init(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    ground_truth TEXT,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    winner TEXT,
                    insight TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pipeline_results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    pipeline_name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                """
            )

    def save_run(self, run: QueryRunResponse) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, question, ground_truth, created_at, mode, winner, insight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.question,
                    run.ground_truth,
                    run.created_at.isoformat(),
                    run.mode,
                    run.winner,
                    run.insight,
                ),
            )
            conn.execute("DELETE FROM pipeline_results WHERE run_id = ?", (run.run_id,))
            for result in run.results:
                conn.execute(
                    """
                    INSERT INTO pipeline_results (id, run_id, pipeline_name, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        f"{run.run_id}:{result.pipeline_name}",
                        run.run_id,
                        result.pipeline_name,
                        result.model_dump_json(),
                    ),
                )

    def list_runs(self, limit: int = 200) -> list[QueryRunResponse]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._hydrate(conn, row) for row in rows]

    def _hydrate(self, conn: sqlite3.Connection, row: sqlite3.Row) -> QueryRunResponse:
        result_rows = conn.execute(
            "SELECT payload_json FROM pipeline_results WHERE run_id = ? ORDER BY pipeline_name",
            (row["run_id"],),
        ).fetchall()
        return QueryRunResponse(
            run_id=row["run_id"],
            question=row["question"],
            ground_truth=row["ground_truth"],
            created_at=datetime.fromisoformat(row["created_at"]),
            mode=row["mode"],
            winner=row["winner"],
            insight=row["insight"],
            results=[PipelineResult.model_validate(json.loads(item["payload_json"])) for item in result_rows],
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn


def utcnow() -> datetime:
    return datetime.now(timezone.utc)

