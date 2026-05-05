from dataclasses import dataclass
from datetime import datetime

from backend.app.schemas import PipelineResult


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    question: str
    ground_truth: str | None
    created_at: datetime
    mode: str
    winner: str | None
    insight: str
    results: list[PipelineResult]

