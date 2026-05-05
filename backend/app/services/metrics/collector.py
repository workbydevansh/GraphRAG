from __future__ import annotations

import time
from dataclasses import dataclass

from backend.app.config import Settings
from backend.app.utils.tokens import estimate_cost, estimate_tokens


@dataclass
class MetricsSnapshot:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float


class MetricsCollector:
    def __init__(self, settings: Settings):
        self.settings = settings

    def start_timer(self) -> float:
        return time.perf_counter()

    def collect(
        self,
        started_at: float,
        prompt: str,
        answer: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> MetricsSnapshot:
        measured_prompt_tokens = prompt_tokens if prompt_tokens is not None else estimate_tokens(prompt)
        measured_completion_tokens = (
            completion_tokens if completion_tokens is not None else estimate_tokens(answer)
        )
        return MetricsSnapshot(
            prompt_tokens=measured_prompt_tokens,
            completion_tokens=measured_completion_tokens,
            total_tokens=measured_prompt_tokens + measured_completion_tokens,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
            estimated_cost_usd=estimate_cost(
                measured_prompt_tokens,
                measured_completion_tokens,
                self.settings.input_cost_per_1k,
                self.settings.output_cost_per_1k,
            ),
        )

