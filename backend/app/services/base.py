from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.schemas import PipelineResult


class BasePipeline(ABC):
    pipeline_name: str

    @abstractmethod
    async def run(self, question: str, ground_truth: str | None = None) -> PipelineResult:
        """Run the pipeline and return the benchmark contract."""

