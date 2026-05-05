from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


PipelineName = Literal["llm_only", "basic_rag", "graphrag"]
RunMode = Literal["live", "dev", "mixed"]


class PipelineResult(BaseModel):
    pipeline_name: str
    answer: str
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0
    estimated_cost_usd: float = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRunAllRequest(BaseModel):
    question: str = Field(min_length=2)
    ground_truth: str | None = None
    save: bool = True


class QueryRunOneRequest(QueryRunAllRequest):
    pipeline_name: PipelineName


class QueryRunResponse(BaseModel):
    run_id: str
    question: str
    ground_truth: str | None = None
    created_at: datetime
    mode: RunMode
    winner: str | None = None
    insight: str
    results: list[PipelineResult]


class BenchmarkRequest(BaseModel):
    limit: int | None = None
    eval_set_path: str | None = None
    save: bool = True


class SummaryResponse(BaseModel):
    total_runs: int
    pipeline_averages: dict[str, dict[str, float]]
    token_reduction_vs_basic: float | None = None
    cost_reduction_vs_basic: float | None = None
    accuracy_delta_vs_basic: float | None = None
    graphrag_wins: int = 0
    recent_results: list[QueryRunResponse] = Field(default_factory=list)


class DatasetStatus(BaseModel):
    corpus_exists: bool
    eval_set_exists: bool
    chroma_index_exists: bool
    corpus_path: str
    eval_set_path: str
    chroma_path: str
    corpus_chunks: int
    eval_questions: int
    estimated_corpus_tokens: int
    mode_note: str


class ReportExportRequest(BaseModel):
    format: Literal["json", "csv", "markdown"] = "markdown"
    limit: int = 200


class ReportExportResponse(BaseModel):
    format: str
    path: str
    content: str

