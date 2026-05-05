from __future__ import annotations

import csv
import io
import json
import statistics
from uuid import uuid4

from backend.app.config import Settings
from backend.app.database import BenchmarkDatabase, utcnow
from backend.app.schemas import PipelineResult, QueryRunResponse, SummaryResponse
from backend.app.services.dataset import DatasetManager
from backend.app.services.evaluation import Evaluator
from backend.app.services.graphrag import TigerGraphGraphRAGClient
from backend.app.services.llm import create_llm_provider
from backend.app.services.metrics import MetricsCollector
from backend.app.services.pipelines import BasicRAGPipeline, GraphRAGPipeline, LLMOnlyPipeline
from backend.app.services.rag import BasicRAGRetriever


class BenchmarkService:
    def __init__(self, settings: Settings, database: BenchmarkDatabase):
        self.settings = settings
        self.database = database
        self.dataset_manager = DatasetManager(settings)
        self.llm = create_llm_provider(settings)
        self.metrics = MetricsCollector(settings)
        self.evaluator = Evaluator(settings)
        self.retriever = BasicRAGRetriever(settings, self.dataset_manager)
        self.graphrag_client = TigerGraphGraphRAGClient(settings, self.dataset_manager)

    def pipelines(self):
        return {
            "llm_only": LLMOnlyPipeline(self.settings, self.llm, self.metrics, self.evaluator),
            "basic_rag": BasicRAGPipeline(
                self.settings,
                self.llm,
                self.retriever,
                self.metrics,
                self.evaluator,
            ),
            "graphrag": GraphRAGPipeline(
                self.settings,
                self.llm,
                self.graphrag_client,
                self.metrics,
                self.evaluator,
            ),
        }

    async def run_all(self, question: str, ground_truth: str | None, save: bool = True) -> QueryRunResponse:
        results = []
        for pipeline in self.pipelines().values():
            results.append(await pipeline.run(question, ground_truth))
        response = self._build_response(question, ground_truth, results)
        if save:
            self.database.save_run(response)
        return response

    async def run_one(
        self,
        pipeline_name: str,
        question: str,
        ground_truth: str | None,
        save: bool = False,
    ) -> QueryRunResponse:
        pipeline = self.pipelines()[pipeline_name]
        result = await pipeline.run(question, ground_truth)
        response = self._build_response(question, ground_truth, [result])
        if save:
            self.database.save_run(response)
        return response

    async def run_batch(self, limit: int | None = None, eval_set_path: str | None = None, save: bool = True) -> list[QueryRunResponse]:
        questions = self.dataset_manager.load_eval_set(eval_set_path)
        selected = questions[: limit or self.settings.default_batch_limit]
        runs = []
        for item in selected:
            runs.append(
                await self.run_all(
                    item["question"],
                    item.get("answer") or item.get("ground_truth"),
                    save=save,
                )
            )
        return runs

    def summary(self, limit: int = 200) -> SummaryResponse:
        return build_summary(self.database.list_runs(limit))

    def export(self, export_format: str, limit: int = 200) -> tuple[str, str]:
        runs = self.database.list_runs(limit)
        if export_format == "json":
            return "json", json.dumps([run.model_dump(mode="json") for run in runs], indent=2)
        if export_format == "csv":
            return "csv", runs_to_csv(runs)
        return "markdown", runs_to_markdown(runs, self.summary(limit).model_dump(mode="json"))

    def _build_response(
        self,
        question: str,
        ground_truth: str | None,
        results: list[PipelineResult],
    ) -> QueryRunResponse:
        modes = {str(result.metadata.get("mode", "dev")) for result in results}
        mode = "live" if modes == {"live"} else "dev" if modes == {"dev"} else "mixed"
        winner = pick_winner(results)
        insight = build_insight(results, winner)
        return QueryRunResponse(
            run_id=str(uuid4()),
            question=question,
            ground_truth=ground_truth,
            created_at=utcnow(),
            mode=mode,
            winner=winner,
            insight=insight,
            results=results,
        )


def quality_score(result: PipelineResult) -> float:
    evaluation = result.metadata.get("evaluation", {})
    for key in ("bertscore_f1", "answer_recall", "exact_match"):
        value = evaluation.get(key)
        if value is not None:
            return float(value)
    judge = evaluation.get("judge") or {}
    return float(judge.get("score") or 0)


def pick_winner(results: list[PipelineResult]) -> str | None:
    if not results:
        return None
    ranked = sorted(
        results,
        key=lambda result: (quality_score(result), -result.total_tokens, -result.estimated_cost_usd),
        reverse=True,
    )
    return ranked[0].pipeline_name


def build_insight(results: list[PipelineResult], winner: str | None) -> str:
    by_name = {result.pipeline_name: result for result in results}
    basic = by_name.get("basic_rag")
    graph = by_name.get("graphrag")
    if not basic or not graph:
        return f"{winner or 'The selected pipeline'} completed with measured metrics."
    token_reduction = reduction(basic.total_tokens, graph.total_tokens)
    cost_reduction = reduction(basic.estimated_cost_usd, graph.estimated_cost_usd)
    accuracy_delta = quality_score(graph) - quality_score(basic)
    return (
        f"Winner: {winner}. GraphRAG used {token_reduction:.1f}% fewer tokens and "
        f"{cost_reduction:.1f}% lower estimated cost than Basic RAG, with "
        f"{accuracy_delta * 100:+.1f} quality points."
    )


def build_summary(runs: list[QueryRunResponse]) -> SummaryResponse:
    per_pipeline: dict[str, list[PipelineResult]] = {}
    for run in runs:
        for result in run.results:
            per_pipeline.setdefault(result.pipeline_name, []).append(result)
    averages = {
        pipeline: {
            "prompt_tokens": avg([item.prompt_tokens for item in results]),
            "completion_tokens": avg([item.completion_tokens for item in results]),
            "total_tokens": avg([item.total_tokens for item in results]),
            "latency_ms": avg([item.latency_ms for item in results]),
            "estimated_cost_usd": avg([item.estimated_cost_usd for item in results]),
            "quality_score": avg([quality_score(item) for item in results]),
        }
        for pipeline, results in per_pipeline.items()
    }
    basic = averages.get("basic_rag")
    graph = averages.get("graphrag")
    return SummaryResponse(
        total_runs=len(runs),
        pipeline_averages=averages,
        token_reduction_vs_basic=reduction(basic["total_tokens"], graph["total_tokens"]) if basic and graph else None,
        cost_reduction_vs_basic=reduction(basic["estimated_cost_usd"], graph["estimated_cost_usd"]) if basic and graph else None,
        accuracy_delta_vs_basic=graph["quality_score"] - basic["quality_score"] if basic and graph else None,
        graphrag_wins=sum(1 for run in runs if run.winner == "graphrag"),
        recent_results=runs[:20],
    )


def reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return round(((baseline - candidate) / baseline) * 100, 4)


def avg(values: list[float]) -> float:
    return round(float(statistics.mean(values)), 6) if values else 0.0


def runs_to_csv(runs: list[QueryRunResponse]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "run_id",
            "created_at",
            "question",
            "ground_truth",
            "winner",
            "pipeline_name",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_ms",
            "estimated_cost_usd",
            "mode",
            "quality_score",
        ]
    )
    for run in runs:
        for result in run.results:
            writer.writerow(
                [
                    run.run_id,
                    run.created_at.isoformat(),
                    run.question,
                    run.ground_truth or "",
                    run.winner or "",
                    result.pipeline_name,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.total_tokens,
                    result.latency_ms,
                    result.estimated_cost_usd,
                    result.metadata.get("mode", "dev"),
                    quality_score(result),
                ]
            )
    return output.getvalue()


def runs_to_markdown(runs: list[QueryRunResponse], summary: dict) -> str:
    lines = [
        "# GraphRAG Benchmark Lab Report",
        "",
        "Rows labelled dev or mixed used explicit fallbacks where live services were unavailable.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Runs",
        "",
        "| Question | Winner | GraphRAG token reduction | GraphRAG cost reduction |",
        "| --- | --- | ---: | ---: |",
    ]
    for run in runs:
        by_name = {result.pipeline_name: result for result in run.results}
        basic = by_name.get("basic_rag")
        graph = by_name.get("graphrag")
        token_reduction = reduction(basic.total_tokens, graph.total_tokens) if basic and graph else 0
        cost_reduction = reduction(basic.estimated_cost_usd, graph.estimated_cost_usd) if basic and graph else 0
        lines.append(
            f"| {run.question.replace('|', '/')} | {run.winner or 'n/a'} | {token_reduction:.1f}% | {cost_reduction:.1f}% |"
        )
    return "\n".join(lines) + "\n"

