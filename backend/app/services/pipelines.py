from __future__ import annotations

from backend.app.config import Settings
from backend.app.schemas import PipelineResult
from backend.app.services.base import BasePipeline
from backend.app.services.dataset import DatasetManager
from backend.app.services.evaluation import Evaluator
from backend.app.services.graphrag import TigerGraphGraphRAGClient
from backend.app.services.llm import LLMProvider
from backend.app.services.metrics import MetricsCollector
from backend.app.services.rag import BasicRAGRetriever


class LLMOnlyPipeline(BasePipeline):
    pipeline_name = "llm_only"

    def __init__(self, settings: Settings, llm: LLMProvider, metrics: MetricsCollector, evaluator: Evaluator):
        self.settings = settings
        self.llm = llm
        self.metrics = metrics
        self.evaluator = evaluator

    async def run(self, question: str, ground_truth: str | None = None) -> PipelineResult:
        started = self.metrics.start_timer()
        prompt = f"Question: {question}\nAnswer directly and concisely."
        response = await self.llm.generate(prompt, system="You are a benchmark QA assistant.")
        snapshot = self.metrics.collect(
            started,
            prompt,
            response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        evaluation = await self.evaluator.evaluate(question, response.text, ground_truth, [])
        return PipelineResult(
            pipeline_name=self.pipeline_name,
            answer=response.text,
            contexts=[],
            prompt_tokens=snapshot.prompt_tokens,
            completion_tokens=snapshot.completion_tokens,
            total_tokens=snapshot.total_tokens,
            latency_ms=snapshot.latency_ms,
            estimated_cost_usd=snapshot.estimated_cost_usd,
            metadata={
                "mode": response.mode,
                "model": response.model,
                "warnings": response.warnings,
                "evaluation": evaluation,
            },
        )


class BasicRAGPipeline(BasePipeline):
    pipeline_name = "basic_rag"

    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        retriever: BasicRAGRetriever,
        metrics: MetricsCollector,
        evaluator: Evaluator,
    ):
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.metrics = metrics
        self.evaluator = evaluator

    async def run(self, question: str, ground_truth: str | None = None) -> PipelineResult:
        started = self.metrics.start_timer()
        contexts, retrieval_warnings = self.retriever.retrieve(question)
        context_block = "\n\n".join(
            f"[{index + 1}] {item['title']}: {item['text']}" for index, item in enumerate(contexts)
        )
        prompt = f"Question: {question}\n\nRetrieved passages:\n{context_block}\n\nAnswer using only retrieved evidence."
        response = await self.llm.generate(prompt, system="You are a grounded RAG QA assistant.")
        snapshot = self.metrics.collect(
            started,
            prompt,
            response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        evaluation = await self.evaluator.evaluate(question, response.text, ground_truth, contexts)
        return PipelineResult(
            pipeline_name=self.pipeline_name,
            answer=response.text,
            contexts=contexts,
            prompt_tokens=snapshot.prompt_tokens,
            completion_tokens=snapshot.completion_tokens,
            total_tokens=snapshot.total_tokens,
            latency_ms=snapshot.latency_ms,
            estimated_cost_usd=snapshot.estimated_cost_usd,
            metadata={
                "mode": "mixed" if retrieval_warnings and response.mode == "live" else response.mode,
                "model": response.model,
                "warnings": [*retrieval_warnings, *response.warnings],
                "evaluation": evaluation,
            },
        )


class GraphRAGPipeline(BasePipeline):
    pipeline_name = "graphrag"

    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        client: TigerGraphGraphRAGClient,
        metrics: MetricsCollector,
        evaluator: Evaluator,
    ):
        self.settings = settings
        self.llm = llm
        self.client = client
        self.metrics = metrics
        self.evaluator = evaluator

    async def run(self, question: str, ground_truth: str | None = None) -> PipelineResult:
        started = self.metrics.start_timer()
        direct_answer, contexts, retrieval_warnings, retrieval_mode = await self.client.retrieve_and_answer(question)
        context_block = "\n\n".join(
            f"[{index + 1}] {item['title']}: {item['text']}" for index, item in enumerate(contexts)
        )
        if direct_answer:
            answer = direct_answer
            prompt = f"Question: {question}\nTigerGraph GraphRAG returned a direct answer."
            prompt_tokens = None
            completion_tokens = None
            model = "tigergraph-graphrag"
            llm_warnings: list[str] = []
        else:
            prompt = f"Question: {question}\n\nCompact graph evidence:\n{context_block}\n\nAnswer using only graph evidence."
            response = await self.llm.generate(prompt, system="You are a graph-grounded QA assistant.")
            answer = response.text
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens
            model = response.model
            llm_warnings = response.warnings
        snapshot = self.metrics.collect(started, prompt, answer, prompt_tokens, completion_tokens)
        evaluation = await self.evaluator.evaluate(question, answer, ground_truth, contexts)
        return PipelineResult(
            pipeline_name=self.pipeline_name,
            answer=answer,
            contexts=contexts,
            prompt_tokens=snapshot.prompt_tokens,
            completion_tokens=snapshot.completion_tokens,
            total_tokens=snapshot.total_tokens,
            latency_ms=snapshot.latency_ms,
            estimated_cost_usd=snapshot.estimated_cost_usd,
            metadata={
                "mode": retrieval_mode if direct_answer else "dev",
                "model": model,
                "warnings": [*retrieval_warnings, *llm_warnings],
                "evaluation": evaluation,
                "trace": [
                    "Retrieve compact graph-connected evidence.",
                    "Generate answer from compact context.",
                    "Collect token, cost, latency, and quality metrics.",
                ],
            },
        )

