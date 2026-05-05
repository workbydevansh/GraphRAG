from __future__ import annotations

import httpx

from backend.app.config import Settings
from backend.app.services.dataset import DatasetManager
from backend.app.services.rag.retriever import BasicRAGRetriever


class TigerGraphGraphRAGClient:
    def __init__(self, settings: Settings, dataset_manager: DatasetManager):
        self.settings = settings
        self.dataset_manager = dataset_manager

    async def retrieve_and_answer(self, question: str) -> tuple[str | None, list[dict], list[str], str]:
        if self.settings.tigergraph_graphrag_url and self._has_py_tigergraph_credentials():
            try:
                return await self._call_pytigergraph(question)
            except Exception as exc:
                answer, contexts, warnings, mode = await self._dev_fallback(question)
                return answer, contexts, [*warnings, f"pyTigerGraph GraphRAG failed: {exc}"], mode
        if self.settings.tigergraph_graphrag_url:
            try:
                return await self._call_http(question)
            except Exception as exc:
                answer, contexts, warnings, mode = await self._dev_fallback(question)
                return answer, contexts, [*warnings, f"TigerGraph GraphRAG HTTP call failed: {exc}"], mode
        return await self._dev_fallback(question)

    def _has_py_tigergraph_credentials(self) -> bool:
        return bool(
            self.settings.tigergraph_host
            and self.settings.tigergraph_username
            and (self.settings.tigergraph_password or self.settings.tigergraph_secret)
        )

    async def _call_pytigergraph(self, question: str) -> tuple[str, list[dict], list[str], str]:
        import asyncio

        def call():
            from pyTigerGraph import TigerGraphConnection

            conn = TigerGraphConnection(
                host=self.settings.tigergraph_host,
                username=self.settings.tigergraph_username,
                password=self.settings.tigergraph_password,
                restppPort=self.settings.tigergraph_restpp_port,
            )
            if self.settings.tigergraph_secret:
                conn.getToken(self.settings.tigergraph_secret)
            conn.graphname = self.settings.tigergraph_graph_name
            conn.ai.configureGraphRAGHost(self.settings.tigergraph_graphrag_url)
            return conn.ai.answerQuestion(
                question,
                method=self.settings.tigergraph_method,
                method_parameters={
                    "indices": ["Document", "DocumentChunk", "Entity", "Relationship"],
                    "top_k": self.settings.graphrag_top_k,
                    "num_hops": 2,
                    "num_seen_min": 2,
                    "verbose": True,
                },
            )

        payload = await asyncio.to_thread(call)
        return self._parse_payload(payload)

    async def _call_http(self, question: str) -> tuple[str, list[dict], list[str], str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.tigergraph_graphrag_api_key:
            headers["Authorization"] = f"Bearer {self.settings.tigergraph_graphrag_api_key}"
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.tigergraph_graphrag_url.rstrip('/')}/{self.settings.tigergraph_graphrag_path.lstrip('/')}",
                headers=headers,
                json={"question": question, "method": self.settings.tigergraph_method},
            )
            response.raise_for_status()
            return self._parse_payload(response.json())

    async def _dev_fallback(self, question: str) -> tuple[str | None, list[dict], list[str], str]:
        retriever = BasicRAGRetriever(self.settings, self.dataset_manager)
        contexts, warnings = retriever.retrieve(question, self.settings.graphrag_top_k)
        compact_contexts = [
            {**context, "text": context.get("text", "")[: self.settings.max_graph_context_chars]}
            for context in contexts
        ]
        return (
            None,
            compact_contexts,
            [
                *warnings,
                "TigerGraph GraphRAG credentials/endpoint missing; labelled compact dev fallback used.",
            ],
            "dev",
        )

    def _parse_payload(self, payload: dict) -> tuple[str, list[dict], list[str], str]:
        answer = payload.get("response") or payload.get("answer") or payload.get("message") or str(payload)
        raw_contexts = payload.get("contexts") or payload.get("chunks") or payload.get("evidence") or []
        contexts = []
        for index, item in enumerate(raw_contexts if isinstance(raw_contexts, list) else []):
            if isinstance(item, str):
                contexts.append({"id": f"tg-{index}", "title": f"TigerGraph context {index + 1}", "text": item, "source": "tigergraph"})
            elif isinstance(item, dict):
                contexts.append(
                    {
                        "id": str(item.get("id") or item.get("chunk_id") or f"tg-{index}"),
                        "title": item.get("title") or item.get("label") or f"TigerGraph context {index + 1}",
                        "text": item.get("text") or item.get("content") or str(item),
                        "source": "tigergraph",
                        "score": item.get("score"),
                    }
                )
        return answer, contexts, [], "live"

