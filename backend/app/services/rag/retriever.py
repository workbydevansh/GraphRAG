from __future__ import annotations

import re

from backend.app.config import Settings
from backend.app.services.dataset import DatasetManager


TOKEN_RE = re.compile(r"[a-z0-9]+")


class BasicRAGRetriever:
    def __init__(self, settings: Settings, dataset_manager: DatasetManager):
        self.settings = settings
        self.dataset_manager = dataset_manager

    def retrieve(self, question: str, top_k: int | None = None) -> tuple[list[dict], list[str]]:
        top_k = top_k or self.settings.basic_rag_top_k
        chroma_results, warnings = self._retrieve_chroma(question, top_k)
        if chroma_results:
            return chroma_results, warnings
        corpus, corpus_warnings = self.dataset_manager.load_corpus()
        return self._lexical(question, corpus, top_k), [
            *warnings,
            *corpus_warnings,
            "Basic RAG used lexical fallback because ChromaDB is unavailable or empty.",
        ]

    def _retrieve_chroma(self, question: str, top_k: int) -> tuple[list[dict], list[str]]:
        if not self.settings.chroma_path.exists() or not any(self.settings.chroma_path.iterdir()):
            return [], [f"ChromaDB index not found at {self.settings.chroma_path}."]
        try:
            import chromadb

            collection = chromadb.PersistentClient(path=str(self.settings.chroma_path)).get_collection(
                "hotpotqa_chunks"
            )
            payload = collection.query(query_texts=[question], n_results=top_k)
            docs = payload.get("documents", [[]])[0]
            ids = payload.get("ids", [[]])[0]
            metadatas = payload.get("metadatas", [[]])[0]
            return [
                {
                    "id": str(doc_id),
                    "title": (metadata or {}).get("title", "Untitled"),
                    "text": doc[: self.settings.max_context_chars],
                    "source": (metadata or {}).get("source", "chroma"),
                    "score": None,
                }
                for doc_id, doc, metadata in zip(ids, docs, metadatas)
            ], []
        except Exception as exc:
            return [], [f"ChromaDB retrieval failed: {exc}"]

    def _lexical(self, question: str, corpus: list[dict], top_k: int) -> list[dict]:
        query_terms = set(TOKEN_RE.findall(question.lower()))
        scored = []
        for item in corpus:
            text = f"{item.get('title', '')} {item.get('text', '')}"
            terms = TOKEN_RE.findall(text.lower())
            score = sum(1 for term in terms if term in query_terms)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "id": str(item.get("id", index)),
                "title": item.get("title", "Untitled"),
                "text": item.get("text", "")[: self.settings.max_context_chars],
                "source": item.get("source", "local"),
                "score": score,
            }
            for index, (score, item) in enumerate(scored[:top_k])
        ]

