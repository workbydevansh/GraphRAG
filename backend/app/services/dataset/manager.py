from __future__ import annotations

import json
from pathlib import Path

from backend.app.config import Settings
from backend.app.utils.tokens import estimate_tokens


DEV_CORPUS = [
    {
        "id": "dev-1",
        "title": "Marie Curie",
        "text": "Marie Curie was born in Warsaw and later worked in Paris on radioactivity.",
        "source": "dev_seed",
    },
    {
        "id": "dev-2",
        "title": "Warsaw",
        "text": "Warsaw is the capital and largest city of Poland.",
        "source": "dev_seed",
    },
    {
        "id": "dev-3",
        "title": "Nobel Prize in Physics",
        "text": "The 1903 Nobel Prize in Physics recognized work on radiation by Henri Becquerel, Pierre Curie, and Marie Curie.",
        "source": "dev_seed",
    },
]


class DatasetManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load_corpus(self) -> tuple[list[dict], list[str]]:
        if not self.settings.corpus_path.exists():
            return DEV_CORPUS, [
                f"Corpus not found at {self.settings.corpus_path}. Using labelled dev seed."
            ]
        rows: list[dict] = []
        with self.settings.corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows or DEV_CORPUS, [] if rows else ["Corpus is empty. Using labelled dev seed."]

    def load_eval_set(self, override_path: str | None = None) -> list[dict]:
        path = Path(override_path) if override_path else self.settings.eval_set_path
        if not path.exists():
            return [
                {
                    "id": "dev-eval-1",
                    "question": "Which country is the city where Marie Curie was born the capital of?",
                    "answer": "Poland",
                }
            ]
        return json.loads(path.read_text(encoding="utf-8"))

    def status(self) -> dict:
        corpus, _ = self.load_corpus()
        eval_rows = self.load_eval_set()
        chroma_exists = self.settings.chroma_path.exists() and any(self.settings.chroma_path.iterdir())
        return {
            "corpus_exists": self.settings.corpus_path.exists(),
            "eval_set_exists": self.settings.eval_set_path.exists(),
            "chroma_index_exists": chroma_exists,
            "corpus_path": str(self.settings.corpus_path),
            "eval_set_path": str(self.settings.eval_set_path),
            "chroma_path": str(self.settings.chroma_path),
            "corpus_chunks": len(corpus),
            "eval_questions": len(eval_rows),
            "estimated_corpus_tokens": sum(estimate_tokens(item.get("text", "")) for item in corpus),
            "mode_note": "Missing artifacts use labelled dev seed data. Run backend/scripts/prepare_hotpotqa.py for real benchmarks.",
        }

