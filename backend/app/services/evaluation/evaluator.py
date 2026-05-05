from __future__ import annotations

import re

from backend.app.config import Settings


TOKEN_RE = re.compile(r"[a-z0-9]+")


class Evaluator:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def evaluate(self, question: str, answer: str, ground_truth: str | None, contexts: list[dict]) -> dict:
        exact_match = self.exact_match(answer, ground_truth)
        answer_recall = self.answer_recall(answer, ground_truth)
        bertscore_f1, bertscore_note = await self.bertscore(answer, ground_truth)
        judge = self.heuristic_judge(answer, ground_truth, contexts)
        return {
            "exact_match": exact_match,
            "answer_recall": answer_recall,
            "bertscore_f1": bertscore_f1,
            "bertscore_note": bertscore_note,
            "judge": judge,
        }

    def exact_match(self, answer: str, ground_truth: str | None) -> float | None:
        if not ground_truth:
            return None
        return 1.0 if " ".join(self._tokens(ground_truth)) in " ".join(self._tokens(answer)) else 0.0

    def answer_recall(self, answer: str, ground_truth: str | None) -> float | None:
        if not ground_truth:
            return None
        truth = set(self._tokens(ground_truth))
        if not truth:
            return None
        return round(len(truth & set(self._tokens(answer))) / len(truth), 4)

    async def bertscore(self, answer: str, ground_truth: str | None) -> tuple[float | None, str | None]:
        if not ground_truth or not self.settings.enable_hf_eval:
            return None, "disabled_or_missing_ground_truth"
        try:
            import evaluate

            scorer = evaluate.load("bertscore")
            result = scorer.compute(
                predictions=[answer],
                references=[ground_truth],
                model_type=self.settings.bertscore_model,
                lang="en",
            )
            return round(float(result["f1"][0]), 4), None
        except Exception as exc:
            return None, f"bertscore_not_run: {exc}"

    def heuristic_judge(self, answer: str, ground_truth: str | None, contexts: list[dict]) -> dict:
        answer_terms = set(self._tokens(answer))
        context_terms = set(self._tokens(" ".join(str(item.get("text", "")) for item in contexts)))
        grounding = len(answer_terms & context_terms) / max(1, len(answer_terms)) if answer_terms else 0.0
        recall = self.answer_recall(answer, ground_truth)
        score = grounding if recall is None else (0.65 * recall + 0.35 * grounding)
        return {
            "label": "dev_heuristic",
            "score": round(float(score), 4),
            "reasoning": "Heuristic judge used unless a live LLM judge is configured.",
        }

    def _tokens(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

