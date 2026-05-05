from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from backend.app.config import REPO_ROOT
from backend.app.services.dataset.manager import DEV_CORPUS
from backend.app.utils.tokens import estimate_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA-style corpus and eval set.")
    parser.add_argument("--target-tokens", type=int, default=2_000_000, help="Minimum corpus size to build.")
    parser.add_argument("--eval-size", type=int, default=40, help="Number of eval questions to write.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-dev-seed", action="store_true", help="Write a tiny labelled dev corpus if download fails.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "data" / "eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "corpus.jsonl"
    eval_path = args.eval_dir / "eval_set.json"

    try:
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "distractor")
        train_rows = dataset["train"]
        validation_rows = dataset["validation"]
        chunks, token_count = build_corpus(train_rows, args.target_tokens)
        eval_rows = build_eval_set(validation_rows, args.eval_size)
        write_jsonl(corpus_path, chunks)
        eval_path.write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")
        print(f"Wrote {len(chunks):,} corpus chunks to {corpus_path}")
        print(f"Estimated corpus tokens: {token_count:,}")
        print(f"Wrote {len(eval_rows):,} eval questions to {eval_path}")
    except Exception as exc:
        if not args.allow_dev_seed:
            raise RuntimeError(
                "HotpotQA download/preparation failed. Re-run with --allow-dev-seed for a tiny labelled dev corpus."
            ) from exc
        write_dev_seed(corpus_path, eval_path, str(exc))
        print(f"HotpotQA preparation failed, so a labelled dev seed was written: {exc}")


def build_corpus(rows: Any, target_tokens: int) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    token_count = 0
    for row_index, row in enumerate(rows):
        context = row.get("context") if isinstance(row, dict) else row["context"]
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        for passage_index, (title, passage_sentences) in enumerate(zip(titles, sentences)):
            text = " ".join(sentence.strip() for sentence in passage_sentences if sentence and sentence.strip())
            key = (title, text[:180])
            if not text or key in seen:
                continue
            seen.add(key)
            tokens = estimate_tokens(text)
            token_count += tokens
            chunks.append(
                {
                    "id": f"hotpot-{row_index}-{passage_index}",
                    "title": title,
                    "text": text,
                    "source": "hotpotqa_distractor_train",
                    "metadata": {
                        "row_index": row_index,
                        "passage_index": passage_index,
                        "estimated_tokens": tokens,
                    },
                }
            )
            if token_count >= target_tokens:
                return chunks, token_count
    return chunks, token_count


def build_eval_set(rows: Any, eval_size: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        item = dict(row)
        supporting_facts = item.get("supporting_facts", {})
        support_titles = supporting_facts.get("title", []) if isinstance(supporting_facts, dict) else []
        if len(set(support_titles)) < 2:
            continue
        candidates.append(
            {
                "id": item.get("id") or item.get("_id") or f"eval-{row_index}",
                "question": item["question"],
                "answer": item["answer"],
                "type": item.get("type", "comparison_or_bridge"),
                "level": item.get("level", "unknown"),
                "supporting_titles": sorted(set(support_titles)),
            }
        )
    random.shuffle(candidates)
    return candidates[:eval_size]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_dev_seed(corpus_path: Path, eval_path: Path, reason: str) -> None:
    with corpus_path.open("w", encoding="utf-8") as handle:
        for chunk in DEV_CORPUS:
            handle.write(
                json.dumps(
                    {
                        "id": chunk["id"],
                        "title": chunk["title"],
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "metadata": {"dev_seed_reason": reason},
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    eval_path.write_text(
        json.dumps(
            [
                {
                    "id": "dev-eval-1",
                    "question": "Which city was Marie Curie born in, and what country is that city the capital of?",
                    "answer": "Warsaw, Poland",
                    "type": "bridge",
                    "level": "dev_seed",
                    "supporting_titles": ["HotpotQA Dev Seed: Marie Curie", "HotpotQA Dev Seed: Warsaw"],
                    "dev_seed_reason": reason,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
