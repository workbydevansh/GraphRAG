from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = REPO_ROOT / "data" / "processed" / "documents.jsonl"
EVAL_PATH = REPO_ROOT / "data" / "eval" / "eval_questions.json"

CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&.'-]+(?:\s+|$)){1,5}")


@dataclass
class DocumentDraft:
    doc_id: str
    title: str
    text_parts: list[str] = field(default_factory=list)
    question_ids: set[str] = field(default_factory=set)
    supporting_fact_sentences: list[str] = field(default_factory=list)

    def add_text(self, text: str) -> None:
        clean = normalize_space(text)
        if clean and clean not in self.text_parts:
            self.text_parts.append(clean)

    @property
    def text(self) -> str:
        return "\n\n".join(self.text_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare HotpotQA documents and evaluation questions for GraphRAG Benchmark Lab."
    )
    parser.add_argument("--min-tokens", type=int, default=2_000_000)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--documents-path", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        train_rows, validation_rows = load_hotpotqa()
        mode = "hotpotqa"
        failure_reason = None
    except Exception as exc:
        train_rows, validation_rows = fallback_rows()
        mode = "fallback_dev_sample"
        failure_reason = str(exc)

    documents, token_count = build_corpus(train_rows, min_tokens=args.min_tokens)
    eval_questions = build_eval_set(validation_rows, eval_size=args.eval_size)

    write_jsonl(args.documents_path, documents)
    write_json(args.eval_path, eval_questions)

    print_summary(
        documents=documents,
        token_count=token_count,
        eval_questions=eval_questions,
        mode=mode,
        failure_reason=failure_reason,
        documents_path=args.documents_path,
        eval_path=args.eval_path,
    )


def load_hotpotqa() -> tuple[Iterable[dict[str, Any]], Iterable[dict[str, Any]]]:
    """Load HotpotQA from Hugging Face.

    The `distractor` config includes the multi-hop context passages needed for
    both Basic RAG and GraphRAG benchmark preparation.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to download HotpotQA from Hugging Face.") from exc

    dataset = load_dataset("hotpot_qa", "distractor")
    return dataset["train"], dataset["validation"]


def build_corpus(rows: Iterable[dict[str, Any]], min_tokens: int) -> tuple[list[dict[str, Any]], int]:
    """Aggregate HotpotQA context passages into document records."""

    drafts: dict[str, DocumentDraft] = {}
    token_count = 0

    for row_index, row in enumerate(rows):
        question_id = get_question_id(row, row_index)
        context_items = extract_context_items(row)
        support_map = supporting_fact_map(row, context_items)

        for passage_index, item in enumerate(context_items):
            title = item["title"]
            sentences = item["sentences"]
            text = normalize_space(" ".join(sentences))
            if not text:
                continue

            doc_id = make_doc_id(title)
            draft = drafts.setdefault(doc_id, DocumentDraft(doc_id=doc_id, title=title))
            before_tokens = estimate_tokens(draft.text)
            draft.add_text(text)
            draft.question_ids.add(question_id)
            draft.supporting_fact_sentences.extend(support_map.get(title, []))
            after_tokens = estimate_tokens(draft.text)
            token_count += max(0, after_tokens - before_tokens)

            if token_count >= min_tokens:
                return serialize_documents(drafts), token_count

    return serialize_documents(drafts), token_count


def build_eval_set(rows: Iterable[dict[str, Any]], eval_size: int) -> list[dict[str, Any]]:
    """Create a diverse 50-question evaluation set when enough rows are available."""

    buckets: dict[str, list[dict[str, Any]]] = {
        "bridge": [],
        "comparison": [],
        "factual": [],
        "synthesis": [],
    }

    for row_index, row in enumerate(rows):
        item = build_eval_item(row, row_index)
        item_type = item["type"]
        difficulty = item["difficulty"].lower()
        support_count = len(set(item["supporting_titles"]))

        if item_type == "comparison":
            buckets["comparison"].append(item)
        elif difficulty == "hard" and support_count >= 2:
            buckets["synthesis"].append({**item, "type": "synthesis"})
        elif item_type == "bridge" and support_count >= 2:
            buckets["bridge"].append(item)
        else:
            buckets["factual"].append({**item, "type": "factual"})

    for values in buckets.values():
        random.shuffle(values)

    target_counts = scaled_eval_targets(eval_size)
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for bucket_name in ["bridge", "comparison", "factual", "synthesis"]:
        selected.extend(take_unique(buckets[bucket_name], target_counts[bucket_name], used_ids))

    if len(selected) < eval_size:
        all_candidates = [item for values in buckets.values() for item in values]
        selected.extend(take_unique(all_candidates, eval_size - len(selected), used_ids))

    return selected[:eval_size]


def build_eval_item(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    context_items = extract_context_items(row)
    support_map = supporting_fact_map(row, context_items)
    supporting_titles = list(support_map.keys())
    supporting_facts = [
        {"title": title, "sentence": sentence}
        for title, sentences in support_map.items()
        for sentence in sentences
    ]

    return {
        "id": get_question_id(row, row_index),
        "question": row["question"],
        "correct_answer": row["answer"],
        "supporting_titles": supporting_titles,
        "supporting_facts": supporting_facts,
        "difficulty": row.get("level", "unknown"),
        "type": row.get("type", "bridge"),
    }


def extract_context_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    context = row.get("context", {})

    if isinstance(context, dict):
        titles = context.get("title", [])
        sentence_groups = context.get("sentences", [])
        return [
            {"title": str(title), "sentences": [str(sentence) for sentence in sentences]}
            for title, sentences in zip(titles, sentence_groups)
        ]

    # Some HotpotQA mirrors expose context as [[title, [sentences...]], ...].
    items = []
    for entry in context:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            title, sentences = entry
            items.append({"title": str(title), "sentences": [str(sentence) for sentence in sentences]})
    return items


def supporting_fact_map(row: dict[str, Any], context_items: list[dict[str, Any]]) -> dict[str, list[str]]:
    supporting_facts = row.get("supporting_facts", {})
    context_by_title = {item["title"]: item["sentences"] for item in context_items}
    mapped: dict[str, list[str]] = defaultdict(list)

    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sentence_ids = supporting_facts.get("sent_id", [])
        pairs = zip(titles, sentence_ids)
    else:
        pairs = supporting_facts

    for title, sentence_id in pairs:
        title = str(title)
        sentences = context_by_title.get(title, [])
        sentence = ""
        if isinstance(sentence_id, int) and 0 <= sentence_id < len(sentences):
            sentence = sentences[sentence_id]
        mapped[title].append(normalize_space(sentence) or f"sentence_id:{sentence_id}")

    return dict(mapped)


def serialize_documents(drafts: dict[str, DocumentDraft]) -> list[dict[str, Any]]:
    documents = []
    for draft in drafts.values():
        supporting_sentences = unique_preserve_order(draft.supporting_fact_sentences)
        documents.append(
            {
                "doc_id": draft.doc_id,
                "title": draft.title,
                "text": draft.text,
                "source": "hotpotqa",
                "entities_hint": extract_entities_hint(draft.title, draft.text),
                "metadata": {
                    "question_ids": sorted(draft.question_ids),
                    "supporting_fact_sentences": supporting_sentences,
                },
            }
        )
    documents.sort(key=lambda item: item["doc_id"])
    return documents


def scaled_eval_targets(eval_size: int) -> dict[str, int]:
    if eval_size == 50:
        return {"bridge": 20, "comparison": 15, "factual": 10, "synthesis": 5}

    base = {"bridge": 20, "comparison": 15, "factual": 10, "synthesis": 5}
    scaled = {key: int(eval_size * value / 50) for key, value in base.items()}
    while sum(scaled.values()) < eval_size:
        for key in ["bridge", "comparison", "factual", "synthesis"]:
            scaled[key] += 1
            if sum(scaled.values()) == eval_size:
                break
    return scaled


def take_unique(items: list[dict[str, Any]], count: int, used_ids: set[str]) -> list[dict[str, Any]]:
    selected = []
    for item in items:
        if len(selected) >= count:
            break
        if item["id"] in used_ids:
            continue
        used_ids.add(item["id"])
        selected.append(item)
    return selected


def estimate_tokens(text: str) -> int:
    """Estimate tokens with tiktoken when installed, otherwise use a stable fallback."""

    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        # Simple fallback: split words/punctuation and approximate long words.
        rough_tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return max(1, int(len(rough_tokens) * 1.15))


def extract_entities_hint(title: str, text: str, limit: int = 12) -> list[str]:
    candidates = [title]
    candidates.extend(match.group(0).strip() for match in CAPITALIZED_PHRASE_RE.finditer(text[:2_500]))
    cleaned = []
    for candidate in candidates:
        candidate = normalize_space(candidate.strip(" .,\n\t"))
        if len(candidate) > 1 and candidate.lower() not in {"the", "a", "an"}:
            cleaned.append(candidate)
    return unique_preserve_order(cleaned)[:limit]


def fallback_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Small HotpotQA-shaped sample used only when Hugging Face download fails."""

    rows = [
        {
            "id": "dev-bridge-1",
            "question": "Which country is the city where Marie Curie was born the capital of?",
            "answer": "Poland",
            "type": "bridge",
            "level": "medium",
            "supporting_facts": {"title": ["Marie Curie", "Warsaw"], "sent_id": [0, 0]},
            "context": {
                "title": ["Marie Curie", "Warsaw", "Nobel Prize in Physics"],
                "sentences": [
                    ["Marie Curie was born in Warsaw and later worked in Paris on radioactivity."],
                    ["Warsaw is the capital and largest city of Poland."],
                    ["The Nobel Prize in Physics is awarded by the Royal Swedish Academy of Sciences."],
                ],
            },
        },
        {
            "id": "dev-comparison-1",
            "question": "Are Marie Curie and Pierre Curie both associated with the 1903 Nobel Prize in Physics?",
            "answer": "yes",
            "type": "comparison",
            "level": "easy",
            "supporting_facts": {"title": ["Marie Curie", "Pierre Curie"], "sent_id": [0, 0]},
            "context": {
                "title": ["Marie Curie", "Pierre Curie"],
                "sentences": [
                    ["Marie Curie shared the 1903 Nobel Prize in Physics."],
                    ["Pierre Curie shared the 1903 Nobel Prize in Physics with Marie Curie and Henri Becquerel."],
                ],
            },
        },
        {
            "id": "dev-synthesis-1",
            "question": "What research area links Marie Curie to the Nobel Prize recognized in 1903?",
            "answer": "radioactivity and radiation phenomena",
            "type": "bridge",
            "level": "hard",
            "supporting_facts": {"title": ["Marie Curie", "Nobel Prize in Physics"], "sent_id": [0, 0]},
            "context": {
                "title": ["Marie Curie", "Nobel Prize in Physics"],
                "sentences": [
                    ["Marie Curie conducted pioneering research on radioactivity."],
                    ["The 1903 Nobel Prize in Physics recognized work on radiation phenomena."],
                ],
            },
        },
    ]
    return rows, rows


def get_question_id(row: dict[str, Any], row_index: int) -> str:
    return str(row.get("id") or row.get("_id") or f"hotpotqa-{row_index}")


def make_doc_id(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"hotpotqa-{slug or 'untitled'}"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(
    documents: list[dict[str, Any]],
    token_count: int,
    eval_questions: list[dict[str, Any]],
    mode: str,
    failure_reason: str | None,
    documents_path: Path,
    eval_path: Path,
) -> None:
    lengths = [len(item["text"]) for item in documents]
    entity_examples = unique_preserve_order(
        entity
        for document in documents[:25]
        for entity in document.get("entities_hint", [])
    )[:10]

    print("\nHotpotQA preparation summary")
    print("----------------------------")
    print(f"mode: {mode}")
    if failure_reason:
        print(f"fallback_reason: {failure_reason}")
    print(f"documents_path: {documents_path}")
    print(f"eval_path: {eval_path}")
    print(f"number_of_documents: {len(documents):,}")
    print(f"estimated_token_count: {token_count:,}")
    print(f"number_of_eval_questions: {len(eval_questions):,}")
    print(f"average_document_length_chars: {mean(lengths):.1f}" if lengths else "average_document_length_chars: 0")
    print(f"top_entity_title_examples: {', '.join(entity_examples) if entity_examples else 'n/a'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
