from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.config import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ChromaDB index for Basic RAG.")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "processed" / "corpus.jsonl")
    parser.add_argument("--index-dir", type=Path, default=REPO_ROOT / "data" / "indexes" / "chroma")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus not found at {args.corpus}. Run backend/scripts/prepare_hotpotqa.py first.")

    import chromadb

    args.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.index_dir))
    collection = client.get_or_create_collection(name="hotpotqa_chunks", metadata={"hnsw:space": "cosine"})

    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict] = []
    total = 0
    with args.corpus.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            ids.append(str(item["id"]))
            docs.append(item["text"])
            metadatas.append(
                {
                    "title": item.get("title", "Untitled"),
                    "source": item.get("source", "hotpotqa"),
                    **{f"meta_{key}": value for key, value in item.get("metadata", {}).items() if isinstance(value, (str, int, float, bool))},
                }
            )
            if len(ids) >= args.batch_size:
                collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
                total += len(ids)
                ids, docs, metadatas = [], [], []
                print(f"Indexed {total:,} chunks")
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
        total += len(ids)
    print(f"ChromaDB index ready at {args.index_dir} with {total:,} chunks")


if __name__ == "__main__":
    main()
