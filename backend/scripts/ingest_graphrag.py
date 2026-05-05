from __future__ import annotations

import argparse

from backend.app.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print TigerGraph GraphRAG ingestion configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without writing to TigerGraph.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    print("TigerGraph GraphRAG ingestion handoff")
    print(f"Corpus: {settings.corpus_path}")
    print(f"Graph: {settings.tigergraph_graph_name}")
    print(f"GraphRAG URL: {settings.tigergraph_graphrag_url or 'not configured'}")
    if args.dry_run or not settings.tigergraph_graphrag_url:
        print("Dry run only. Configure TIGERGRAPH_* env vars to ingest with your deployment workflow.")


if __name__ == "__main__":
    main()

