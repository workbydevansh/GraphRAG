from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend.app.config import get_settings
from backend.app.database import BenchmarkDatabase
from backend.app.services.benchmark import BenchmarkService, build_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all benchmark pipelines over the eval set.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-set", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    database = BenchmarkDatabase(settings.database_path)
    service = BenchmarkService(settings, database)
    runs = await service.run_batch(
        limit=args.limit,
        eval_set_path=str(args.eval_set) if args.eval_set else None,
        save=not args.no_save,
    )
    summary = build_summary(runs).model_dump(mode="json")
    report_path = settings.report_dir / "latest_batch_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())

