from __future__ import annotations

import argparse

from backend.app.config import get_settings
from backend.app.database import BenchmarkDatabase
from backend.app.services.benchmark import BenchmarkService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark results to JSON, CSV, or Markdown.")
    parser.add_argument("--format", choices=["json", "csv", "markdown"], default="markdown")
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    service = BenchmarkService(settings, BenchmarkDatabase(settings.database_path))
    export_format, content = service.export(args.format, args.limit)
    suffix = "md" if export_format == "markdown" else export_format
    path = settings.report_dir / f"benchmark_report.{suffix}"
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {export_format} report to {path}")


if __name__ == "__main__":
    main()

