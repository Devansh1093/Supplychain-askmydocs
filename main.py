from __future__ import annotations

import argparse
import json
import sys

from src.config import load_config
from src.ingestion.ingest import ingest_all
from src.pipeline import SupplyDocsPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="SupplyDocsQA command line")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Parse PDFs and build local indexes").add_argument(
        "--accept-report", action="store_true", help="Proceed when extraction checks flag a document"
    )
    ask_parser = sub.add_parser("ask", help="Ask a question against the documents")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--mode", choices=["dense", "hybrid", "hybrid_rerank"], default="hybrid_rerank")
    args = parser.parse_args()
    config = load_config()

    if args.command == "ingest":
        report = ingest_all(config, accept_report=args.accept_report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    try:
        result = SupplyDocsPipeline(config).ask(args.query, args.mode)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
