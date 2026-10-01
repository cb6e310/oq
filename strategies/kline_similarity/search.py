"""Command-line entry point for historical K-line similarity search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import KlineSimilarityEngine, ParquetDataProvider, SimilarityConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Search similar historical A-share K-line windows")
    parser.add_argument("symbol")
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--timeframe", "--period", default="1d", choices=("1d", "1w", "1m"))
    parser.add_argument("--data", default="database/processed/stock_daily_qfq.parquet")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--recall-n", type=int, default=1000)
    parser.add_argument("--include-future", action="store_true", help="allow candidates ending on/after query start")
    parser.add_argument("--no-volume", action="store_true")
    args = parser.parse_args()
    provider = ParquetDataProvider(Path(args.data))
    config = SimilarityConfig(top_k=args.top_k, recall_n=args.recall_n, use_volume=not args.no_volume)
    result = KlineSimilarityEngine(provider, config).search(
        args.symbol, args.start, args.end, timeframe=args.timeframe,
        history_only=not args.include_future,
    )
    print(json.dumps([x.to_dict() for x in result], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
