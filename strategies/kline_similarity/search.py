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
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--recall-n", type=int, default=1000)
    parser.add_argument("--include-future", action="store_true", help="allow candidates ending on/after query start")
    parser.add_argument("--latest-only", action="store_true", help="compare each stock's latest window only")
    parser.add_argument("--no-volume", action="store_true")
    parser.add_argument("--match-mode", choices=("fixed", "elastic"), default="fixed",
                        help="fixed keeps equal-length windows; elastic searches bounded variable lengths")
    parser.add_argument("--min-length", type=int, default=None,
                        help="minimum candidate length in elastic mode")
    parser.add_argument("--max-length", type=int, default=None,
                        help="maximum candidate length in elastic mode")
    args = parser.parse_args()
    provider = ParquetDataProvider(Path(args.data))
    config = SimilarityConfig(top_k=args.top_k or (10 if args.latest_only else 20), recall_n=args.recall_n, use_volume=not args.no_volume,
                              length_mode=args.match_mode)
    result = KlineSimilarityEngine(provider, config).search(
        args.symbol, args.start, args.end, timeframe=args.timeframe,
        history_only=not args.include_future,
        match_mode=args.match_mode, min_length=args.min_length, max_length=args.max_length,
        latest_only=args.latest_only,
    )
    print(json.dumps([x.to_dict() for x in result], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
