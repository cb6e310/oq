"""Repeatable performance benchmark for the similarity engine."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .engine import KlineSimilarityEngine, ParquetDataProvider, SimilarityConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("symbol", nargs="?", default="601567.SH")
    p.add_argument("start", nargs="?", default="2026-09-14")
    p.add_argument("end", nargs="?", default="2026-09-29")
    p.add_argument("--timeframe", default="1d", choices=("1d", "1w", "1m"))
    p.add_argument("--data", default="database/processed/stock_daily_qfq.parquet")
    p.add_argument("--recall-n", type=int, default=1000)
    p.add_argument("--match-mode", choices=("fixed", "elastic"), default="fixed")
    p.add_argument("--elastic-coarse-step", type=int, default=8,
                   help="Compiled elastic recall stride; use 1 for exhaustive recall")
    p.add_argument("--latest-only", action="store_true")
    args = p.parse_args()
    start = time.perf_counter()
    provider = ParquetDataProvider(args.data)
    load_s = time.perf_counter() - start
    top_k = 10 if args.latest_only else 20
    engine = KlineSimilarityEngine(provider, SimilarityConfig(
        recall_n=args.recall_n, top_k=top_k, elastic_coarse_step=args.elastic_coarse_step,
    ))
    start = time.perf_counter()
    result = engine.search(args.symbol, args.start, args.end, timeframe=args.timeframe,
                           top_k=top_k, history_only=True, recall_n=args.recall_n,
                           match_mode=args.match_mode, latest_only=args.latest_only)
    search_s = time.perf_counter() - start
    print(json.dumps({
        "symbol": args.symbol, "start": args.start, "end": args.end,
        "timeframe": args.timeframe, "match_mode": args.match_mode,
        "candidate_scope": "latest" if args.latest_only else "history",
        "symbols": len(provider.get_symbols()),
        "recall_n": args.recall_n, "load_s": round(load_s, 3),
        "search_s": round(search_s, 3), "results": len(result),
        "top_symbols": [x.symbol for x in result[:10]],
        "search_stats": engine.last_search_stats,
        "pid": os.getpid(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
