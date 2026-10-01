"""Standardized Top-10 similarity results with single-chart SVG visuals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .engine import KlineSimilarityEngine, ParquetDataProvider, SimilarityConfig, _get_bars, _normalize_timeframe


_PERIOD = {"1d": "D", "1w": "W", "1m": "M"}


def _chart_tools():
    # Keep the plotting convention shared with the existing strategy charts.
    import sys
    root = Path(__file__).resolve().parents[2]
    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from candlestick_svg import aggregate_bars, mobile_css, render_chart
    return aggregate_bars, mobile_css, render_chart


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def build_visual_results(
    engine: KlineSimilarityEngine,
    query_symbol: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    timeframe: str = "1d",
    output_dir: str | Path = "strategies/kline_similarity/results",
    top_k: int = 10,
    history_only: bool = True,
) -> list[dict]:
    """Search and render a standardized result list.

    Each SVG contains exactly the matched interval plus up to 44 bars before
    and after it, using the project's logarithmic-price, volume, MA5/10/20
    candlestick style.  ``image_path`` is absolute so callers can embed it
    directly.  The returned dictionaries are JSON serializable.
    """
    tf = _normalize_timeframe(timeframe)
    matches = engine.search(query_symbol, start_date, end_date, timeframe=tf,
                            top_k=top_k, history_only=history_only)
    aggregate_bars, mobile_css, render_chart = _chart_tools()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    period = _PERIOD[tf]
    query_bars = aggregate_bars(engine.provider.get_symbol_data(query_symbol), period)
    query_start, query_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    q = query_bars[(query_bars.date >= query_start) & (query_bars.date <= query_end)]
    if q.empty:
        raise ValueError("query interval has no bars in the selected timeframe")
    results = []
    for rank, match in enumerate(matches, 1):
        bars = aggregate_bars(engine.provider.get_symbol_data(match.symbol), period)
        target = bars[(bars.date >= match.start_date) & (bars.date <= match.end_date)]
        if target.empty:
            continue
        first, last = int(target.index[0]), int(target.index[-1])
        window = bars.iloc[max(0, first - 44): min(len(bars), last + 45)].reset_index(drop=True)
        filename = f"{rank:02d}_{_safe_name(match.symbol)}_{match.start_date:%Y%m%d}_{match.end_date:%Y%m%d}_{tf}.svg"
        path = out / filename
        subtitle = (f"相似度 {match.score:.4f} · MASS {match.mass_distance:.4f} · "
                    f"DTW {match.dtw_distance:.4f} · 区间 {match.start_date:%Y-%m-%d} 至 {match.end_date:%Y-%m-%d} · "
                    f"前后各44根（当前历史/未来走势）")
        svg = render_chart(window, f"{match.symbol} {period}线 · 相似排名 {rank}", subtitle,
                           target_date=match.start_date, period=period,
                           target_start=match.start_date, target_end=match.end_date)
        path.write_text(svg, encoding="utf-8")
        item = match.to_dict()
        item.update({"rank": rank, "timeframe": tf, "interval": {
            "start_date": item["start_date"], "end_date": item["end_date"],
        }, "image_path": str(path), "image": str(path)})
        results.append(item)
    index = out / f"similarity_{_safe_name(query_symbol)}_{tf}_top{top_k}.html"
    body = ["<!doctype html><meta charset='utf-8'><style>", mobile_css(), "</style><div class='wrap'>",
            f"<div class='head'><h1>{query_symbol} {tf} 相似 K 线 Top {len(results)}</h1>",
            f"<p>查询区间：{query_start:%Y-%m-%d} 至 {query_end:%Y-%m-%d}｜目标周期前后各44根 K 线</p></div>"]
    for item in results:
        body.append(f"<div class='sample'><div class='sample-head'>排名 {item['rank']} · {item['symbol']} · "
                    f"{item['start_date']} 至 {item['end_date']} · 相似度 {item['score']:.4f}</div>")
        body.append(Path(item["image_path"]).read_text(encoding="utf-8"))
        body.append("</div>")
    body.append("</div>")
    index.write_text("".join(body), encoding="utf-8")
    for item in results:
        item["html_path"] = str(index)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-10 similar K-line intervals with SVG charts")
    parser.add_argument("symbol")
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--timeframe", "--period", default="1d", choices=("1d", "1w", "1m"))
    parser.add_argument("--data", default="database/processed/stock_daily_qfq.parquet")
    parser.add_argument("--output-dir", default="strategies/kline_similarity/results")
    parser.add_argument("--recall-n", type=int, default=1000)
    parser.add_argument("--include-future", action="store_true")
    args = parser.parse_args()
    engine = KlineSimilarityEngine(ParquetDataProvider(args.data), SimilarityConfig(recall_n=args.recall_n, top_k=10))
    result = build_visual_results(engine, args.symbol, args.start, args.end,
                                  timeframe=args.timeframe, output_dir=args.output_dir,
                                  top_k=10, history_only=not args.include_future)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
