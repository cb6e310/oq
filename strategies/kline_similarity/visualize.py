"""Standardized Top-10 similarity results with single-chart SVG visuals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .engine import KlineSimilarityEngine, ParquetDataProvider, SimilarityConfig, _normalize_timeframe


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


def _calendar_range(start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert a matched period's labels into its covered calendar dates."""
    if timeframe == "1w":
        return start - pd.Timedelta(days=6), end
    if timeframe == "1m":
        return start.to_period("M").start_time, end.to_period("M").end_time.normalize()
    return start, end


def _bar_overlaps(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, period: str) -> pd.Series:
    dates = pd.to_datetime(bars.date)
    if period == "D":
        left, right = dates, dates
    elif period == "W":
        left, right = dates - pd.Timedelta(days=6), dates
    else:
        left = dates.dt.to_period("M").dt.start_time
        right = dates.dt.to_period("M").dt.end_time.dt.normalize()
    return (left <= end) & (right >= start)


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

    Each result contains three charts in the existing strategy style: daily
    K-lines with up to 90 bars before and after the match, plus weekly and
    monthly K-lines with up to 22 bars on each side.  ``image_path`` remains
    the daily chart for compatibility; ``image_paths`` contains all three.
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
        daily_bars = aggregate_bars(engine.provider.get_symbol_data(match.symbol), "D")
        target_start, target_end = _calendar_range(pd.Timestamp(match.start_date), pd.Timestamp(match.end_date), tf)
        daily_target = daily_bars[_bar_overlaps(daily_bars, target_start, target_end, "D")]
        if daily_target.empty:
            continue
        first, last = int(daily_target.index[0]), int(daily_target.index[-1])
        daily_window = daily_bars.iloc[max(0, first - 90): min(len(daily_bars), last + 91)].reset_index(drop=True)
        weekly_bars = aggregate_bars(engine.provider.get_symbol_data(match.symbol), "W")
        monthly_bars = aggregate_bars(engine.provider.get_symbol_data(match.symbol), "M")
        weekly_target = weekly_bars[_bar_overlaps(weekly_bars, target_start, target_end, "W")]
        monthly_target = monthly_bars[_bar_overlaps(monthly_bars, target_start, target_end, "M")]
        def centered_period(bars: pd.DataFrame, target: pd.DataFrame, side: int) -> pd.DataFrame:
            if target.empty:
                return bars.iloc[0:0].copy()
            lo, hi = int(target.index[0]), int(target.index[-1])
            return bars.iloc[max(0, lo - side): min(len(bars), hi + side + 1)].reset_index(drop=True)
        weekly_window = centered_period(weekly_bars, weekly_target, 22)
        monthly_window = centered_period(monthly_bars, monthly_target, 22)
        stem = f"{rank:02d}_{_safe_name(match.symbol)}_{match.start_date:%Y%m%d}_{match.end_date:%Y%m%d}_{tf}"
        subtitle = (f"相似度 {match.score:.4f} · MASS {match.mass_distance:.4f} · "
                    f"DTW {match.dtw_distance:.4f} · 区间 {match.start_date:%Y-%m-%d} 至 {match.end_date:%Y-%m-%d} · "
                    f"特殊板事件距离 {match.event_distance:.4f}")
        daily_path = out / f"{stem}_daily.svg"
        weekly_path = out / f"{stem}_weekly.svg"
        monthly_path = out / f"{stem}_monthly.svg"
        daily_path.write_text(render_chart(
            daily_window, f"{match.symbol} 日线 · 相似排名 {rank}",
            subtitle + " · 日线目标前后各90根",
            target_date=match.start_date, period="D", target_start=target_start, target_end=target_end,
        ), encoding="utf-8")
        daily_range = (daily_window.date.min(), daily_window.date.max())
        weekly_path.write_text(render_chart(
            weekly_window, f"{match.symbol} 周线 · 相似排名 {rank}",
            subtitle + " · 周线目标前后各22根",
            target_date=match.start_date, period="W", range_start=daily_range[0], range_end=daily_range[1],
            target_start=target_start, target_end=target_end,
        ), encoding="utf-8")
        monthly_path.write_text(render_chart(
            monthly_window, f"{match.symbol} 月线 · 相似排名 {rank}",
            subtitle + " · 月线目标前后各22根",
            target_date=match.start_date, period="M", range_start=daily_range[0], range_end=daily_range[1],
            target_start=target_start, target_end=target_end,
        ), encoding="utf-8")
        item = match.to_dict()
        item.update({"rank": rank, "timeframe": tf, "interval": {
            "start_date": item["start_date"], "end_date": item["end_date"],
        }, "image_path": str(daily_path), "image": str(daily_path), "image_paths": {
            "daily": str(daily_path), "weekly": str(weekly_path), "monthly": str(monthly_path),
        }})
        results.append(item)
    index = out / f"similarity_{_safe_name(query_symbol)}_{tf}_top{top_k}.html"
    body = ["<!doctype html><meta charset='utf-8'><style>", mobile_css(), "</style><div class='wrap'>",
            f"<div class='head'><h1>{query_symbol} {tf} 相似 K 线 Top {len(results)}</h1>",
            f"<p>查询区间：{query_start:%Y-%m-%d} 至 {query_end:%Y-%m-%d}｜日线前后各90根；周线、月线前后各22根</p></div>"]
    for item in results:
        body.append(f"<div class='sample'><div class='sample-head'>排名 {item['rank']} · {item['symbol']} · "
                    f"{item['start_date']} 至 {item['end_date']} · 相似度 {item['score']:.4f}</div>")
        for label in ("daily", "weekly", "monthly"):
            body.append(f"<div class='period-label'>{label}</div>")
            body.append(Path(item["image_paths"][label]).read_text(encoding="utf-8"))
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
