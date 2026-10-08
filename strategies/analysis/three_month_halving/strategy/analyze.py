"""Study the first >50% adjusted-close drawdown within 63 stock trading bars.

Run: python strategies/analysis/three_month_halving/strategy/analyze.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "strategies" / "analysis" / "three_month_halving"
DAILY = ROOT / "database" / "processed" / "stock_daily.parquet"
BASIC = ROOT / "database" / "processed" / "stock_basic_tdx.parquet"
RESULTS = PROJECT / "results"
DOCS = PROJECT / "docs"
WINDOW = 63
HORIZON = 25
TWO_YEAR_BARS = 504
BINS = [-np.inf, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, np.inf]
LABELS = ["<-50%", "-50%~-30%", "-30%~-20%", "-20%~-10%", "-10%~0%",
          "0%~10%", "10%~20%", "20%~30%", "30%~50%", "50%~100%", ">=100%"]


def study_stock(dates: np.ndarray, prices: np.ndarray, highs: np.ndarray | None = None,
                window: int = WINDOW,
                horizon: int = HORIZON, two_year_bars: int = TWO_YEAR_BARS) -> list[dict]:
    """Use prior-window peak and the first strict crossing below half that peak.

    A full prior window is required. Accepted events are at least `window` stock
    bars apart; this prevents one prolonged decline from dominating the sample.
    """
    n = len(prices)
    if highs is None:
        highs = prices
    if len(highs) != n:
        raise ValueError("highs and prices must have the same length")
    if n <= window:
        return []
    prior_two_year_high = (pd.Series(prices).shift(1)
                           .rolling(two_year_bars, min_periods=two_year_bars).max().to_numpy())
    prior_peak = pd.Series(prices).shift(1).rolling(window, min_periods=window).max().to_numpy()
    drawdown = prices / prior_peak - 1
    previous = np.r_[np.nan, drawdown[:-1]]
    crossed = (drawdown < -0.5) & (~np.isfinite(previous) | (previous >= -0.5))
    events = []
    previous_event = -window
    for t in np.flatnonzero(crossed):
        if t - previous_event < window or not np.isfinite(prices[t]) or prices[t] <= 0:
            continue
        peak_at = t - window + int(np.argmax(prices[t - window:t]))
        end = t + horizon
        complete = (end < n and np.isfinite(prices[end]) and prices[end] > 0
                    and np.isfinite(highs[t + 1:end + 1]).all())
        best_at = t + 1 + int(np.argmax(highs[t + 1:end + 1])) if complete else None
        events.append({
            "peak_date": dates[peak_at], "peak_close_adj": prices[peak_at],
            "peak_prior_2y_max_close_adj": prior_two_year_high[peak_at],
            "peak_has_2y_history": bool(np.isfinite(prior_two_year_high[peak_at])),
            "peak_is_2y_new_high": bool(np.isfinite(prior_two_year_high[peak_at])
                                         and prices[peak_at] > prior_two_year_high[peak_at]),
            "signal_date": dates[t], "signal_close_adj": prices[t],
            "peak_to_signal_bars": t - peak_at,
            "drawdown": prices[t] / prices[peak_at] - 1,
            "t25_date": dates[end] if complete else pd.NaT,
            "t25_close_adj": prices[end] if complete else np.nan,
            "terminal_return_t25": prices[end] / prices[t] - 1 if complete else np.nan,
            "best_date_t25": dates[best_at] if complete else pd.NaT,
            "best_high_adj_t25": highs[best_at] if complete else np.nan,
            "best_return_t25": highs[best_at] / prices[t] - 1 if complete else np.nan,
            "observation_status": "complete" if complete else "no_complete_t25_window",
        })
        previous_event = t
    return events


def load_events(window: int, horizon: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    daily = pd.read_parquet(DAILY, columns=["ts_code", "date", "high", "close", "adj_factor"])
    daily = daily.sort_values(["ts_code", "date"], kind="stable")
    latest = daily.date.max()
    daily["adjusted_close"] = daily.close * daily.adj_factor
    daily["adjusted_high"] = daily.high * daily.adj_factor
    rows = []
    for code, bars in daily.groupby("ts_code", observed=True, sort=False):
        valid = bars.adjusted_close.notna() & np.isfinite(bars.adjusted_close) & (bars.adjusted_close > 0)
        # Invalid price rows break the contiguous window rather than silently disappearing.
        if not valid.all():
            bars = bars.copy()
            bars.loc[~valid, "adjusted_close"] = np.nan
        for item in study_stock(bars.date.to_numpy(), bars.adjusted_close.to_numpy(),
                                bars.adjusted_high.to_numpy(), window, horizon):
            item["ts_code"] = code
            rows.append(item)
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)
        if BASIC.exists():
            names = pd.read_parquet(BASIC, columns=["ts_code", "name"]).drop_duplicates("ts_code")
            events = events.merge(names, on="ts_code", how="left")
    return events, latest


def describe(returns: pd.Series) -> dict:
    x = returns.dropna()
    if x.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p10": np.nan,
                "p25": np.nan, "p75": np.nan, "p90": np.nan, "positive_rate": np.nan}
    return {"n": len(x), "mean": x.mean(), "median": x.median(),
            "p10": x.quantile(.1), "p25": x.quantile(.25), "p75": x.quantile(.75),
            "p90": x.quantile(.9), "positive_rate": (x > 0).mean()}


def pct(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:+.2%}"


def distribution_for(returns: pd.Series, metric: str) -> pd.DataFrame:
    categories = pd.cut(returns, BINS, labels=LABELS, right=False)
    result = categories.value_counts(sort=False).rename_axis("bucket").reset_index(name="n")
    result.insert(0, "metric", metric)
    result["share"] = result.n / len(returns) if len(returns) else 0.0
    return result


def make_report(events: pd.DataFrame, latest: pd.Timestamp,
                distribution: pd.DataFrame, yearly: pd.DataFrame,
                window: int, horizon: int) -> str:
    selected = events[events.peak_is_2y_new_high]
    complete = selected[selected.observation_status == "complete"]
    rows = [
        "# 下跌前起点创两年新高：腰斩后表现分布", "",
        f"数据截止 {latest.date()}。符合条件的事件 {len(selected):,} 个，其中具有完整 "
        f"T+{horizon} 窗口的样本 {len(complete):,} 个。", "",
        "## 研究口径", "",
        f"- 腰斩信号 T0：个股先有完整前 {window} 根日线，当日后复权收盘价首次严格低于"
        "这段区间最高后复权收盘价的一半。相邻入选事件至少间隔同样多的个股交易日。",
        f"- 下跌前起点：T0 前 {window} 根个股日线内最高后复权收盘价所在日。"
        "阶段下跌期间允许反弹。",
        f"- 起点须有此前完整 {TWO_YEAR_BARS} 根个股日线，其后复权收盘价严格高于"
        "这些日线的最高后复权收盘价。起点当天不参与历史基准；历史不足两年的事件不纳入筛选。",
        f"- 最高点涨幅取 T+1 至 T+{horizon} 最高后复权价，终点收益取 T+{horizon} "
        "后复权收盘价，均除以 T0 后复权收盘价再减 1。T+25 按个股交易日计算，停牌日不计；"
        "只统计完整窗口。", "",
        "## 总体表现", "",
        "| 指标 | 数量 | 均值 | 中位数 | P10 | P25 | P75 | P90 | >0 占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, name in (("best_return_t25", "窗口最高点涨幅"),
                         ("terminal_return_t25", f"T+{horizon} 终点收益")):
        s = describe(complete[metric])
        rows.append(f"| {name} | {s['n']:,} | {pct(s['mean'])} | "
                    f"{pct(s['median'])} | {pct(s['p10'])} | {pct(s['p25'])} | "
                    f"{pct(s['p75'])} | {pct(s['p90'])} | {s['positive_rate']:.1%} |")
    for metric, title in (("best_return_t25", "窗口最高点涨幅"),
                          ("terminal_return_t25", f"T+{horizon} 终点收益")):
        rows.extend(["", f"## {title}分布", "", "| 收益区间 | 样本数 | 占比 |",
                     "|---|---:|---:|"])
        rows.extend(f"| {r.bucket} | {r.n:,} | {r.share:.1%} |"
                    for r in distribution[distribution.metric == metric].itertuples())
    rows.extend(["", "## 按信号年份", "",
                 "| 年份 | 样本数 | 最高点均值 | 最高点中位数 | 最高点>0 | 终点均值 | 终点中位数 | 终点>0 |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    rows.extend(f"| {int(r.year)} | {int(r.n):,} | {pct(r.best_mean)} | {pct(r.best_median)} | "
                f"{r.best_positive_rate:.1%} | {pct(r.terminal_mean)} | {pct(r.terminal_median)} | "
                f"{r.terminal_positive_rate:.1%} |" for r in yearly.itertuples())
    rows.extend(["", "## 解读边界", "",
                 "起点由事后形成的腰斩区间识别。窗口最高价也是事后可见的盘中极值，"
                 "不是可预先锁定的卖出价；两种收益均未计实际买卖价格与费用。"
                 "已退市或数据末尾不足 T+25 日线的事件不进入收益分布。", "",
                 "## 文件", "",
                 "- `../results/events.csv`：事件明细，含起点、信号日、窗口最高点及终点。",
                 "- `../results/distribution.csv`：两个收益指标的分桶。",
                 "- `../results/yearly.csv`：年度统计。", ""])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=WINDOW, help="近三个月个股交易日数")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="信号后个股交易日数")
    args = parser.parse_args()
    if args.window < 2 or args.horizon < 1:
        parser.error("window 至少为 2，horizon 至少为 1")
    events, latest = load_events(args.window, args.horizon)
    if events.empty:
        raise RuntimeError("没有找到符合条件的事件，请检查数据和研究口径")
    selected = events[events.peak_is_2y_new_high].copy()
    selected_complete = selected[selected.observation_status == "complete"].copy()
    selected_distribution = pd.concat([
        distribution_for(selected_complete.best_return_t25, "best_return_t25"),
        distribution_for(selected_complete.terminal_return_t25, "terminal_return_t25"),
    ], ignore_index=True)
    selected_complete["year"] = selected_complete.signal_date.dt.year
    selected_yearly_rows = []
    for year, group in selected_complete.groupby("year"):
        best = describe(group.best_return_t25)
        terminal = describe(group.terminal_return_t25)
        selected_yearly_rows.append({"year": int(year), "n": best["n"],
                                     **{f"best_{k}": v for k, v in best.items() if k != "n"},
                                     **{f"terminal_{k}": v for k, v in terminal.items() if k != "n"}})
    selected_yearly = pd.DataFrame(selected_yearly_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    selected.to_csv(RESULTS / "events.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    selected_distribution.to_csv(RESULTS / "distribution.csv", index=False, encoding="utf-8-sig")
    selected_yearly.to_csv(RESULTS / "yearly.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    (DOCS / "report.md").write_text(
        make_report(events, latest, selected_distribution, selected_yearly, args.window, args.horizon),
        encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"数据截止 {latest.date()}；两年新高起点事件 {len(selected):,}；"
          f"完整 T+{args.horizon} 样本 {len(selected_complete):,}")
    print(f"报告：{DOCS / 'report.md'}")


if __name__ == "__main__":
    main()
