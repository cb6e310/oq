"""Rank leading A-shares over a date interval and summarize adjusted price extremes.

Run from the repository root:
    python strategies/analysis/interval_stock_performance/strategy/analyze.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "strategies" / "analysis" / "interval_stock_performance"
DAILY = ROOT / "database" / "processed" / "stock_daily_qfq.parquet"
NAMES = ROOT / "database" / "raw" / "tdx" / "meta" / "infoharbor_ex.code"


def load_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_bytes().decode("gbk", errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0].isdigit():
            result[parts[0]] = parts[1].strip()
    return result


def analyze(start: str, end: str, top_n: int) -> tuple[pd.DataFrame, int, int]:
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date:
        raise ValueError("--start must be on or before --end")
    daily = pd.read_parquet(DAILY, columns=["ts_code", "date", "high", "low", "close"])
    daily = daily[daily.date.between(start_date, end_date)].copy()
    if daily.empty:
        raise ValueError(f"No market data found in {start} through {end}")
    daily["adj_high"] = daily.high
    daily["adj_low"] = daily.low
    daily["adj_close"] = daily.close
    # Ordinary listed shares only; omit indices, funds and other non-stock instruments.
    daily = daily[daily.ts_code.str.match(r"^\d{6}\.(SH|SZ|BJ)$", na=False)]
    groups = daily.groupby("ts_code", sort=False, observed=True)
    endpoints = daily[daily.date.isin([start_date, end_date])].pivot_table(
        index="ts_code", columns="date", values="adj_close", aggfunc="last", observed=True
    )
    if start_date not in endpoints or end_date not in endpoints:
        raise ValueError("The local data does not contain both interval endpoint dates")
    endpoints = endpoints[[start_date, end_date]].dropna()
    endpoints = endpoints[(endpoints[start_date] > 0) & (endpoints[end_date] > 0)]
    by_stock = {code: frame for code, frame in groups}
    rows = []
    for code, endpoint in endpoints.iterrows():
        frame = by_stock[code].sort_values("date")
        high_idx, low_idx = frame.adj_high.idxmax(), frame.adj_low.idxmin()
        high_row, low_row = frame.loc[high_idx], frame.loc[low_idx]
        rows.append({
            "ts_code": code,
            "start_adj_close": endpoint[start_date],
            "end_adj_close": endpoint[end_date],
            "interval_return": endpoint[end_date] / endpoint[start_date] - 1,
            "interval_high_adj": high_row.adj_high,
            "interval_high_date": high_row.date,
            "interval_low_adj": low_row.adj_low,
            "interval_low_date": low_row.date,
            "observed_bars": len(frame),
        })
    result = pd.DataFrame(rows).sort_values(
        ["interval_return", "ts_code"], ascending=[False, True]
    ).head(top_n).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    names = load_names(NAMES)
    result.insert(2, "name", result.ts_code.str[:6].map(names).fillna("未匹配"))
    result["interval_high_date"] = pd.to_datetime(result.interval_high_date).dt.strftime("%Y-%m-%d")
    result["interval_low_date"] = pd.to_datetime(result.interval_low_date).dt.strftime("%Y-%m-%d")
    return result, len(endpoints), daily.date.nunique()


def write_report(result: pd.DataFrame, start: str, end: str, eligible: int, sessions: int) -> str:
    lines = [
        f"# 区间领涨股票统计：{start} 至 {end}", "",
        f"区间共 {sessions} 个交易日；首尾两日均有有效复权收盘价的股票 {eligible:,} 只。",
        f"按区间首日到末日的前复权收盘价涨幅排序，列出前 {len(result)} 名。",
        f"Top {len(result)} 涨幅范围为 {result.interval_return.min():+.2%} 至 {result.interval_return.max():+.2%}。", "",
        "## 统计口径", "",
        "- 涨幅 = 末日复权收盘价 ÷ 首日复权收盘价 - 1；首尾日期均包含在内。",
        "- 区间最高/最低分别为前复权日线中的每日最高价/最低价，并附对应日期。",
        "- 范围为沪深北股票代码格式；要求首尾日期都有有效收盘记录。停牌期间无日线，不作估算。",
        "- 前复权价格取自本地生成的前复权日线，按当前数据库末日锚定；后续重建数据可能改变绝对价格数值，但不改变本区间收益率口径。",
        "- 名称来自本地通达信当前证券名称表，可能与 2015 年历史名称不同；“未匹配”表示当前名称表无记录。", "",
        f"## Top {len(result)}", "",
        "| 排名 | 代码 | 名称 | 区间涨幅 | 首日复权收盘 | 末日复权收盘 | 区间最高复权价（日期） | 区间最低复权价（日期） | 有行情日数 |",
        "|---:|---|---|---:|---:|---:|---|---|---:|",
    ]
    for r in result.itertuples(index=False):
        lines.append(f"| {r.rank} | {r.ts_code} | {r.name} | {r.interval_return:+.2%} | "
                     f"{r.start_adj_close:.4f} | {r.end_adj_close:.4f} | "
                     f"{r.interval_high_adj:.4f}（{r.interval_high_date}） | "
                     f"{r.interval_low_adj:.4f}（{r.interval_low_date}） | {r.observed_bars} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-09-15")
    parser.add_argument("--end", default="2015-12-22")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    result, eligible, sessions = analyze(args.start, args.end, args.top)
    results_dir = PROJECT / "results"
    docs_dir = PROJECT / "docs"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stem = f"leaders_{args.start}_{args.end}"
    result.to_csv(results_dir / f"{stem}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    report = write_report(result, args.start, args.end, eligible, sessions)
    (docs_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Saved report to {docs_dir / 'report.md'}")
    print(f"Saved CSV to {results_dir / f'{stem}.csv'}")


if __name__ == "__main__":
    main()
