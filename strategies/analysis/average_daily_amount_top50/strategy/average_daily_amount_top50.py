"""统计指定区间内日均成交额最大的股票。

口径：读取 ``stock_daily.parquet`` 的 ``amount`` 字段（成交额，元），
对每只股票在区间内实际存在的日线记录求算术平均，不使用 ``vol`` 成交量。

用法示例::

    python strategies/analysis/average_daily_amount_top50/strategy/average_daily_amount_top50.py
    python ... --start 2021-12-01 --end 2022-04-30 --top 50 --min-days 80

输出：``results/top50_average_daily_amount.csv``。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DAILY = ROOT / "database" / "processed" / "stock_daily.parquet"
BASIC = ROOT / "database" / "processed" / "stock_basic_tdx.parquet"
STRATEGY = ROOT / "strategies" / "analysis" / "average_daily_amount_top50"
RESULTS = STRATEGY / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按成交额统计区间内日均成交额前N只股票")
    parser.add_argument("--start", default="2021-12-01", help="开始日期，含当日")
    parser.add_argument("--end", default="2022-04-30", help="结束日期，含当日")
    parser.add_argument("--top", type=int, default=50, help="输出股票数量")
    parser.add_argument(
        "--min-days",
        type=int,
        default=1,
        help="区间内至少有多少条实际日线记录；默认1，不把缺失日填成0",
    )
    return parser.parse_args()


def load_basic() -> pd.DataFrame:
    if not BASIC.exists():
        return pd.DataFrame(columns=["ts_code", "name"])
    return pd.read_parquet(BASIC, columns=["ts_code", "name"])


def calculate(start: str, end: str, top: int = 50, min_days: int = 1) -> pd.DataFrame:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if end_date < start_date:
        raise ValueError("--end 不能早于 --start")
    if top <= 0 or min_days <= 0:
        raise ValueError("--top 和 --min-days 必须为正整数")

    # 只读 amount；vol 明确不参与任何计算。
    daily = pd.read_parquet(
        DAILY,
        columns=["date", "ts_code", "amount"],
        filters=[("date", ">=", start_date), ("date", "<=", end_date)],
    )
    daily = daily.dropna(subset=["ts_code", "amount"])
    daily = daily[daily["amount"] >= 0]
    market_days = int(daily["date"].nunique())

    summary = (
        daily.groupby("ts_code", observed=True)
        .agg(
            trading_days=("date", "nunique"),
            total_amount=("amount", "sum"),
            average_daily_amount=("amount", "mean"),
            max_daily_amount=("amount", "max"),
        )
        .reset_index()
    )
    summary = summary[summary["trading_days"] >= min_days].copy()
    summary["coverage"] = summary["trading_days"] / market_days if market_days else 0.0
    summary = summary.merge(load_basic(), on="ts_code", how="left")
    summary["rank"] = summary["average_daily_amount"].rank(method="first", ascending=False).astype(int)
    summary = summary.sort_values(
        ["average_daily_amount", "trading_days", "ts_code"],
        ascending=[False, False, True],
    ).head(top)
    summary["rank"] = range(1, len(summary) + 1)
    summary["average_daily_amount_yi"] = summary["average_daily_amount"] / 1e8
    summary["total_amount_yi"] = summary["total_amount"] / 1e8
    summary["max_daily_amount_yi"] = summary["max_daily_amount"] / 1e8
    summary["period_start"] = start_date.date().isoformat()
    summary["period_end"] = end_date.date().isoformat()
    summary["market_days"] = market_days
    columns = [
        "rank", "ts_code", "name", "average_daily_amount", "average_daily_amount_yi",
        "trading_days", "coverage", "total_amount", "total_amount_yi",
        "max_daily_amount", "max_daily_amount_yi", "period_start", "period_end", "market_days",
    ]
    return summary[columns]


def main() -> None:
    args = parse_args()
    result = calculate(args.start, args.end, args.top, args.min_days)
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"top{args.top}_average_daily_amount.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"区间: {args.start} ~ {args.end}")
    print(f"成交额字段: stock_daily.amount（元）；市场交易日数: {int(result.market_days.iloc[0]) if len(result) else 0}")
    print(f"结果: {output}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
