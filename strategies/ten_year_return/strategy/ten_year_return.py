"""Simple backtest: how did A-share stocks do over the last N years?

1. Per-stock holding return (后复权) from the start date (or listing, if later) to the end date
   (or last bar, if delisted/suspended). Delisted stocks are kept to avoid survivorship bias.
2. Equal-weight portfolio rebalanced daily: each day's return = mean return of stocks that
   traded that day. Compared against 沪深300 / 上证指数 price indices.

Input : database/processed/stock_daily.parquet (run build_adjusted.py first)
Output: results/ten_year_return_by_stock.csv
Usage : python scripts/ten_year_return.py [--years 10]
"""
import argparse
import sys
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, "database", "processed", "stock_daily.parquet")
RAW = os.path.join(ROOT, "database", "raw", "tdx")
OUT = os.path.join(ROOT, "results")
BENCHMARKS = {"沪深300": "sh000300", "上证指数": "sh000001"}


def load_index(key):
    dt = np.dtype([("date", "<u4"), ("o", "<u4"), ("h", "<u4"), ("l", "<u4"), ("close", "<u4"),
                   ("amount", "<f4"), ("vol", "<u4"), ("r", "<u4")])
    a = np.fromfile(os.path.join(RAW, key[:2], "lday", key + ".day"), dtype=dt)
    return pd.Series(a["close"] / 100.0, index=pd.to_datetime(a["date"].astype(str), format="%Y%m%d"))


def annualize(total, days):
    return (1 + total) ** (365.25 / days) - 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to GBK
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_parquet(DAILY, columns=["date", "ts_code", "close", "adj_factor"])
    end = df.date.max()
    start = end - pd.DateOffset(years=args.years)
    df = df[df.date >= start].sort_values(["ts_code", "date"])
    df["hfq"] = df.close * df.adj_factor

    # ---- 1. per-stock holding return
    g = df.groupby("ts_code", observed=True)
    s = pd.DataFrame({"first_date": g.date.first(), "last_date": g.date.last(),
                      "first_hfq": g.hfq.first(), "last_hfq": g.hfq.last()})
    s["total_return"] = s.last_hfq / s.first_hfq - 1
    s["days"] = (s.last_date - s.first_date).dt.days
    s["annual_return"] = np.where(s.days >= 365, annualize(s.total_return, s.days.clip(lower=1)), np.nan)
    s["full_period"] = (s.first_date <= start + pd.Timedelta(days=10)) & (s.last_date == end)
    s["status"] = np.where(s.last_date == end, "trading", "delisted/suspended")
    os.makedirs(OUT, exist_ok=True)
    s.sort_values("total_return", ascending=False).to_csv(
        os.path.join(OUT, "ten_year_return_by_stock.csv"), encoding="utf-8-sig", float_format="%.4f")

    print(f"区间: {start.date()} ~ {end.date()}  (后复权, 含退市股)\n")
    groups = {"全部股票": s, "区间内一直在交易": s[s.full_period],
              "区间内新上市": s[s.first_date > start + pd.Timedelta(days=10)],
              "已退市/停牌": s[s.status != "trading"]}
    print(f"{'分组':<14}{'股票数':>7}{'平均涨幅':>10}{'中位数':>10}{'上涨占比':>10}{'年化(中位)':>12}")
    for name, x in groups.items():
        print(f"{name:<14}{len(x):>9}{x.total_return.mean():>12.1%}{x.total_return.median():>12.1%}"
              f"{(x.total_return > 0).mean():>12.1%}{x.annual_return.median():>13.1%}")
    full = s[s.full_period].total_return
    print("\n一直在交易的股票，涨幅分位数:",
          ", ".join(f"p{int(q*100)}={full.quantile(q):.0%}" for q in (0.1, 0.25, 0.5, 0.75, 0.9)))

    # ---- 2. equal-weight daily-rebalanced portfolio vs benchmarks
    df["ret"] = df.groupby("ts_code", observed=True).hfq.pct_change()
    port = df.dropna(subset=["ret"]).groupby("date").ret.mean()
    curves = {"等权组合(每日再平衡)": (1 + port).cumprod()}
    for name, key in BENCHMARKS.items():
        idx = load_index(key)
        idx = idx[idx.index >= start]
        curves[name] = idx / idx.iloc[0]
    print(f"\n{'组合/指数':<18}{'总收益':>10}{'年化':>9}{'最大回撤':>10}")
    for name, c in curves.items():
        days = (c.index[-1] - c.index[0]).days
        mdd = (c / c.cummax() - 1).min()
        print(f"{name:<18}{c.iloc[-1] - 1:>12.1%}{annualize(c.iloc[-1] - 1, days):>11.1%}{mdd:>12.1%}")
    print(f"\n逐股明细 -> {os.path.join(OUT, 'ten_year_return_by_stock.csv')}")


if __name__ == "__main__":
    main()
