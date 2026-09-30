"""Pattern: 前三根 K 线无涨停的准一字板，随后两根内首次回调到板价下方.

Pattern bars:
  1. D0 closes at limit-up and OHLC stays within one cent of the limit price
  2. none of the three stock bars immediately before D0 is a limit-up
  3. D1 and D2 are the two bars after D0; the first close below D0 is the signal
  4. Exclude suspected ST one-price boards (main-board 5% limit-up proxy)
  5. D0 adjusted price must not exceed 1.2x the minimum adjusted low over the prior 63 valid stock bars

Outputs: occurrence counts, forward-return stats (D2+1..D2+20, entry at D2 close / D3 open),
signal+20 return distribution, plus results/yizi_pullback_d2_{events,summary}.csv
Usage : python scripts/pattern_yizi_pullback.py [--years 10]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import event_study as es

HORIZONS = (1, 3, 5, 10, 20)
YIZI_TOLERANCE_CENTS = 1
DROP_BINS = [-np.inf, -0.07, -0.03, 0]
DROP_LABELS = ["暴力(<-7%)", "中等(-7%~-3%)", "温和(-3%~0)"]


def find_events(df, start):
    n = len(df)
    pos = np.arange(n)
    code = df.ts_code.cat.codes.values

    lu10 = (df.close_c == df.up_c).values
    lu5 = (df.close_c == df.up5_c).values
    any_limit_up = lu10 | lu5
    open_c = np.round(df.open.values * 100)
    high_c = np.round(df.high.values * 100)
    low_c = np.round(df.low.values * 100)
    close_c = df.close_c.values
    # Include near-one-price boards such as 002428.SZ on 2026-03-20, where
    # open/low were one cent below the limit but close/high were at limit-up.
    yizi_like = ((high_c - low_c <= YIZI_TOLERANCE_CENTS)
                 & (close_c - open_c <= YIZI_TOLERANCE_CENTS)
                 & (close_c - low_c <= YIZI_TOLERANCE_CENTS))
    yizi_up = yizi_like & any_limit_up & ~df.new_stock.values

    # This count was calculated on full per-stock history before the event window was trimmed.
    d0 = yizi_up & (df.limit_ups_prev3.values == 0)

    hfq_close = df.hfq_close.values
    k1 = pos - 1
    safe_k1 = np.maximum(k1, 0)
    same1 = (k1 >= 0) & (code[safe_k1] == code)
    sig_d1 = same1 & d0[safe_k1] & (hfq_close < hfq_close[safe_k1])

    k2 = pos - 2
    safe_k2 = np.maximum(k2, 0)
    same2 = (k2 >= 0) & (code[safe_k2] == code)
    # D2 is used only if D1 did not already close below D0, avoiding duplicate events.
    d1_below = hfq_close[np.maximum(pos - 1, 0)] < hfq_close[safe_k2]
    sig_d2 = same2 & d0[safe_k2] & ~d1_below & (hfq_close < hfq_close[safe_k2])

    month_ok = ((df.current_month_return.values <= 0.50)
                & (df.prev_month_return.values <= 0.50))
    # D0 adjusted price cannot exceed twice the minimum adjusted low over the
    # Prior three months: approximately 63 valid stock bars. The current D0
    # bar is excluded upstream; the D1/D2 candidate is checked against D0.
    prior_year_d1_ok = (same1 & (hfq_close[safe_k1] <= 1.2 * df.prior_year_low.values[safe_k1]))
    prior_year_d2_ok = (same2 & (hfq_close[safe_k2] <= 1.2 * df.prior_year_low.values[safe_k2]))
    sig = (((sig_d1 & prior_year_d1_ok) | (sig_d2 & prior_year_d2_ok))
           & month_ok & (df.date.values >= np.datetime64(start)))
    t = pos[sig]
    k = np.where(sig_d1[sig], k1[sig], k2[sig])
    d1 = k + 1
    gap = t - k

    st_like = yizi_up & lu5 & ~lu10
    pullback = hfq_close[t] / hfq_close[k] - 1
    d1_return = hfq_close[d1] / hfq_close[k] - 1
    d1_limit_up = any_limit_up[d1]

    ev = pd.DataFrame({
        "ts_code": df.ts_code.values[t], "signal_date": df.date.values[t],
        "yizi_date": df.date.values[k], "d1_date": df.date.values[d1], "gap_bars": gap,
        "board": df.board.values[t], "st_like": st_like[k],
        "yizi_close": df.close.values[k], "d1_close": df.close.values[d1],
        "close": df.close.values[t], "d1_return": d1_return,
        "drop_pct": pullback, "signal_day_pct": df.pct.values[t],
        "d1_limit_up": d1_limit_up,
        "current_month_return": df.current_month_return.values[t],
        "prev_month_return": df.prev_month_return.values[t],
        "prior_year_low": df.prior_year_low.values[k],
        "prior_year_ratio": hfq_close[k] / df.prior_year_low.values[k] - 1,
    })
    ev["drop_bucket"] = pd.cut(ev.drop_pct, DROP_BINS, labels=DROP_LABELS).astype(str)
    ev["d1_direction"] = np.select(
        [ev.d1_return < 0, ev.d1_return > 0], ["D1收跌", "D1收涨"], "D1收平"
    )
    ev["year"] = pd.DatetimeIndex(ev.signal_date).year
    n_first_yizi = int((d0 & (df.date >= pd.Timestamp(start))).sum())
    return ev, t, n_first_yizi


def fmt_pct(x):
    return "" if pd.isna(x) else f"{x:+.2%}"


def print_stats(s, title):
    print(f"\n== {title} ==")
    cols = ["group", "entry", "horizon", "n", "mean", "median", "win", "p10", "p90", "t", "ex_mean", "ex_win"]
    out = s[cols].copy()
    for c in ["mean", "median", "p10", "p90", "ex_mean"]:
        out[c] = out[c].map(fmt_pct)
    for c in ["win", "ex_win"]:
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    out["t"] = out["t"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    print(out.to_string(index=False))


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to GBK
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    args = ap.parse_args()

    end = es.latest_date()
    start = end - pd.DateOffset(years=args.years)
    df = es.load_bars(start)

    ev, t, n_first_yizi = find_events(df, start)
    fr = es.forward_returns(df, t, HORIZONS)
    ev = pd.concat([ev, fr], axis=1)
    # The source data has no formal ST flag.  st_like is the documented proxy:
    # a main-board one-price 5% limit-up that is not also the normal 10% limit.
    n_before_st = len(ev)
    ev = ev[~ev.st_like].reset_index(drop=True)
    tag = "yizi_pullback_d2"

    # ---- occurrence counts
    print(f"区间: {start.date()} ~ {end.date()}")
    print(f"前3根K线无涨停且本日为一字涨停(非新股期): {n_first_yizi:,} 次"
          f"  -> 去除疑似ST: {n_before_st:,} -> {len(ev):,} 次"
          f"  ->  D2收盘低于一字板价: {len(ev):,} 次, 涉及 {ev.ts_code.nunique():,} 只股票")
    print(f"其中次日一字涨停无法开盘买入: {int(ev.entry_blocked.sum())} 次 ({ev.entry_blocked.mean():.1%})，"
          f"t+20 数据不足: {int(ev.ret_close_20.isna().sum())} 次")
    for col, name in [("year", "年份"), ("drop_bucket", "信号日相对一字板回调力度"),
                      ("gap_bars", "首次跌破发生在D1/D2"),
                      ("d1_direction", "D1方向"), ("d1_limit_up", "D1是否涨停"),
                      ("board", "板块"), ("st_like", "疑似ST(5%板)")]:
        c = ev[col].value_counts().sort_index()
        print(f"  按{name}: " + ", ".join(f"{k}={v}" for k, v in c.items()))

    # ---- forward-return stats
    overall = es.summarize(ev, None, HORIZONS)
    print_stats(overall, "全部事件, 各持有期 (entry=close 信号日收盘买入; open 次日开盘买入, 剔除买不进)")
    parts = [overall]
    for col in ["drop_bucket", "gap_bars", "d1_direction", "d1_limit_up", "board", "st_like", "year"]:
        s = es.summarize(ev, col, HORIZONS)
        parts.append(s)
        print_stats(s[s.horizon == 20], f"按 {col} 分组, t+20")

    # ---- t+20 distribution
    d = pd.concat({"close买入": es.distribution(ev.ret_close_20),
                   "open买入": es.distribution(ev.ret_open_20[~ev.entry_blocked])}, axis=1)
    for c in d.columns:
        if c[1] != "count":
            d[c] = d[c].map(lambda v: f"{v:.1%}")
    print("\n== t+20 收益分布 ==")
    print(d.to_string())

    os.makedirs(es.RESULTS, exist_ok=True)
    ev.to_csv(os.path.join(es.RESULTS, f"{tag}_events.csv"), index=False,
              encoding="utf-8-sig", float_format="%.4f")
    pd.concat(parts).to_csv(os.path.join(es.RESULTS, f"{tag}_summary.csv"), index=False,
                            encoding="utf-8-sig", float_format="%.4f")
    print(f"\n明细 -> results/{tag}_events.csv, 统计 -> results/{tag}_summary.csv")


if __name__ == "__main__":
    main()

