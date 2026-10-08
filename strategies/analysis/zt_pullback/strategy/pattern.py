"""统计给定通达信 ZT 回调公式的事件与前向收益。"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import event_study as es

HORIZONS = (1, 3, 5, 10, 20)
DROP_BINS = [-np.inf, -0.07, -0.03, 0, np.inf]
DROP_LABELS = ["暴力(<-7%)", "中等(-7%~-3%)", "温和(-3%~0)", "上涨"]


def find_events(df, start):
    rows = []
    positions = []
    zt_positions = []
    for code, group in df.groupby("ts_code", observed=True, sort=False):
        group = group.reset_index()
        close = group.close.to_numpy()
        open_ = group.open.to_numpy()
        high = group.high.to_numpy()
        low = group.low.to_numpy()
        prev = group.prev_close.to_numpy()
        zt = np.zeros(len(group), dtype=bool)
        zt[1:] = close[1:] / prev[1:] > 1.09
        last_zt = -1
        for i in range(len(group)):
            if zt[i]:
                last_zt = i
                continue
            t = i - last_zt
            if last_zt < 0 or t < 1 or t > 7:
                continue
            up_c = close[last_zt]
            up_pre = close[last_zt - 1] if last_zt > 0 else np.nan
            if not np.isfinite(up_pre):
                continue
            window = slice(last_zt + 1, i + 1)
            window_close = close[window]
            window_high = high[window]
            window_low = low[window]
            window_open = open_[window]
            prior_close = close[last_zt: i]
            close_ratio = window_close / prior_close
            open_ratio = window_close / window_open
            bad_yin = (close_ratio <= 0.95) | (open_ratio <= 0.95)
            half = up_pre + (up_c - up_pre) * 0.5
            ok = (
                (np.nanmax(window_high) <= up_c * 1.08)
                and (np.nanmin(window_low) >= half)
                and (window_close.mean() >= up_c * 0.98)
                and (np.nanmin(window_close) >= up_c * 0.96)
                and (close[i] >= up_c * 0.97)
                and (int(bad_yin.sum()) == 0)
                and (group.date.iloc[i] >= pd.Timestamp(start))
            )
            if not ok:
                continue
            rows.append({
                "ts_code": str(code), "signal_date": group.date.iloc[i], "zt_date": group.date.iloc[last_zt],
                "zt_pre_date": group.date.iloc[last_zt - 1], "t": t, "board": group.board.iloc[i],
                "is_main": str(code).startswith(("60", "00")), "zt_close": up_c,
                "zt_pre_close": up_pre, "zt_return": close[last_zt] / prev[last_zt] - 1,
                "signal_close": close[i], "signal_day_pct": group.pct.iloc[i],
                "pullback_pct": close[i] / up_c - 1, "center_ratio": window_close.mean() / up_c - 1,
                "high_ratio": np.nanmax(window_high) / up_c - 1, "low_ratio": np.nanmin(window_low) / up_c - 1,
                "close_low_ratio": np.nanmin(window_close) / up_c - 1, "bad_yin_count": int(bad_yin.sum()),
            })
            positions.append(int(group["index"].iloc[i]))
            zt_positions.append(int(group["index"].iloc[last_zt]))
    events = pd.DataFrame(rows)
    if events.empty:
        return events, np.array([], dtype=int), np.array([], dtype=int)
    events["pullback_bucket"] = pd.cut(events.pullback_pct, DROP_BINS, labels=DROP_LABELS).astype(str)
    events["signal_direction"] = np.select(
        [events.signal_day_pct < 0, events.signal_day_pct > 0], ["信号日收跌", "信号日收涨"], default="信号日收平"
    )
    events["year"] = pd.DatetimeIndex(events.signal_date).year
    return events, np.asarray(positions), np.asarray(zt_positions)


def fmt_pct(value):
    return "" if pd.isna(value) else f"{value:+.2%}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()
    end = es.latest_date()
    start = end - pd.DateOffset(years=args.years)
    df = es.load_bars(start)
    events, signal_pos, zt_pos = find_events(df, start)
    if events.empty:
        raise RuntimeError("公式未找到事件")
    forward = es.forward_returns(df, signal_pos, HORIZONS)
    path = es.signal_close_to_t25_best_close(df, signal_pos)
    market = es.market_index(df).close_idx.reindex(df.date).to_numpy()
    start_path = path.return_start_pos.to_numpy()
    end_path = path.return_end_pos.to_numpy()
    valid_path = (start_path >= 0) & (end_path >= 0)
    market_path = np.full(len(path), np.nan)
    market_path[valid_path] = market[end_path[valid_path]] / market[start_path[valid_path]] - 1
    path["ex_close_to_best_close_25"] = path.ret_close_to_best_close_25 - market_path
    events = pd.concat([events.reset_index(drop=True), forward, path], axis=1)
    print(f"区间: {start.date()} ~ {end.date()}")
    print(f"事件数: {len(events):,}，涉及股票: {events.ts_code.nunique():,}，主板: {events.is_main.sum():,}")
    for col in ["year", "t", "pullback_bucket", "signal_direction", "board", "is_main"]:
        print(f"按 {col}: " + ", ".join(f"{k}={v}" for k, v in events[col].value_counts().sort_index().items()))
    summary = [es.summarize(events)]
    for col in ["t", "pullback_bucket", "signal_direction", "board", "is_main", "year"]:
        summary.append(es.summarize(events, col))
    overall = es.stats(events.ret_close_to_best_close_25, events.ex_close_to_best_close_25)
    summary.append(pd.DataFrame([{"group_by": "全部", "group": "全部", "entry": "signal_close_to_best_close", "horizon": 25, **overall}]))
    es.RESULTS.mkdir(parents=True, exist_ok=True)
    events.to_csv(es.RESULTS / "events.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.concat(summary, ignore_index=True).to_csv(es.RESULTS / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print("T+25路径: " + ", ".join(f"{k}={fmt_pct(v)}" for k, v in overall.items() if k in ["mean", "median", "p10", "p90", "ex_mean"]))
    print(f"明细 -> {es.RESULTS / 'events.csv'}")


if __name__ == "__main__":
    main()
