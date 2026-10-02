"""Reusable event-study helpers: load bars, price limits, new-stock mask, forward returns.

A pattern script builds a boolean signal on the frame returned by `load_bars`, passes the
signal row positions to `forward_returns`, then calls `summarize`.

Conventions
- Returns use 后复权 prices (close * adj_factor).
- Limit prices use unadjusted prices against the ex-right reference of the previous close.
- Horizons count the stock's own bars (suspended days are skipped).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
STRATEGY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, "database", "processed", "stock_daily.parquet")
RESULTS = os.path.join(STRATEGY_ROOT, "results")

CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")
NEW_STOCK_MIN_BARS = 60
LIMIT_UP_LOOKBACK = 3
PRIOR_YEAR_BARS = 63  # approximately three months of valid daily bars


def board_of(ts_code):
    """Vectorized board label from ts_code prefix."""
    p = ts_code.str[:2]
    return pd.Series(np.select([p.isin(["60", "00"]), p == "30", p == "68", p == "92"],
                               ["主板", "创业板", "科创板", "北交所"], "其他"), index=ts_code.index)


def _cents_half_up(x):
    return np.floor(x * 100 + 0.5 + 1e-6)


def latest_date():
    return pd.read_parquet(DAILY, columns=["date"]).date.max()


def load_bars(start, lookback_days=120):
    """All stocks, sorted by (ts_code, date), from `start - lookback_days` onwards.

    Adds: hfq_open/hfq_close/hfq_high/hfq_low, pct (hfq close-to-close), bar_no (0 = listing day,
    counted on the full history), board, limit_pct, up_c/down_c (limit prices in cents),
    up5_c/down5_c (5% limits for suspected ST on 主板), close_c, is_yizi (O=H=L=C), new_stock.
    """
    df = pd.read_parquet(DAILY)
    df = df.sort_values(["ts_code", "date"], kind="stable").reset_index(drop=True)
    g = df.groupby("ts_code", observed=True)
    df["bar_no"] = g.cumcount()
    df["is_yizi"] = (df.open == df.high) & (df.high == df.low) & (df.low == df.close)
    df["new_stock"] = new_stock_mask(df)

    g = df.groupby("ts_code", observed=True)
    same = g.cumcount() > 0
    df["hfq_open"] = df.open * df.adj_factor
    df["hfq_high"] = df.high * df.adj_factor
    df["hfq_close"] = df.close * df.adj_factor
    df["hfq_low"] = df.low * df.adj_factor
    df["pct"] = (df.hfq_close / g.hfq_close.shift() - 1).where(same)
    df["prior_year_low"] = (df.groupby("ts_code", observed=True)["hfq_low"]
                              .transform(lambda s: s.shift(1).rolling(PRIOR_YEAR_BARS, min_periods=PRIOR_YEAR_BARS).min()))

    # Monthly momentum features, aligned so the current month uses only the
    # signal-day close while the previous month is a completed calendar month.
    df["month"] = df.date.dt.to_period("M")
    monthly = (df.groupby(["ts_code", "month"], observed=True, sort=True)["hfq_close"]
                 .last().rename("month_close").reset_index())
    monthly["prev_month_close"] = monthly.groupby("ts_code", observed=True).month_close.shift(1)
    monthly["prev2_month_close"] = monthly.groupby("ts_code", observed=True).month_close.shift(2)
    monthly["prev_month_return"] = (monthly.prev_month_close / monthly.prev2_month_close - 1)
    monthly["current_month_return_at_close"] = (monthly.month_close / monthly.prev_month_close - 1)
    df = df.merge(monthly[["ts_code", "month", "prev_month_return"]],
                  on=["ts_code", "month"], how="left", sort=False)
    month_prev = monthly[["ts_code", "month", "prev_month_close"]].rename(
        columns={"prev_month_close": "month_start_close"})
    df = df.merge(month_prev, on=["ts_code", "month"], how="left", sort=False)
    df["current_month_return"] = df.hfq_close / df.month_start_close - 1
    ref_prev = (g.close.shift() * g.adj_factor.shift() / df.adj_factor).where(same)

    code = df.ts_code.astype(str)
    df["board"] = board_of(code).astype("category")
    df["limit_pct"] = np.select(
        [df.board == "主板", (df.board == "创业板") & (df.date < CHINEXT_20PCT_FROM),
         df.board.isin(["创业板", "科创板"]), df.board == "北交所"], [0.10, 0.10, 0.20, 0.30], np.nan)
    df["up_c"] = _cents_half_up(ref_prev * (1 + df.limit_pct))
    df["down_c"] = _cents_half_up(ref_prev * (1 - df.limit_pct))
    main = df.board == "主板"
    df["up5_c"] = _cents_half_up(ref_prev * 1.05).where(main)
    df["down5_c"] = _cents_half_up(ref_prev * 0.95).where(main)
    df["close_c"] = np.round(df.close * 100)

    # Exact preceding-20-stock-bar limit-up count, computed on full history before
    # trimming the research window (important for suspensions and sparse trading).
    any_limit_up = ((df.close_c == df.up_c) | (df.close_c == df.up5_c)).fillna(False).values
    codes = df.ts_code.cat.codes.values
    pos = np.arange(len(df))
    cum_limit_up = pd.Series(any_limit_up.astype(np.int8)).groupby(codes).cumsum().values
    prev_idx = pos - 1
    before_idx = pos - (LIMIT_UP_LOOKBACK + 1)
    prev_cum = np.where(
        (prev_idx >= 0) & (codes[np.maximum(prev_idx, 0)] == codes),
        cum_limit_up[np.maximum(prev_idx, 0)], 0,
    )
    before_cum = np.where(
        (before_idx >= 0) & (codes[np.maximum(before_idx, 0)] == codes),
        cum_limit_up[np.maximum(before_idx, 0)], 0,
    )
    df["limit_ups_prev3"] = prev_cum - before_cum

    # Trim only after all features that depend on historical bars have been computed.
    df = df[df.date >= pd.Timestamp(start) - pd.Timedelta(days=lookback_days)].reset_index(drop=True)
    df["ts_code"] = df.ts_code.cat.remove_unused_categories()
    return df


def new_stock_mask(df):
    """True from listing until the first non-一字 bar (开板), and for the first N bars."""
    opened = (~df.is_yizi).astype(int).groupby(df.ts_code, observed=True).cummax().astype(bool)
    return (~opened) | (df.bar_no < NEW_STOCK_MIN_BARS)


def market_index(df):
    """Equal-weight market, daily rebalanced.

    Returns a DataFrame indexed by date with `close_idx` (close-to-close compounded) and
    `open_idx` (index level at the open: previous close_idx * (1 + mean overnight return)).
    """
    g = df.groupby("ts_code", observed=True)
    overnight = (df.hfq_open / g.hfq_close.shift() - 1).where(g.cumcount() > 0)
    m = pd.DataFrame({"r": df.pct, "on": overnight, "date": df.date}).groupby("date").mean()
    close_idx = (1 + m.r.fillna(0)).cumprod()
    open_idx = close_idx.shift().fillna(1.0) * (1 + m.on.fillna(0))
    return pd.DataFrame({"close_idx": close_idx, "open_idx": open_idx})


def forward_returns(df, pos, horizons=(1, 3, 5, 10, 20), mkt=None):
    """Forward returns for signal rows `pos` (integer positions into df).

    entry=close: buy at signal-day close; entry=open: buy at next bar's open.
    Exit at the close of bar t+h. Columns: ret_{entry}_{h}, ex_{entry}_{h} (minus the
    equal-weight market over the same dates), plus entry_blocked (next bar is 一字涨停).
    Rows without h bars ahead (delisted / end of data) are NaN.
    """
    if mkt is None:
        mkt = market_index(df)
    pos = np.asarray(pos)
    n = len(df)
    codes = df.ts_code.cat.codes.values
    hc, ho = df.hfq_close.values, df.hfq_open.values
    dates = df.date.values
    mc = mkt.close_idx.reindex(dates).values
    mo = mkt.open_idx.reindex(dates).values

    def ahead(k):
        j = pos + k
        ok = (j < n) & (codes[np.minimum(j, n - 1)] == codes[pos])
        return np.where(ok, j, -1)

    out = pd.DataFrame(index=pd.RangeIndex(len(pos)))
    nxt = ahead(1)
    has_next = nxt >= 0
    nx = np.where(has_next, nxt, 0)
    # Only an actual one-price limit-up is treated as impossible to fill at the open.
    at_limit_up = ((df.close_c.values[nx] == df.up_c.values[nx])
                   | (df.close_c.values[nx] == df.up5_c.values[nx]))
    out["entry_blocked"] = has_next & df.is_yizi.values[nx] & at_limit_up
    for h in horizons:
        j = ahead(h)
        ok = j >= 0
        jj = np.where(ok, j, 0)
        out[f"ret_close_{h}"] = np.where(ok, hc[jj] / hc[pos] - 1, np.nan)
        out[f"ex_close_{h}"] = out[f"ret_close_{h}"] - np.where(ok, mc[jj] / mc[pos] - 1, np.nan)
        out[f"ret_open_{h}"] = np.where(ok, hc[jj] / ho[nx] - 1, np.nan)
        out[f"ex_open_{h}"] = out[f"ret_open_{h}"] - np.where(ok, mc[jj] / mo[nx] - 1, np.nan)
    return out


def d3_low_bar_close_to_t25_best_close(df, d0_pos, signal_pos, horizon=25):
    """Buy at the close of the lowest-low bar in D1-D3; sell at the best T+1..T+25 close.

    Both selection of the D1-D3 bar and the best exit are retrospective. A
    lowest-low bar before the D1/D2 signal is not a possible strategy entry;
    that event has no valid entry rather than selecting a different bar.
    T+1 excludes the buy bar in accordance with the one-day settlement rule.
    Incomplete D1-D3 and T+25 windows produce no return.
    """
    d0 = np.asarray(d0_pos, dtype=int)
    signals = np.asarray(signal_pos, dtype=int)
    if len(d0) != len(signals):
        raise ValueError("d0_pos and signal_pos must have the same length")
    n = len(df)
    codes = df.ts_code.cat.codes.values
    lc = df.hfq_low.values
    cc = df.hfq_close.values

    start_close = np.full(len(d0), np.nan)
    start_date = np.full(len(d0), np.datetime64("NaT"), dtype="datetime64[ns]")
    start_pos = np.full(len(d0), -1, dtype=int)
    end_close = np.full(len(d0), np.nan)
    end_date = np.full(len(d0), np.datetime64("NaT"), dtype="datetime64[ns]")
    end_pos = np.full(len(d0), -1, dtype=int)
    horizon_date = np.full(len(d0), np.datetime64("NaT"), dtype="datetime64[ns]")

    for i, k in enumerate(d0):
        signal = signals[i]
        if k < 0 or k + 3 >= n or signal <= k or signal > k + 3 or codes[k + 3] != codes[k]:
            continue
        entry_bars = np.arange(k + 1, k + 4)
        if not np.isfinite(lc[entry_bars]).all():
            continue
        buy_idx = entry_bars[np.argmin(lc[entry_bars])]
        if buy_idx < signal:
            continue
        last_idx = buy_idx + horizon
        if last_idx >= n or codes[last_idx] != codes[k] or not np.isfinite(cc[buy_idx]) or cc[buy_idx] <= 0:
            continue
        sell_bars = np.arange(buy_idx + 1, last_idx + 1)
        if not np.isfinite(cc[sell_bars]).all():
            continue
        sell_idx = sell_bars[np.argmax(cc[sell_bars])]
        start_close[i] = cc[buy_idx]
        start_date[i] = df.date.values[buy_idx]
        start_pos[i] = buy_idx
        end_close[i] = cc[sell_idx]
        end_date[i] = df.date.values[sell_idx]
        end_pos[i] = sell_idx
        horizon_date[i] = df.date.values[last_idx]

    ret = end_close / start_close - 1
    return pd.DataFrame({
        "return_start_close": start_close,
        "return_start_date": start_date,
        "return_start_pos": start_pos,
        "return_end_close": end_close,
        "return_end_date": end_date,
        "return_end_pos": end_pos,
        "return_horizon_date": horizon_date,
        "ret_close_to_best_close_25": ret,
    })


def _stats(r, ex):
    r, ex = r.dropna(), ex.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    sd = r.std()
    return {"n": n, "mean": r.mean(), "median": r.median(), "win": (r > 0).mean(), "std": sd,
            "p10": r.quantile(.1), "p25": r.quantile(.25), "p75": r.quantile(.75), "p90": r.quantile(.9),
            "t": r.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
            "ex_mean": ex.mean(), "ex_win": (ex > 0).mean()}


def summarize(ev, by=None, horizons=(1, 3, 5, 10, 20), entries=("close", "open")):
    """Long table of stats per (group, entry, horizon). entry=open drops entry_blocked rows."""
    rows = []
    groups = [("全部", ev)] if by is None else list(ev.groupby(by, observed=True))
    for key, x in groups:
        for e in entries:
            xe = x[~x.entry_blocked] if e == "open" else x
            for h in horizons:
                rows.append({"group_by": by or "全部", "group": key, "entry": e, "horizon": h,
                             **_stats(xe[f"ret_{e}_{h}"], xe[f"ex_{e}_{h}"])})
    return pd.DataFrame(rows)


DIST_BINS = [-np.inf, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]


def distribution(r, bins=DIST_BINS):
    """Histogram table of a return series: count, share, cumulative share per bin."""
    def label(a, b):
        if np.isinf(a):
            return f"< {b:+.0%}"
        if np.isinf(b):
            return f">= {a:+.0%}"
        return f"{a:+.0%} ~ {b:+.0%}"

    r = r.dropna()
    labels = [label(a, b) for a, b in zip(bins[:-1], bins[1:])]
    c = pd.cut(r, bins, labels=labels, right=False).value_counts(sort=False)
    return pd.DataFrame({"count": c, "share": c / c.sum(), "cum_share": c.cumsum() / c.sum()})
