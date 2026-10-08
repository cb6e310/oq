"""Shared data loading and forward-return helpers for the ZT pullback study."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
STRATEGY_ROOT = ROOT / "strategies" / "analysis" / "zt_pullback"
DAILY = ROOT / "database" / "processed" / "stock_daily.parquet"
RESULTS = STRATEGY_ROOT / "results"


def latest_date():
    return pd.read_parquet(DAILY, columns=["date"]).date.max()


def load_bars(start, lookback_days=30):
    df = pd.read_parquet(DAILY).sort_values(["ts_code", "date"], kind="stable").reset_index(drop=True)
    code = df.ts_code.astype(str)
    df["board"] = np.select(
        [code.str[:2].isin(["60", "00"]), code.str[:2].eq("30"), code.str[:2].eq("68"), code.str[:2].eq("92")],
        ["主板", "创业板", "科创板", "北交所"], "其他",
    )
    g = df.groupby("ts_code", observed=True, sort=False)
    df["prev_close"] = g.close.shift()
    df["pct_raw"] = df.close / df.prev_close - 1
    df["hfq_open"] = df.open * df.adj_factor
    df["hfq_close"] = df.close * df.adj_factor
    df["hfq_high"] = df.high * df.adj_factor
    df["hfq_low"] = df.low * df.adj_factor
    df["pct"] = df.hfq_close / g.hfq_close.shift() - 1
    df = df[df.date >= pd.Timestamp(start) - pd.Timedelta(days=lookback_days)].reset_index(drop=True)
    df["ts_code"] = df.ts_code.cat.remove_unused_categories()
    return df


def market_index(df):
    g = df.groupby("ts_code", observed=True, sort=False)
    returns = pd.DataFrame({"r": df.pct, "date": df.date}).groupby("date").r.mean()
    return pd.DataFrame({"close_idx": (1 + returns.fillna(0)).cumprod()})


def forward_returns(df, positions, horizons=(1, 3, 5, 10, 20), mkt=None):
    if mkt is None:
        mkt = market_index(df)
    positions = np.asarray(positions, dtype=int)
    codes = df.ts_code.cat.codes.to_numpy()
    close = df.hfq_close.to_numpy()
    dates = df.date.to_numpy()
    market = mkt.close_idx.reindex(dates).to_numpy()
    n = len(df)

    def ahead(pos, bars):
        target = pos + bars
        ok = target < n
        same = np.zeros(len(pos), dtype=bool)
        same[ok] = codes[target[ok]] == codes[pos[ok]]
        return np.where(ok & same, target, -1)

    out = pd.DataFrame(index=range(len(positions)))
    for horizon in horizons:
        target = ahead(positions, horizon)
        ok = target >= 0
        safe = np.maximum(target, 0)
        out[f"ret_close_{horizon}"] = np.where(ok, close[safe] / close[positions] - 1, np.nan)
        out[f"ex_close_{horizon}"] = np.where(ok, close[safe] / close[positions] - 1, np.nan) - np.where(
            ok, market[safe] / market[positions] - 1, np.nan
        )
    out["entry_blocked"] = False
    return out


def signal_close_to_t25_best_close(df, positions, horizon=25):
    positions = np.asarray(positions, dtype=int)
    codes = df.ts_code.cat.codes.to_numpy()
    prices = df.hfq_close.to_numpy()
    dates = df.date.to_numpy()
    starts = np.full(len(positions), np.nan)
    start_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
    start_pos = np.full(len(positions), -1, dtype=int)
    ends = np.full(len(positions), np.nan)
    end_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
    end_pos = np.full(len(positions), -1, dtype=int)
    horizon_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
    n = len(df)
    for i, pos in enumerate(positions):
        last = pos + horizon
        if last >= n or codes[last] != codes[pos]:
            continue
        future = np.arange(pos + 1, last + 1)
        if not np.isfinite(prices[future]).all():
            continue
        best = future[np.argmax(prices[future])]
        starts[i] = prices[pos]
        start_dates[i] = dates[pos]
        start_pos[i] = pos
        ends[i] = prices[best]
        end_dates[i] = dates[best]
        end_pos[i] = best
        horizon_dates[i] = dates[last]
    return pd.DataFrame({
        "return_start_close": starts,
        "return_start_date": start_dates,
        "return_start_pos": start_pos,
        "return_end_close": ends,
        "return_end_date": end_dates,
        "return_end_pos": end_pos,
        "return_horizon_date": horizon_dates,
        "ret_close_to_best_close_25": ends / starts - 1,
    })


def stats(returns, excess):
    values = returns.dropna()
    excess = excess.reindex(values.index).dropna()
    if values.empty:
        return {"n": 0}
    sd = values.std()
    return {
        "n": len(values), "mean": values.mean(), "median": values.median(), "win": (values > 0).mean(),
        "std": sd, "p10": values.quantile(.1), "p25": values.quantile(.25), "p75": values.quantile(.75),
        "p90": values.quantile(.9), "t": values.mean() / (sd / np.sqrt(len(values))) if len(values) > 1 and sd > 0 else np.nan,
        "ex_mean": excess.mean(), "ex_win": (excess > 0).mean(),
    }


def summarize(events, by=None, horizons=(1, 3, 5, 10, 20)):
    groups = [("全部", events)] if by is None else list(events.groupby(by, observed=True))
    rows = []
    for key, group in groups:
        for horizon in horizons:
            rows.append({"group_by": by or "全部", "group": key, "entry": "close", "horizon": horizon,
                         **stats(group[f"ret_close_{horizon}"], group[f"ex_close_{horizon}"])})
    return pd.DataFrame(rows)
