"""Build A-share daily bars with adjustment factors from TDX .day files + gbbq.

Inputs  (database/raw/tdx):  {sh,sz,bj}/lday/*.day, gbbq
Outputs (database/processed):
    stock_daily.parquet      unadjusted OHLCV + adj_factor (后复权因子, 1.0 at listing)
    stock_daily_qfq.parquet  前复权 OHLC (anchored to each stock's last bar)
    xdxr.parquet             decoded 除权除息 events and how each was applied

Usage:  python scripts/build_adjusted.py
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from pytdx.reader import GbbqReader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "database", "raw", "tdx")
OUT = os.path.join(ROOT, "database", "processed")

DAY_DTYPE = np.dtype([("date", "<u4"), ("open", "<u4"), ("high", "<u4"), ("low", "<u4"),
                      ("close", "<u4"), ("amount", "<f4"), ("vol", "<u4"), ("reserved", "<u4")])

# A-share stocks only. bj legacy codes (43/83/87) are excluded: since the 2025 code switch
# their full history lives under the new 920xxx files and gbbq only carries 920xxx events.
STOCK_RE = re.compile(r"^(sh(60[0-5]|68[89])|sz(00[0-3]|30[0-2])|bj92)\d+$")
GBBQ_MARKET = {0: "sz", 1: "sh", 2: "bj"}


def to_ts_code(key):
    """'sh600000' -> '600000.SH' (tushare style, for later joins)."""
    return f"{key[2:]}.{key[:2].upper()}"


def read_day(path):
    a = np.fromfile(path, dtype=DAY_DTYPE)
    if len(a) == 0:
        return None
    df = pd.DataFrame({"date": pd.to_datetime(a["date"].astype(str), format="%Y%m%d")})
    for c in ("open", "high", "low", "close"):
        df[c] = a[c] / 100.0                      # stocks are stored in 0.01 CNY
    df["vol"] = a["vol"].astype(np.int64)         # shares
    df["amount"] = a["amount"].astype(np.float64) # CNY (float32 at source, ~7 significant digits)
    return df


def load_xdxr():
    g = GbbqReader().get_df(os.path.join(RAW, "gbbq"))
    g = g[g.category == 1]                        # 1 = 除权除息; other categories are share-count changes
    ev = pd.DataFrame({
        "key": g.market.map(GBBQ_MARKET) + g.code,
        "ex_date": pd.to_datetime(g.datetime.astype(str), format="%Y%m%d"),
        "cash_per10": g.hongli_panqianliutong,    # 每10股派现 (CNY)
        "rights_price": g.peigujia_qianzongguben, # 配股价
        "bonus_per10": g.songgu_qianzongguben,    # 每10股送转
        "rights_per10": g.peigu_houzongguben,     # 每10股配股
    })
    return ev.sort_values(["key", "ex_date"]).reset_index(drop=True)


def adj_factor(bars, ev):
    """Return (后复权 factor per bar, per-event log rows).

    Ex-right reference price:
        ref = (C_prev - cash/10 + rights_price*rights/10) / (1 + bonus/10 + rights/10)
    Every bar on/after the ex-date is multiplied by C_prev/ref. Ex-dates falling on a
    non-trading day (suspension/holiday) map to the next trading day.
    """
    dates, close = bars.date.values, bars.close.values
    step = np.ones(len(bars))
    log = []
    for e in ev.itertuples(index=False):
        pos = dates.searchsorted(np.datetime64(e.ex_date))
        row = {"trade_date": pd.NaT, "prev_close": np.nan, "ref_price": np.nan, "ratio": np.nan}
        if pos == 0:
            row["status"] = "before_listing"
        elif pos >= len(dates):
            row["status"] = "after_data_end"
        else:
            c = close[pos - 1] / step[pos]        # chain events that land on the same bar
            ref = (c - e.cash_per10 / 10 + e.rights_price * e.rights_per10 / 10) \
                / (1 + e.bonus_per10 / 10 + e.rights_per10 / 10)
            row.update(trade_date=pd.Timestamp(dates[pos]), prev_close=c, ref_price=ref)
            if ref <= 0 or c <= 0:
                row["status"] = "invalid"
            else:
                step[pos] *= c / ref
                row.update(ratio=c / ref, status="applied")
        log.append(row)
    return np.cumprod(step), log


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    ev_all = load_xdxr()
    ev_by_key = dict(tuple(ev_all.groupby("key")))

    files = [(mk, f) for mk in ("sh", "sz", "bj")
             for f in sorted(os.listdir(os.path.join(RAW, mk, "lday")))
             if STOCK_RE.match(f[:-4])]
    parts, logs, empty = [], [], []
    for mk, f in tqdm(files, desc="stocks", ncols=80, mininterval=5):
        key = f[:-4]
        bars = read_day(os.path.join(RAW, mk, "lday", f))
        if bars is None:
            empty.append(key)
            continue
        ev = ev_by_key.get(key)
        if ev is not None:
            bars["adj_factor"], log = adj_factor(bars, ev)
            logs.append(pd.concat([ev.reset_index(drop=True), pd.DataFrame(log)], axis=1))
        else:
            bars["adj_factor"] = 1.0
        bars.insert(1, "ts_code", to_ts_code(key))
        parts.append(bars)

    daily = pd.concat(parts, ignore_index=True)
    daily["ts_code"] = daily.ts_code.astype("category")
    daily.to_parquet(os.path.join(OUT, "stock_daily.parquet"), index=False)

    last = daily.groupby("ts_code", observed=True).adj_factor.transform("last")
    qfq = daily[["date", "ts_code"]].copy()
    for c in ("open", "high", "low", "close"):
        qfq[c] = (daily[c] * daily.adj_factor / last).round(4)  # 4dp keeps file size sane
    qfq[["vol", "amount"]] = daily[["vol", "amount"]]
    qfq.to_parquet(os.path.join(OUT, "stock_daily_qfq.parquet"), index=False)

    xdxr = pd.concat(logs, ignore_index=True)
    xdxr.insert(0, "ts_code", xdxr.pop("key").map(to_ts_code))
    xdxr.to_parquet(os.path.join(OUT, "xdxr.parquet"), index=False)

    print(f"stocks={daily.ts_code.nunique()} rows={len(daily):,} "
          f"dates={daily.date.min().date()}..{daily.date.max().date()} empty_files={empty}")
    print("xdxr events:", xdxr.status.value_counts().to_dict())
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
