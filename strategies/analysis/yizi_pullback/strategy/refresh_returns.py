"""Refresh path-potential returns from an existing yizi event list without loading all bars."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import event_study as es


def market_close_index():
    sums = pd.Series(dtype=float)
    counts = pd.Series(dtype=int)
    previous_code = None
    previous_close = np.nan
    parquet = pq.ParquetFile(es.DAILY)
    for batch in parquet.iter_batches(batch_size=131072, columns=["date", "ts_code", "close", "adj_factor"]):
        bars = batch.to_pandas()
        adjusted = bars.close.to_numpy() * bars.adj_factor.to_numpy()
        codes = bars.ts_code.to_numpy()
        same = codes[1:] == codes[:-1]
        returns = np.full(len(bars), np.nan)
        returns[1:][same] = adjusted[1:][same] / adjusted[:-1][same] - 1
        if codes[0] == previous_code:
            returns[0] = adjusted[0] / previous_close - 1
        previous_code, previous_close = codes[-1], adjusted[-1]
        by_date = pd.DataFrame({"date": bars.date, "return": returns}).groupby("date")["return"]
        sums = sums.add(by_date.sum(), fill_value=0)
        counts = counts.add(by_date.count(), fill_value=0)
    return (1 + (sums / counts).sort_index().fillna(0)).cumprod()


def refresh():
    path = Path(es.RESULTS) / "events.csv"
    events = pd.read_csv(path, parse_dates=["signal_date", "yizi_date"])
    codes = events.ts_code.astype(str).unique().tolist()
    bars = pd.read_parquet(es.DAILY, columns=["date", "ts_code", "close", "high", "low", "adj_factor"],
                           filters=[("ts_code", "in", codes)])
    bars = bars.sort_values(["ts_code", "date"], kind="stable").reset_index(drop=True)
    bars["ts_code"] = bars.ts_code.astype("category")
    for price in ("close", "high", "low"):
        bars[f"hfq_{price}"] = bars[price] * bars.adj_factor
    positions = pd.Series(np.arange(len(bars)), index=pd.MultiIndex.from_frame(bars[["ts_code", "date"]]))
    d0 = positions.reindex(pd.MultiIndex.from_frame(events[["ts_code", "yizi_date"]].rename(
        columns={"yizi_date": "date"}))).to_numpy()
    signal = positions.reindex(pd.MultiIndex.from_frame(events[["ts_code", "signal_date"]].rename(
        columns={"signal_date": "date"}))).to_numpy()
    if np.isnan(d0).any() or np.isnan(signal).any():
        raise ValueError("An event date is missing from daily prices")
    result = es.d3_low_bar_close_to_t25_best_close(bars, d0.astype(int), signal.astype(int))
    market = market_close_index()
    entry = market.reindex(result.return_start_date).to_numpy()
    exit_price = market.reindex(result.return_end_date).to_numpy()
    result["ex_close_to_best_close_25"] = result.ret_close_to_best_close_25 - (exit_price / entry - 1)
    events = events.drop(columns=[column for column in events if column.startswith(("return_", "ret_low_to_high_", "ex_low_to_high_", "ret_close_to_best_close_", "ex_close_to_best_close_"))])
    for column in result:
        events[column] = result[column].to_numpy()
    events.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.4f")

    summary_path = Path(es.RESULTS) / "summary.csv"
    summary = pd.read_csv(summary_path)
    mask = summary.entry.isin(["first_downtrend_low_to_later_high", "d1_d3_low_bar_close_to_best_close"])
    summary.loc[mask, "entry"] = "d1_d3_low_bar_close_to_best_close"
    summary.loc[mask, "horizon"] = 25
    path_stats = es._stats(events.ret_close_to_best_close_25, events.ex_close_to_best_close_25)
    for column, value in path_stats.items():
        summary.loc[mask, column] = value
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"有效路径样本 {path_stats['n']:,}；均值 {path_stats['mean']:+.2%}；中位数 {path_stats['median']:+.2%}")


if __name__ == "__main__":
    refresh()
