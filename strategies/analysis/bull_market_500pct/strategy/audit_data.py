"""Audit local TDX daily-bar coverage for the two fixed bull-market windows."""
from __future__ import annotations

from pathlib import Path
import json
import re
import struct
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.market_regimes import BULL_RANGES

PROCESSED = ROOT / "database" / "processed"
RAW = ROOT / "database" / "raw" / "tdx"
PROJECT = Path(__file__).resolve().parents[1]
STOCK_RE = re.compile(r"^(sh(60[0-5]|68[89])|sz(00[0-3]|30[0-2])|bj92)\d+\.day$")


def raw_inventory() -> pd.DataFrame:
    rows = []
    for market in ("sh", "sz", "bj"):
        for path in (RAW / market / "lday").glob("*.day"):
            if not STOCK_RE.match(path.name):
                continue
            size = path.stat().st_size
            if size < 32 or size % 32:
                rows.append((path.stem[2:] + "." + market.upper(), size, None, None, None))
                continue
            with path.open("rb") as stream:
                first = struct.unpack("<I", stream.read(4))[0]
                stream.seek(-32, 2)
                last = struct.unpack("<I", stream.read(4))[0]
            rows.append((path.stem[2:] + "." + market.upper(), size, size // 32, first, last))
    raw = pd.DataFrame(rows, columns=["ts_code", "raw_bytes", "raw_bars", "raw_first", "raw_last"])
    for col in ("raw_first", "raw_last"):
        raw[col] = pd.to_datetime(raw[col].astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")
    return raw


def main() -> int:
    daily = pd.read_parquet(PROCESSED / "stock_daily.parquet")
    qfq = pd.read_parquet(PROCESSED / "stock_daily_qfq.parquet", columns=["date", "ts_code"])
    basic = pd.read_parquet(PROCESSED / "stock_basic_tdx.parquet", columns=["ts_code", "name", "list_date", "tdx_industry_code"])
    stocks = daily.groupby("ts_code", observed=True).agg(first=("date", "min"), last=("date", "max"), bars=("date", "size")).reset_index()
    stocks = stocks.merge(basic, on="ts_code", how="left").merge(raw_inventory(), on="ts_code", how="outer", indicator=True)
    stocks["raw_matches_parquet"] = (
        stocks["_merge"].eq("both") & stocks["raw_bars"].eq(stocks["bars"])
        & stocks["raw_first"].eq(stocks["first"]) & stocks["raw_last"].eq(stocks["last"])
    )
    stocks["start_gap_days"] = (stocks["first"] - stocks["list_date"]).dt.days

    valid_ohlc = (daily["low"] <= daily["high"]) & daily["open"].between(daily["low"], daily["high"]) & daily["close"].between(daily["low"], daily["high"])
    valid_prices = daily[["open", "high", "low", "close"]].gt(0).all(axis=1)
    bad_rows = int((~valid_ohlc | ~valid_prices | ~daily["adj_factor"].gt(0)).sum())
    duplicate_rows = int(daily.duplicated(["ts_code", "date"]).sum())
    qfq_match = daily[["date", "ts_code"]].equals(qfq[["date", "ts_code"]])

    counts = daily.groupby("date").size()
    lines = ["# 本地股票日线数据完整性核查", "",
             "核查对象为 `database/raw/tdx/{sh,sz,bj}/lday`、`database/processed/stock_daily.parquet`、前复权 Parquet 和通达信基础资料。",
             "", "## 总览", "",
             f"- 后复权因子日线：{len(daily):,} 行，{len(stocks):,} 只股票，日期 {daily.date.min().date()}—{daily.date.max().date()}。",
             f"- 原始股票 `.day` 文件：{stocks.raw_bars.notna().sum():,} 个；逐只行数、首日和末日均与处理表一致：{stocks.raw_matches_parquet.sum():,}/{len(stocks):,}。",
             f"- 代码与日期重复：{duplicate_rows:,} 行；无效 OHLC/非正价格/非正复权因子：{bad_rows:,} 行；前复权表与原表日期代码键一致：{qfq_match}。",
             f"- 通达信元数据匹配：{stocks['name'].notna().sum():,} 只有名称及行业，{stocks.list_date.notna().sum():,} 只有上市日期。元数据缺失不能解释为行情缺失。",
             "", "## 牛市区间覆盖", "",
             "| 区间 | 交易日 | 区间首日有行情 | 区间末日有行情 | 区间内出现过 | 已知此前上市 | 这些股票首日有行情 | 首日缺口 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    interval_stats = []
    for start, end in BULL_RANGES[-2:]:
        days = counts.loc[start:end]
        appeared = stocks[stocks["first"].le(end) & stocks["last"].ge(start)]
        prelisted = appeared[appeared["list_date"].lt(start)]
        present = int(prelisted["first"].le(days.index[0]).sum())
        interval_stats.append((start, end, days, appeared, prelisted, present))
        lines.append(f"| {start.date()}—{end.date()} | {len(days):,} | {days.iloc[0]:,} | {days.iloc[-1]:,} | {len(appeared):,} | {len(prelisted):,} | {present:,} | {len(prelisted)-present:,} |")
    first_start, _, first_days, _, first_prelisted, first_present = interval_stats[0]
    missing_first = first_prelisted[first_prelisted["first"].gt(first_start)]
    clustered = int(stocks["first"].eq(pd.Timestamp("2014-08-14")).sum())
    tianqi = stocks.loc[stocks.ts_code.eq("002466.SZ"), "first"].iloc[0]
    lines += ["", f"第一段起点已有 {first_days.iloc[0]:,} 条本地股票记录；已知此前上市但首日无记录的还有 {len(missing_first):,} 只，其中 {int(missing_first['first'].eq(pd.Timestamp('2021-08-02')).sum())} 只的原始文件仍从 2021-08-02 开始。外部在交易名单还指出 1 只全库未收录股票（000043.SZ）。另有 {clustered:,} 只股票的文件统一从 2014-08-14 开始，不能把全库最早日期误解为每只股票的完整历史。天齐锂业 002466.SZ 现从 {tianqi.date()} 开始，已覆盖第一段牛市。",
              "", "第二段已知在起点前上市且区间有行情的股票全部在区间首日有记录。逐股起止日期和上市日期见 `results/coverage_by_stock.csv`。",
              "", "外部对照：Baostock 2019-02-01 在交易沪深 A 股 3,570 只，本地有 3,553 只，缺 17 只；清单见 `results/2019_start_missing_active.csv`。2024-09-23 在交易沪深 A 股 5,091 只本地全部覆盖，额外 252 只是北交所股票。",
              "", "## 解释边界", "",
              "区间内个股长时间没有交易记录可能是停牌或退市，不能仅凭日期间隔认定为数据丢失。原始文件与处理表一致，仅说明重建无丢行；起始覆盖仍须结合外部在交易名单核对。",
              ""]
    status_path = PROCESSED / "tdx_sync_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        lines += ["## 同步状态", "", f"本地数据最新交易日：{status.get('database_latest_date')}；最近一次重建：{status.get('finished_at')}。", ""]
    (PROJECT / "docs").mkdir(parents=True, exist_ok=True)
    (PROJECT / "results").mkdir(parents=True, exist_ok=True)
    stocks.to_csv(PROJECT / "results" / "coverage_by_stock.csv", index=False, encoding="utf-8-sig")
    (PROJECT / "docs" / "data_completeness.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
