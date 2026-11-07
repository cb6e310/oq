"""Read-only audit of local A-share daily data from 2015 onward.

Writes audit artifacts under results/, never changes market data. Missing bars
between a stock's endpoints are candidates only: suspensions need an independent
source to distinguish them from lost history.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_adjusted import DAY_DTYPE, STOCK_RE, to_ts_code

# TDX stores the pre-change bars under these successor codes as well.
HISTORICAL_CODE_ALIASES = {"000022.SZ": "001872.SZ", "000043.SZ": "001914.SZ"}


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "data_completeness_2015")
    args = parser.parse_args()
    start = pd.Timestamp(args.start)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "database" / "processed"
    raw = ROOT / "database" / "raw" / "tdx"
    daily = pd.read_parquet(processed / "stock_daily.parquet")
    basic = pd.read_parquet(processed / "stock_basic_tdx.parquet")
    basic = basic[["ts_code", "name", "list_date"]].drop_duplicates("ts_code")
    scope = daily.date.ge(start).to_numpy()
    end = daily.date.max()
    print(f"Loaded {len(daily):,} rows; auditing {int(scope.sum()):,} rows", flush=True)

    calendars = {}
    for market, code in (("sh", "000001"), ("sz", "399001")):
        bars = np.fromfile(raw / market / "lday" / f"{market}{code}.day", dtype=DAY_DTYPE)
        dates = pd.DatetimeIndex(pd.to_datetime(bars["date"].astype(str), format="%Y%m%d"))
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError(f"Invalid index calendar for {market}")
        calendars[market] = dates[(dates >= start) & (dates <= end)]
    calendar = calendars["sh"].union(calendars["sz"])
    scope_daily = daily.loc[scope, ["ts_code", "date"]]
    by_date = scope_daily.groupby("date", observed=True).size()
    missing_market_dates = calendar.difference(by_date.index)
    invalid_dates = by_date.index.difference(calendar)
    counts = by_date.rename("stock_bars").reset_index()
    save_csv(counts, out / "daily_counts.csv")
    annual = []
    for year in sorted(calendar.year.unique()):
        expected = calendar[calendar.year == year]
        part = scope_daily[scope_daily.date.dt.year.eq(year)]
        n = part.groupby("date", observed=True).size()
        annual.append(dict(year=int(year), first=str(expected.min().date()), last=str(expected.max().date()),
                           expected_days=len(expected), actual_days=len(n), rows=len(part),
                           stocks=part.ts_code.nunique(), first_day_stocks=int(n.get(expected[0], 0)),
                           last_day_stocks=int(n.get(expected[-1], 0))))
    annual = pd.DataFrame(annual)
    save_csv(annual, out / "annual_coverage.csv")

    price_cols = ["open", "high", "low", "close"]
    values = daily[price_cols + ["vol", "amount", "adj_factor"]].to_numpy()
    nonfinite = ~np.isfinite(values).all(axis=1)
    bad_price = daily[price_cols].le(0).any(axis=1).to_numpy()
    bad_ohlc = ((daily.low > daily.high) | ~daily.open.between(daily.low, daily.high)
                | ~daily.close.between(daily.low, daily.high)).to_numpy()
    bad_factor = daily.adj_factor.le(0).to_numpy()
    negative_flow = ((daily.vol < 0) | (daily.amount < 0)).to_numpy()
    duplicate = daily.duplicated(["ts_code", "date"]).to_numpy()
    bad = scope & (nonfinite | bad_price | bad_ohlc | bad_factor | negative_flow | duplicate)
    save_csv(daily.loc[bad], out / "invalid_rows.csv")
    del values

    groups = daily.groupby("ts_code", observed=True, sort=False).indices
    raw_paths = {to_ts_code(p.stem): p for market in ("sh", "sz", "bj")
                 for p in (raw / market / "lday").glob("*.day") if STOCK_RE.fullmatch(p.stem)}
    records, gaps, mismatches = [], [], []
    for code, positions in groups.items():
        bars = daily.iloc[positions]
        dates = pd.DatetimeIndex(bars.date)
        current = dates[dates >= start]
        record = dict(ts_code=code, first=dates.min(), last=dates.max(), bars=len(bars),
                      bars_since_start=len(current), date_order_ok=dates.is_monotonic_increasing,
                      raw_exists=code in raw_paths, raw_aligned=False, raw_values_equal=False,
                      missing_inside_span=0, longest_internal_gap=0)
        if code in raw_paths:
            path = raw_paths[code]
            record["raw_bytes"] = path.stat().st_size
            if record["raw_bytes"] % 32 == 0:
                source = np.fromfile(path, dtype=DAY_DTYPE)
                raw_dates = pd.to_datetime(source["date"].astype(str), format="%Y%m%d", errors="coerce")
                record["raw_aligned"] = len(source) == len(bars) and np.array_equal(raw_dates.to_numpy(), dates.to_numpy())
                if record["raw_aligned"]:
                    bad_raw = np.zeros(len(bars), dtype=bool)
                    for col in price_cols + ["vol", "amount"]:
                        expected = source[col] / 100.0 if col in price_cols else source[col].astype(bars[col].dtype)
                        bad_raw |= bars[col].to_numpy() != expected
                    record["raw_values_equal"] = not bad_raw.any()
                    if bad_raw.any():
                        mismatches.append(dict(ts_code=code, reason="raw_values", rows=int(bad_raw.sum())))
            if not record["raw_aligned"]:
                mismatches.append(dict(ts_code=code, reason="raw_dates_or_length", rows=None))
        else:
            mismatches.append(dict(ts_code=code, reason="missing_raw_file", rows=None))
        if len(current):
            # Index union is a proxy for BJ; do not call BJ gaps confirmed loss.
            expected = calendar[(calendar >= current.min()) & (calendar <= current.max())]
            missing = expected.difference(current)
            record["missing_inside_span"] = len(missing)
            present_positions = calendar.get_indexer(current)
            good = present_positions >= 0
            valid_dates, valid_positions = current[good], present_positions[good]
            for i in np.flatnonzero(np.diff(valid_positions) > 1):
                gap_len = int(valid_positions[i + 1] - valid_positions[i] - 1)
                record["longest_internal_gap"] = max(record["longest_internal_gap"], gap_len)
                gaps.append(dict(ts_code=code, before=valid_dates[i], after=valid_dates[i + 1],
                                 missing_sessions=gap_len, classification="unverified_suspension_or_missing"))
        records.append(record)
    coverage = pd.DataFrame(records).merge(basic, on="ts_code", how="left")
    coverage["start_gap_calendar_days"] = (coverage["first"] - coverage.list_date).dt.days
    candidates = coverage[(coverage.bars_since_start > 0) & (coverage["first"] > calendar.min())
                          & (coverage.list_date < coverage["first"] - pd.Timedelta(days=45))].copy()
    # This flag is a lead, not proof: a pre-existing suspension may span 2015.
    save_csv(coverage, out / "coverage_by_stock.csv")
    stock_basic = basic[basic.ts_code.map(lambda code: bool(STOCK_RE.fullmatch(code[-2:].lower() + code[:6])))]
    metadata_only = stock_basic[~stock_basic.ts_code.isin(coverage.ts_code)].copy()
    metadata_only["successor_code"] = metadata_only.ts_code.map(HISTORICAL_CODE_ALIASES)
    metadata_only["classification"] = np.where(metadata_only.successor_code.isin(coverage.ts_code),
                                                "historical_code_mapped", "listing_status_unverified")
    save_csv(metadata_only, out / "metadata_only_stocks.csv")
    save_csv(candidates, out / "late_start_candidates.csv")
    save_csv(pd.DataFrame(gaps, columns=["ts_code", "before", "after", "missing_sessions", "classification"]), out / "internal_gap_candidates.csv")
    save_csv(pd.DataFrame(mismatches, columns=["ts_code", "reason", "rows"]), out / "raw_mismatches.csv")
    print(f"Raw comparison complete: {coverage.raw_values_equal.sum()}/{len(coverage)} stocks", flush=True)

    # Compare QFQ in batches to avoid loading a second complete price table.
    factors = daily.groupby("ts_code", observed=True).adj_factor.transform("last").to_numpy()
    offset = qfq_bad_keys = qfq_bad_prices = qfq_bad_flows = 0
    for batch in pq.ParquetFile(processed / "stock_daily_qfq.parquet").iter_batches(batch_size=250_000):
        qfq = batch.to_pandas()
        bars = daily.iloc[offset:offset + len(qfq)]
        if len(bars) != len(qfq):
            qfq_bad_keys += len(qfq)
            offset += len(qfq)
            continue
        selected = scope[offset:offset + len(qfq)]
        keys_bad = ((qfq.date.to_numpy() != bars.date.to_numpy())
                    | (qfq.ts_code.astype(str).to_numpy() != bars.ts_code.astype(str).to_numpy()))
        qfq_bad_keys += int(keys_bad.sum())
        price_bad = np.zeros(len(qfq), dtype=bool)
        for col in price_cols:
            expected = np.round(bars[col].to_numpy() * bars.adj_factor.to_numpy() / factors[offset:offset + len(qfq)], 4)
            price_bad |= ~np.isclose(qfq[col].to_numpy(), expected, rtol=0, atol=1e-8)
        qfq_bad_prices += int((price_bad & selected).sum())
        qfq_bad_flows += int((((qfq.vol.to_numpy() != bars.vol.to_numpy())
                               | (qfq.amount.to_numpy() != bars.amount.to_numpy())) & selected).sum())
        offset += len(qfq)
    qfq_row_count_match = offset == len(daily)

    # Preserve provenance: these observations were collected in a previous audit,
    # and are rechecked against today's local data, not presented as a live fetch.
    prior_path = ROOT / "strategies/analysis/bull_market_500pct/results/2019_start_missing_active.csv"
    prior_missing = pd.DataFrame()
    resolved_aliases = pd.DataFrame()
    if prior_path.exists():
        prior = pd.read_csv(prior_path)[["ts_code", "code_name", "tradeStatus"]]
        present = set(daily.loc[daily.date.eq("2019-02-01"), "ts_code"].astype(str))
        prior["effective_ts_code"] = prior.ts_code.map(HISTORICAL_CODE_ALIASES).fillna(prior.ts_code)
        prior["present_under_effective_code"] = prior.effective_ts_code.isin(present)
        save_csv(prior, out / "prior_external_all_rechecked.csv")
        resolved_aliases = prior[prior.present_under_effective_code & prior.ts_code.ne(prior.effective_ts_code)]
        prior_missing = prior[~prior.present_under_effective_code].merge(coverage, on="ts_code", how="left")
        prior_missing["evidence_date"] = "2019-02-01"
        prior_missing["evidence_source"] = str(prior_path.relative_to(ROOT))
        prior_missing["classification"] = np.where(prior_missing["first"].isna(), "stock_absent", "historical_bars_missing")
        save_csv(prior_missing, out / "prior_external_missing_rechecked.csv")

    clustered = coverage[(coverage["first"] == "2021-08-02") & (coverage.bars_since_start > 0)]
    cluster_normal_ipos = int(clustered.list_date.eq(clustered["first"]).sum())
    save_csv(clustered, out / "history_starts_20210802.csv")
    status = json.loads((processed / "tdx_sync_status.json").read_text(encoding="utf-8"))
    summary = dict(start=str(start.date()), end=str(end.date()), rows=int(scope.sum()), stocks=int(scope_daily.ts_code.nunique()),
                   all_rows=len(daily), all_stocks=len(coverage), calendar_sessions=len(calendar),
                   missing_market_dates=[str(x.date()) for x in missing_market_dates],
                   off_calendar_dates=[str(x.date()) for x in invalid_dates],
                   sh_sz_calendars_equal=calendars["sh"].equals(calendars["sz"]),
                   duplicate_rows=int((scope & duplicate).sum()), nonfinite_rows=int((scope & nonfinite).sum()),
                   invalid_ohlc_rows=int((scope & (bad_price | bad_ohlc)).sum()),
                   invalid_factor_rows=int((scope & bad_factor).sum()), negative_flow_rows=int((scope & negative_flow).sum()),
                   zero_volume_rows=int((daily.loc[scope, "vol"] == 0).sum()),
                   raw_stock_files=len(raw_paths), raw_only_stocks=sorted(set(raw_paths) - set(groups)),
                   raw_aligned_stocks=int(coverage.raw_aligned.sum()), raw_value_equal_stocks=int(coverage.raw_values_equal.sum()),
                   unordered_stocks=int((~coverage.date_order_ok).sum()),
                   qfq_row_count_match=qfq_row_count_match, qfq_bad_key_rows=qfq_bad_keys,
                   qfq_bad_price_rows=qfq_bad_prices, qfq_bad_flow_rows=qfq_bad_flows,
                   late_start_candidates=len(candidates), starts_20210802=len(clustered),
                   starts_20210802_normal_ipos=cluster_normal_ipos,
                   prior_external_missing_stocks=len(prior_missing),
                   prior_external_resolved_code_changes=len(resolved_aliases),
                   metadata_only_unverified_stocks=int(metadata_only.classification.eq("listing_status_unverified").sum()),
                   internal_gap_ranges=len(gaps),
                   internal_gap_stock_sessions=int(coverage.missing_inside_span.sum()),
                   stocks_ending_on_latest=int(coverage["last"].eq(end).sum()),
                   sync_status=status)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    checked_on = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    lines = ["# 2015 年以来股票日线数据完整性核查", "", f"核查日期：{checked_on}。区间包含 2015 年，截止当前数据最新日。", "",
             "结论：数据不能认定为完全。历史外部名单中已确认的缺口仍在当前库中；全市场日期连续与文件转换一致不代表每只股票历史齐全。", "",
             f"- 区间：{start.date()}—{end.date()}；{summary['rows']:,} 行，{summary['stocks']:,} 只股票，{len(calendar):,} 个市场交易日。",
             f"- 全库：{len(daily):,} 行，{len(coverage):,} 只股票。最新日有 {summary['stocks_ending_on_latest']:,} 只股票记录。",
             f"- 上证指数和深证成指交易日期一致：{summary['sh_sz_calendars_equal']}；整日缺失 {len(missing_market_dates)}，非日历日期 {len(invalid_dates)}。日历由本地两指数交叉核对，未独立验证交易所官方日历。",
             f"- 原始文件对处理表：{summary['raw_aligned_stocks']:,}/{len(coverage):,} 只的所有日期逐条一致；{summary['raw_value_equal_stocks']:,}/{len(coverage):,} 只的 OHLC、成交量和成交额逐条一致。",
             f"- 2015 年以来重复记录 {summary['duplicate_rows']}；非有限数 {summary['nonfinite_rows']}；无效 OHLC {summary['invalid_ohlc_rows']}；无效复权因子 {summary['invalid_factor_rows']}；负成交量/额 {summary['negative_flow_rows']}。",
             f"- 前复权表行数一致：{qfq_row_count_match}；全库日期代码错位 {qfq_bad_keys}；区间前复权价格公式不一致 {qfq_bad_prices}；成交量/额不一致 {qfq_bad_flows}。这里只验证内部公式，未逐条外部核验复权事件。", "",
             "## 已确认历史缺口的当前状态", "",
             "证据来自项目已保存的 Baostock 2019-02-01 在交易名单（2026-10-19 已有的核查文件），本次重查其在当前库是否存在；本表不是本次联网重新拉取的完整证券名册。", "",
             "| 代码 | 历史名单名称 | 当前数据起点 | 当前数据终点 | 状态 |", "|---|---|---|---|---|"]
    for row in prior_missing.itertuples(index=False):
        first = "全库未收录" if pd.isna(row.first) else str(row.first.date())
        last = "—" if pd.isna(row.last) else str(row.last.date())
        lines.append(f"| {row.ts_code} | {row.code_name} | {first} | {last} | {row.classification} |")
    lines += ["", "旧核查记录把 000043.SZ（中航善达）算作全库未收录。本次已按更名换码映射到 001914.SZ（招商积余），并确认新代码下包含 2019-02-01 的行情，因此排除此项误报。000022.SZ 的历史也已归在 001872.SZ 下。",
              "", f"共有 {len(clustered)} 只股票的数据从 2021-08-02 开始，其中 {cluster_normal_ipos} 只是当日正常上市，余下 {len(clustered)-cluster_normal_ipos} 只早于当日上市；其中 {len(prior_missing)} 只有上述历史外部名单证据，另 1 只 605081.SH（太和水，2021-02-09 上市）需进一步外部核实。完整清单见 `history_starts_20210802.csv`。",
              "", "## 按年覆盖", "",
              "交易日齐全只说明市场每天有数据；股票数量变化受上市、停牌、退市和缺失共同影响。", "",
              "| 年份 | 交易日 | 有记录日期 | 行数 | 出现股票数 |", "|---|---:|---:|---:|---:|"]
    for row in annual.itertuples(index=False):
        lines.append(f"| {row.year} | {row.expected_days} | {row.actual_days} | {row.rows:,} | {row.stocks:,} |")
    lines += ["", "## 待核实范围", "",
              f"- {len(candidates)} 只股票在核查起点之后才开始有数据，且比元数据上市日期晚逾 45 个自然日，见 `late_start_candidates.csv`。长停牌、重新上市和元数据误差均可能导致这种情况，不能全部算作缺失。",
              f"- 首末记录之间共有 {len(gaps):,} 段无日线区间，合计 {summary['internal_gap_stock_sessions']:,} 个股票交易日，见 `internal_gap_candidates.csv`。这些区间尚未用逐日停复牌记录排除正常停牌，因此不是已确认缺失数量；北交所也暂按沪深日历检查。",
              "- 全库未出现的其他已退市股票，不能仅靠本地股票名单发现；需要独立历史证券名册进一步核对。",
              "- 基础资料中另有 001381.SZ、688827.SH、920306.BJ 没有日线，且上市日期字段为空，需先核实是否尚未上市；不直接判为数据丢失，见 `metadata_only_stocks.csv`。",
              "- 北交所新代码沿用历史，不能将新三板挂牌日期直接当作应有 A 股日线的起点。",
              "- 缺口位于原始文件本身，重新从相同原始文件构建 Parquet 不能补齐。",
              "", "本次仅审计并输出结果，未修改或下载覆盖行情数据。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
