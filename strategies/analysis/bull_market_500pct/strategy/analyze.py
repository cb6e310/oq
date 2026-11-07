"""统计固定牛市区间内最大涨幅超过 500% 的股票及行业。

同时提供两个口径：首日后复权收盘至区间最高价，以及区间低点至其后最高价。
股票不要求在区间开始前上市；区间内新上市股票从其第一个有效交易日计算。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.market_regimes import BULL_RANGES

DAILY = ROOT / "database" / "processed" / "stock_daily.parquet"
BASIC = ROOT / "database" / "processed" / "stock_basic_tdx.parquet"
RESULTS = Path(__file__).resolve().parents[1] / "results"

REGIMES = {
    f"牛市{i}_{start.year}-{end.year}": (start, end)
    for i, (start, end) in enumerate(BULL_RANGES[-2:], start=1)
}

BOARD_ORDER = ("主板", "创业板", "科创板", "北交所", "其他")


def stock_board(ts_code: str) -> str:
    """按证券代码所属交易板块分类，保留北交所和未识别代码。"""
    symbol, exchange = ts_code.split(".")
    if exchange == "SZ" and symbol.startswith(("300", "301", "302")):
        return "创业板"
    if exchange == "SH" and symbol.startswith(("688", "689")):
        return "科创板"
    if exchange == "BJ" and symbol.startswith("92"):
        return "北交所"
    if (exchange == "SH" and symbol.startswith(("600", "601", "603", "605"))) or (
        exchange == "SZ" and symbol.startswith(("000", "001", "002", "003"))
    ):
        return "主板"
    return "其他"


def fetch_baostock_industry(date: pd.Timestamp) -> pd.DataFrame:
    """读取 Baostock 指定日期行业快照；失败时返回空表。"""
    try:
        import baostock as bs
    except ImportError:
        return pd.DataFrame()
    login = bs.login()
    if login.error_code != "0":
        return pd.DataFrame()
    try:
        query = bs.query_stock_industry(date=date.date().isoformat())
        rows = []
        while query.next():
            rows.append(query.get_row_data())
        if query.error_code != "0" or not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows, columns=query.fields)
    finally:
        bs.logout()
    frame = frame.rename(columns={"code": "baostock_code", "code_name": "name", "industry": "industry"})
    frame["ts_code"] = frame["baostock_code"].str[-6:] + "." + frame["baostock_code"].str[:2].str.upper()
    return frame[["ts_code", "name", "industry", "industryClassification", "updateDate"]]


def interval_returns(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = ["date", "ts_code", "close", "high", "low", "adj_factor"]
    daily = pd.read_parquet(DAILY, columns=columns)
    daily = daily[daily.date.between(start, end)].dropna(subset=["close", "high", "low", "adj_factor"])
    daily["hfq_close"] = daily["close"] * daily["adj_factor"]
    daily["hfq_high"] = daily["high"] * daily["adj_factor"]
    daily["hfq_low"] = daily["low"] * daily["adj_factor"]
    daily = daily[(daily["hfq_close"] > 0) & (daily["hfq_high"] > 0) & (daily["hfq_low"] > 0)]
    daily = daily.sort_values(["ts_code", "date"])
    first = daily.groupby("ts_code", observed=True).first()[["date", "hfq_close"]]
    first = first.rename(columns={"date": "first_trade_date", "hfq_close": "start_hfq_close"})
    max_pos = daily.groupby("ts_code", observed=True)["hfq_high"].idxmax()
    peak = daily.loc[max_pos, ["ts_code", "date", "hfq_high"]].set_index("ts_code")
    peak = peak.rename(columns={"date": "peak_date", "hfq_high": "peak_hfq_high"})
    groups = daily.groupby("ts_code", observed=True)
    # 仅使用此前交易日的低点；日线无法确定同日高低价的盘中先后。
    daily["prior_low"] = groups["hfq_low"].cummin()
    new_low = daily["hfq_low"].eq(daily["prior_low"])
    daily["prior_low_date"] = daily["date"].where(new_low)
    daily["prior_low_date"] = daily.groupby("ts_code", observed=True)["prior_low_date"].ffill()
    daily["prior_low"] = daily.groupby("ts_code", observed=True)["prior_low"].shift()
    daily["prior_low_date"] = daily.groupby("ts_code", observed=True)["prior_low_date"].shift()
    daily["runup_pct"] = (daily["hfq_high"] / daily["prior_low"] - 1) * 100
    best_pos = daily.dropna(subset=["runup_pct"]).groupby("ts_code", observed=True)["runup_pct"].idxmax()
    best = daily.loc[best_pos, ["ts_code", "date", "prior_low_date", "prior_low", "hfq_high", "runup_pct"]].set_index("ts_code")
    best = best.rename(columns={"date": "runup_peak_date", "prior_low_date": "runup_low_date", "prior_low": "runup_prior_low", "hfq_high": "runup_peak_high", "runup_pct": "max_runup_pct"})
    out = first.join(peak, how="inner").join(best, how="left").reset_index()
    out["start_to_peak_pct"] = (out["peak_hfq_high"] / out["start_hfq_close"] - 1) * 100
    out["max_gain_pct"] = out["max_runup_pct"]
    out["interval_start"] = start.date().isoformat()
    out["interval_end"] = end.date().isoformat()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-baostock", action="store_true", help="不请求 Baostock，优先复用已保存的行业快照")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    basic = pd.read_parquet(BASIC) if BASIC.exists() else pd.DataFrame()
    if not basic.empty:
        basic_cols = [c for c in ["ts_code", "name", "tdx_industry_code", "tdx_industry_no", "list_date"] if c in basic]
        basic = basic[basic_cols].drop_duplicates("ts_code")

    summary_rows = []
    industry_rows = []
    board_rows = []
    board_200_rows = []
    for regime, (start, end) in REGIMES.items():
        result = interval_returns(start, end)
        first_dates = result.set_index("ts_code")["first_trade_date"]
        listed = basic.set_index("ts_code")["list_date"] if "list_date" in basic else pd.Series(dtype="datetime64[ns]")
        listed = pd.to_datetime(listed, errors="coerce")
        preexisting = listed.index[listed.lt(start)]
        incomplete = first_dates.loc[first_dates.index.intersection(preexisting)] > start + pd.Timedelta(days=15)
        coverage_incomplete = int(incomplete.sum())
        snapshot = RESULTS / f"{regime}_industry_snapshot.csv"
        industry = pd.read_csv(snapshot) if args.no_baostock and snapshot.exists() else (
            pd.DataFrame() if args.no_baostock else fetch_baostock_industry(end)
        )
        if not industry.empty:
            industry.to_csv(snapshot, index=False, encoding="utf-8-sig")
        if not basic.empty:
            result = result.merge(basic, on="ts_code", how="left", suffixes=("", "_tdx"))
        if not industry.empty:
            result = result.merge(industry.rename(columns={"name": "baostock_name"}), on="ts_code", how="left")
            result["name"] = result.get("name", pd.Series(index=result.index, dtype="object")).fillna(result["baostock_name"])
        if "name" not in result:
            result["name"] = pd.NA
        if "tdx_industry_code" not in result:
            result["tdx_industry_code"] = pd.NA
        if "industry" not in result:
            result["industry"] = pd.NA
        result["industry_display"] = result["industry"].fillna(
            "通达信代码 " + result["tdx_industry_code"].astype("string")
        ).fillna("未知")
        result["regime"] = regime
        result["board"] = result["ts_code"].map(stock_board)
        result["over_500pct"] = result["max_runup_pct"] > 500
        # A mathematically exact 200% can appear as 200.00000000000006 after
        # floating-point multiplication by the adjustment factor.
        result["over_200pct"] = result["max_runup_pct"] > 200 + 1e-8
        result["start_to_peak_over_500pct"] = result["start_to_peak_pct"] > 500
        result = result.sort_values("max_gain_pct", ascending=False)
        result.to_csv(RESULTS / f"{regime}_all_stocks.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
        winners = result[result.over_500pct].copy()
        winners.to_csv(RESULTS / f"{regime}_over_500pct.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
        summary_rows.append({
            "regime": regime, "start": start.date(), "end": end.date(),
            "stocks_with_data": len(result), "over_500pct": len(winners),
            "start_to_peak_over_500pct": int(result.start_to_peak_over_500pct.sum()),
            "max_gain_pct": result.max_runup_pct.max(),
            "preexisting_stocks_start_late": coverage_incomplete,
        })
        board_counts = result.groupby("board", observed=True).agg(
            stocks_with_data=("ts_code", "size"),
            over_500pct=("over_500pct", "sum"),
            start_to_peak_over_500pct=("start_to_peak_over_500pct", "sum"),
        )
        for board in BOARD_ORDER:
            if board not in board_counts.index:
                continue
            row = board_counts.loc[board]
            board_500 = winners.loc[winners["board"].eq(board), "max_runup_pct"]
            board_200 = result.loc[result["board"].eq(board) & result["over_200pct"]]
            board_rows.append({
                "regime": regime, "board": board,
                "stocks_with_data": int(row.stocks_with_data),
                "over_500pct": int(row.over_500pct),
                "share_of_winners_pct": 100 * row.over_500pct / len(winners),
                "share_of_board_pct": 100 * row.over_500pct / row.stocks_with_data,
                "median_gain_pct": board_500.median(),
                "start_to_peak_over_500pct": int(row.start_to_peak_over_500pct),
            })
            board_200_rows.append({
                "regime": regime, "board": board,
                "stocks_with_data": int(row.stocks_with_data),
                "over_200pct": len(board_200),
                "share_of_board_pct": 100 * len(board_200) / row.stocks_with_data,
                "median_gain_pct": board_200["max_runup_pct"].median(),
            })
        if len(winners):
            g = winners.groupby("industry_display", dropna=False).agg(
                stock_count=("ts_code", "size"),
                max_gain_pct=("max_runup_pct", "max"),
                avg_gain_pct=("max_runup_pct", "mean"),
            ).reset_index().sort_values(["stock_count", "max_gain_pct"], ascending=[False, False])
            g.insert(0, "regime", regime)
            industry_rows.append(g)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    board_summary = pd.DataFrame(board_rows)
    board_summary.to_csv(RESULTS / "board_summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    board_200_summary = pd.DataFrame(board_200_rows)
    board_200_summary.to_csv(RESULTS / "board_over_200pct_summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    if industry_rows:
        industry_path = RESULTS / "industry_summary.csv"
        try:
            pd.concat(industry_rows, ignore_index=True).to_csv(industry_path, index=False, encoding="utf-8-sig", float_format="%.4f")
        except PermissionError as exc:
            raise RuntimeError(f"请关闭占用 {industry_path} 的程序后重试") from exc

    intervals = "、".join(f"{start.date()}—{end.date()}" for start, end in REGIMES.values())
    lines = ["# 最近两次牛市：最大涨幅超过 500% 的股票", "", "## 口径", "",
             f"- 牛市区间沿用 `scripts/market_regimes.py` 的固定定义：{intervals}，边界包含。",
             "- 主口径最大涨幅 =（区间内某日后复权最高价 ÷ 此前交易日的最低后复权价 − 1）× 100%，要求低点严格早于高点；同日高低价的盘中顺序无法由日线确认，故不纳入该日配对。",
             "- 对照口径为区间首个有效交易日后复权收盘至全区间后复权最高价。",
             "- 股票从区间内首个有效交易日开始计算，因此包含区间内新上市股票；阈值为严格大于 500%。",
             "- 行业使用 Baostock 在对应牛市结束日可查询到的证监会行业分类快照；缺失时保留通达信行业代码。两者为不同分类体系，代码不等于证监会行业名称。", "", "## 结果", "",
             "| 牛市区间 | 有行情股票数 | 低点至随后高点 >500% | 首日收盘至最高价 >500% | 区间内最高涨幅 |", "|---|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.regime} | {row.stocks_with_data:,} | {row.over_500pct:,} | {row.start_to_peak_over_500pct:,} | {row.max_gain_pct:.2f}% |")
    lines += ["", f"**数据完整性提示：**第一段牛市有 {summary.iloc[0].preexisting_stocks_start_late:,} 只已上市股票的本地首条行情晚于牛市开始日 15 天以上；其中部分可能因停牌而没有交易。天齐锂业（002466.SZ）现已覆盖完整区间并入选。外部 2019-02-01 在交易名单仍有 17 只沪深 A 股未见本地行情，第一段数量可能低估；详见 [数据完整性报告](data_completeness.md)。", ""]
    lines += ["", "## 板块分布", "",
              "按股票代码所属板块统计：深市 300/301/302 为创业板，沪市 688/689 为科创板，沪市 600/601/603/605 与深市 000/001/002/003 为主板，北交所 92 开头单列。表中数量采用低点至随后高点的主口径；完整数据见 `results/board_summary.csv`。", "",
              "| 牛市区间 | 板块 | 有行情股票 | 涨幅 >500% | 占本区间超 500% 股票 | 板块内占比 | 入选股票涨幅中位数 | 首日收盘至最高价 >500% |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in board_summary.itertuples(index=False):
        lines.append(f"| {row.regime} | {row.board} | {row.stocks_with_data:,} | {row.over_500pct:,} | {row.share_of_winners_pct:.1f}% | {row.share_of_board_pct:.1f}% | {row.median_gain_pct:.2f}% | {row.start_to_peak_over_500pct:,} |")
    lines += ["", "### 涨幅超过 200% 的板块分布", "",
              "沿用上文的最大后复权涨幅口径，严格大于 200% 入选；中位数仅对各板块入选股票计算。北交所单列，避免三个沪深板块之和与全部股票数混淆。完整结果见 `results/board_over_200pct_summary.csv`。", "",
              "| 牛市区间 | 板块 | 有行情股票 | 涨幅 >200% | 板块内占比 | 入选股票涨幅中位数 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in board_200_summary.itertuples(index=False):
        lines.append(f"| {row.regime} | {row.board} | {row.stocks_with_data:,} | {row.over_200pct:,} | {row.share_of_board_pct:.1f}% | {row.median_gain_pct:.2f}% |")
    lines += ["", "## 行业分布", ""]
    if industry_rows:
        industry_summary = pd.concat(industry_rows, ignore_index=True)
        for regime in REGIMES:
            subset = industry_summary[industry_summary.regime.eq(regime)]
            lines += [f"### {regime}", "", "| 行业 | 股票数 | 占该区间超 500% 股票 |", "|---|---:|---:|"]
            total = int(summary.loc[summary.regime.eq(regime), "over_500pct"].iloc[0])
            for row in subset.itertuples(index=False):
                lines.append(f"| {row.industry_display} | {row.stock_count} | {row.stock_count / total:.1%} |")
            lines.append("")
    lines += ["## 股票名单", "", "完整逐股明细和两个涨幅口径见 `results/*_over_500pct.csv`，所有股票的计算结果见 `results/*_all_stocks.csv`。", ""]
    for winners in (pd.read_csv(RESULTS / f"{regime}_over_500pct.csv") for regime in REGIMES):
        regime = winners.regime.iloc[0]
        lines += [f"### {regime}", "", "| 代码 | 名称 | 最大涨幅 | 行业 |", "|---|---|---:|---|"]
        for row in winners.head(20).itertuples(index=False):
            lines.append(f"| {row.ts_code} | {row.name if pd.notna(row.name) else '未知'} | {row.max_runup_pct:.2f}% | {row.industry_display} |")
        if len(winners) > 20:
            lines.append(f"| 其余 {len(winners) - 20} 只 | 见逐股 CSV | | |")
        lines.append("")
    (RESULTS.parent / "docs").mkdir(parents=True, exist_ok=True)
    (RESULTS.parent / "docs" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"results -> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
