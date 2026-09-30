"""Download basic metadata for all mainland China A-share stocks from Tushare.

The default call downloads currently listed stocks. With ``--include-delisted``
it also requests delisted and paused-listing stocks, subject to Tushare quota.

Usage:
    $env:TUSHARE_TOKEN = "your_token"
    python scripts/download_stock_basic.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Protocol

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "database" / "processed" / "stock_basic.parquet"
LIST_STATUSES = ("L", "D", "P")
FIELDS = (
    "ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,"
    "exchange,curr_type,list_status,list_date,delist_date,is_hs,"
    "act_name,act_ent_type"
)
A_SHARE_MARKETS = {"主板", "创业板", "科创板", "北交所"}
A_SHARE_EXCHANGES = {"SSE", "SZSE", "BSE"}


class StockBasicApi(Protocol):
    def stock_basic(self, **kwargs: str) -> pd.DataFrame: ...


def download_stock_basic(api: StockBasicApi, statuses: tuple[str, ...] = LIST_STATUSES) -> pd.DataFrame:
    """Fetch requested listing statuses and return one normalized table.

    Tushare limits ``stock_basic`` to one request per minute or hour for many accounts.
    The CLI therefore defaults to one request for listed stocks; use
    ``--include-delisted`` to request D and P as separate calls after the
    account limit allows it.
    """
    parts: list[pd.DataFrame] = []
    for status in statuses:
        kwargs = {"fields": FIELDS}
        if status:
            kwargs["list_status"] = status
        frame = api.stock_basic(**kwargs)
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        if status:
            frame["list_status"] = status
        elif "list_status" not in frame.columns:
            frame["list_status"] = "L"
        parts.append(frame)

    if not parts:
        raise RuntimeError("Tushare stock_basic returned no rows")

    result = pd.concat(parts, ignore_index=True)
    if "ts_code" not in result.columns:
        raise RuntimeError("Tushare response is missing required column: ts_code")

    # stock_basic can also expose B shares and other non-A instruments. Filter
    # by Tushare's market/exchange labels so historical 43/83/87 BJ codes are
    # retained as well as the newer 920xxx codes used by the TDX price files.
    if "market" in result.columns and "exchange" in result.columns:
        result = result[
            result["market"].isin(A_SHARE_MARKETS)
            & result["exchange"].isin(A_SHARE_EXCHANGES)
        ].copy()
    if result.empty:
        raise RuntimeError("Tushare returned no Shanghai/Shenzhen/Beijing A-share rows")

    # If the API returns duplicates, prefer listed, then paused, then delisted.
    status_priority = {"L": 0, "P": 1, "D": 2}
    result["_status_priority"] = result["list_status"].map(status_priority).fillna(9)
    result = (
        result.sort_values(["ts_code", "_status_priority"])
        .drop_duplicates("ts_code", keep="first")
        .drop(columns="_status_priority")
        .sort_values("ts_code")
        .reset_index(drop=True)
    )

    for column in ("list_date", "delist_date"):
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column].replace("", pd.NA), format="%Y%m%d", errors="coerce"
            )

    # Keep repeated strings compact in Parquet, matching stock_daily.parquet.
    category_columns = (
        "area", "industry", "market", "exchange", "curr_type",
        "list_status", "is_hs", "act_ent_type",
    )
    for column in category_columns:
        if column in result.columns:
            result[column] = result[column].astype("category")

    return result


def save_stock_basic(frame: pd.DataFrame, output: Path, write_csv: bool = True) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    if write_csv:
        frame.to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download listed/delisted/paused A-share basic metadata from Tushare."
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("TUSHARE_TOKEN"),
        help="Tushare token; defaults to environment variable TUSHARE_TOKEN.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Parquet output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Only write Parquet and skip the human-readable CSV copy.",
    )
    parser.add_argument(
        "--include-delisted",
        action="store_true",
        help="Also request D/P statuses (three API calls; subject to Tushare rate limits).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token:
        print(
            "Missing Tushare token. Set TUSHARE_TOKEN or pass --token.",
            file=sys.stderr,
        )
        return 2

    api = ts.pro_api(args.token)
    statuses = LIST_STATUSES if args.include_delisted else ("L",)
    frame = download_stock_basic(api, statuses=statuses)
    save_stock_basic(frame, args.output, write_csv=not args.no_csv)

    counts = frame["list_status"].value_counts().to_dict()
    print(f"stocks={len(frame):,} status={counts}")
    print(f"parquet -> {args.output}")
    if not args.no_csv:
        print(f"csv     -> {args.output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
