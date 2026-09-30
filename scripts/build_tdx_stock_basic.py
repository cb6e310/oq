"""Build project stock metadata from copied TongdaXin files.

Input files live under ``database/raw/tdx/meta`` and are copied from the local
TongdaXin installation.  The output follows the project's normal processed-data
layout: ``database/processed/stock_basic_tdx.parquet`` and ``.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META = ROOT / "database" / "raw" / "tdx" / "meta"
DEFAULT_OUTPUT = ROOT / "database" / "processed" / "stock_basic_tdx.parquet"


def read_dbf(path: Path) -> pd.DataFrame:
    """Read the simple dBase III table used by TongdaXin's ``base.dbf``."""
    raw = path.read_bytes()
    record_count = int.from_bytes(raw[4:8], "little")
    header_size = int.from_bytes(raw[8:10], "little")
    record_size = int.from_bytes(raw[10:12], "little")
    fields: list[tuple[str, int]] = []
    for offset in range(32, header_size - 1, 32):
        name = raw[offset : offset + 11].split(b"\0", 1)[0].decode("ascii")
        fields.append((name, raw[offset + 16]))

    rows: list[dict[str, str]] = []
    for row_no in range(record_count):
        start = header_size + row_no * record_size
        if raw[start : start + 1] == b"*":
            continue
        offset = start + 1
        row: dict[str, str] = {}
        for name, width in fields:
            value = raw[offset : offset + width].decode("ascii", errors="ignore").strip()
            row[name] = value
            offset += width
        rows.append(row)
    return pd.DataFrame(rows)


def read_name_file(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_bytes().decode("gbk", errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0].isdigit():
            rows.append({"symbol": parts[0], "name": parts[1].strip()})
    return pd.DataFrame(rows).drop_duplicates("symbol")


def read_industry_file(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        parts = line.split("|")
        if len(parts) >= 6 and len(parts[1]) == 6 and parts[1].isdigit():
            rows.append(
                {
                    "symbol": parts[1],
                    "tdx_market": parts[0],
                    "tdx_industry_code": parts[2],
                    "tdx_industry_code_2": parts[5],
                }
            )
    return pd.DataFrame(rows).drop_duplicates("symbol")


def exchange_for(symbol: str, market: str = "") -> str:
    if market == "1" or symbol.startswith(("6", "68")):
        return "SH"
    if market == "2" or symbol.startswith("8") or symbol.startswith("4"):
        return "BJ"
    return "SZ"


def build_stock_basic(meta_dir: Path) -> pd.DataFrame:
    base = read_dbf(meta_dir / "base.dbf")
    names = read_name_file(meta_dir / "infoharbor_ex.code")
    industry = read_industry_file(meta_dir / "tdxhy.cfg")

    base = base.rename(columns={"GPDM": "symbol", "DY": "tdx_area_code", "HY": "tdx_industry_no"})
    keep = ["symbol", "tdx_area_code", "tdx_industry_no", "SSDATE", "GXRQ", "ZGB", "LTAG", "ZZC", "JZC", "ZYSY", "JLY"]
    base = base[[c for c in keep if c in base.columns]]
    result = names.merge(base, on="symbol", how="outer").merge(industry, on="symbol", how="left")
    result = result[result["symbol"].str.fullmatch(r"\d{6}", na=False)].copy()
    result["exchange"] = [exchange_for(s, m) for s, m in zip(result["symbol"], result["tdx_market"].fillna(""))]
    result["ts_code"] = result["symbol"] + "." + result["exchange"]

    for column in ("SSDATE", "GXRQ"):
        if column in result:
            result[column] = pd.to_datetime(result[column].replace("", pd.NA), format="%Y%m%d", errors="coerce")
    result = result.rename(columns={"SSDATE": "list_date", "GXRQ": "tdx_update_date"})
    for column in ("ZGB", "LTAG", "ZZC", "JZC", "ZYSY", "JLY"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.rename(columns={"ZGB": "total_shares", "LTAG": "float_shares", "ZZC": "total_assets", "JZC": "net_assets", "ZYSY": "operating_income", "JLY": "net_profit"})
    columns = ["ts_code", "symbol", "name", "exchange", "tdx_area_code", "tdx_industry_no", "tdx_industry_code", "tdx_industry_code_2", "list_date", "tdx_update_date", "total_shares", "float_shares", "total_assets", "net_assets", "operating_income", "net_profit"]
    result = result[[c for c in columns if c in result]].sort_values("ts_code").drop_duplicates("ts_code").reset_index(drop=True)
    for column in ("exchange", "tdx_area_code", "tdx_industry_no", "tdx_industry_code", "tdx_industry_code_2"):
        if column in result:
            result[column] = result[column].astype("category")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build stock metadata from TongdaXin files.")
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = build_stock_basic(args.meta_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    frame.to_csv(args.output.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"stocks={len(frame):,} exchanges={frame.exchange.value_counts().to_dict()}")
    print(f"parquet -> {args.output}")
    print(f"csv     -> {args.output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
