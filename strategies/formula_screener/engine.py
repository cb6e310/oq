"""Python bridge from the local Parquet database to formula-go."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "database" / "processed" / "stock_daily_qfq.parquet"
DEFAULT_BASIC = ROOT / "database" / "processed" / "stock_basic_tdx.parquet"


def go_command() -> list[str]:
    configured = os.environ.get("FORMULA_SCREENER_GO")
    bundled_user_install = Path("D:/software/Go/1.27.3/bin/go.exe")
    go = configured or shutil.which("go") or (str(bundled_user_install) if bundled_user_install.exists() else None)
    if not go:
        raise RuntimeError("未找到 Go。请安装 Go 1.25.4 或更新版本，并重启服务。")
    return [go, "run", "./cmd/worker"]


def load_market_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到日线数据库：{path}")
    columns = ["date", "ts_code", "open", "high", "low", "close", "vol", "amount"]
    frame = pd.read_parquet(path, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["ts_code", "date"], kind="stable")


def load_names(path: Path = DEFAULT_BASIC) -> dict[str, str]:
    if not path.exists():
        return {}
    basic = pd.read_parquet(path, columns=["ts_code", "name"])
    return {str(row.ts_code): str(row.name) for row in basic.itertuples(index=False)}


def stock_requests(frame: pd.DataFrame, formula_text: str, start: str | None,
                   end: str | None) -> Iterator[dict]:
    for symbol, group in frame.groupby("ts_code", sort=False, observed=True):
        if end:
            group = group[group.date <= pd.Timestamp(end)]
        if group.empty:
            continue
        rows = group[["date", "open", "high", "low", "close", "vol", "amount"]].copy()
        rows["date"] = rows.date.dt.strftime("%Y-%m-%d")
        rows = rows.astype({"open": "float64", "high": "float64", "low": "float64",
                            "close": "float64", "vol": "float64", "amount": "float64"})
        yield {"formula": formula_text, "symbol": str(symbol),
               "bars": rows.rename(columns={"date": "date"}).to_dict("records")}


class FormulaWorker:
    """One persistent formula-go process, fed one stock at a time as JSONL."""

    def __init__(self):
        self.process = subprocess.Popen(
            go_command(), cwd=PROJECT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
        )

    def evaluate(self, request: dict) -> dict:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("公式执行进程不可用")
        self.process.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(error.strip() or "公式执行进程提前退出")
        result = json.loads(line)
        if result.get("error"):
            raise ValueError(result["error"])
        return result

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


def collect_events(frame: pd.DataFrame, formula_text: str,
                   start: str | None = None, end: str | None = None,
                   progress=None) -> list[dict]:
    worker = FormulaWorker()
    events: list[dict] = []
    requests = stock_requests(frame, formula_text, start, end)
    total = frame.ts_code.nunique()
    try:
        for index, request in enumerate(requests, 1):
            result = worker.evaluate(request)
            events.extend({"ts_code": result["symbol"], **signal}
                          for signal in result.get("signals", [])
                          if not start or signal["date"] >= start)
            update_interval = max(1, total // 100)
            if progress and index % update_interval == 0 and index < total:
                progress(index, total)
    finally:
        worker.close()
    if progress:
        progress(total, total)
    return events


def add_forward_returns(events: pd.DataFrame, market: pd.DataFrame,
                        horizons: tuple[int, ...] = (1, 3, 5, 10, 20)) -> pd.DataFrame:
    """Measure close-to-close returns over each symbol's next trading bars."""
    if events.empty:
        return events.copy()
    parts: list[dict | None] = [None] * len(events)
    lookup = {}
    for symbol, group in market.groupby("ts_code", sort=False, observed=True):
        if not group.date.is_monotonic_increasing:
            group = group.sort_values("date")
        lookup[symbol] = (pd.to_datetime(group.date), group.close.to_numpy())
    event_groups = events.groupby("ts_code", sort=False, observed=True).indices
    event_values = events.value.to_numpy()
    for symbol, event_indices in event_groups.items():
        cached = lookup.get(symbol)
        if cached is None:
            continue
        dates, closes = cached
        event_dates = pd.to_datetime(events.date.iloc[event_indices])
        positions = dates.searchsorted(event_dates)
        date_values = event_dates.to_numpy()
        valid = positions < len(closes)
        valid_indices = [offset for offset, is_valid in enumerate(valid) if is_valid]
        valid_indices = [offset for offset in valid_indices
                         if dates.iloc[positions[offset]] == date_values[offset]]
        for offset in valid_indices:
            event_index = event_indices[offset]
            position = positions[offset]
            row = {"ts_code": symbol, "signal_date": pd.Timestamp(date_values[offset]),
                   "signal_close": float(closes[position]), "signal_value": float(event_values[event_index])}
            for horizon in horizons:
                future = position + horizon
                row[f"return_{horizon}d"] = (
                    float(closes[future] / closes[position] - 1)
                    if future < len(closes) else None
                )
            parts[event_index] = row
    return pd.DataFrame([part for part in parts if part is not None])
