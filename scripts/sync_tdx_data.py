"""Synchronize local TongDaXin files and rebuild the project market database.

The local TDX client is still the source of truth and is intentionally manual:
this script only copies files that already exist locally.  It records both the
source-data date and the rebuilt database date so a stale local TDX download is
visible to the UI.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TDX = Path(r"D:\software\tdx")
STATUS_FILE = ROOT / "database" / "processed" / "tdx_sync_status.json"


def _copy_tree(source: Path, target: Path, pattern: str) -> int:
    if not source.exists():
        return 0
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in source.glob(pattern):
        if not src.is_file():
            continue
        dst = target / src.name
        if not dst.exists() or src.stat().st_size != dst.stat().st_size or src.stat().st_mtime_ns > dst.stat().st_mtime_ns:
            shutil.copy2(src, dst)
            count += 1
    return count


def source_latest_date(tdx_root: Path) -> str | None:
    """Return the latest valid date encoded by the last record in stock .day files."""
    import struct

    latest: int | None = None
    import re
    stock_re = re.compile(r"^(sh(60[0-5]|68[89])|sz(00[0-3]|30[0-2])|bj92)\d+\.day$")
    for market in ("sh", "sz", "bj"):
        for path in (tdx_root / "vipdoc" / market / "lday").glob(f"{market}*.day"):
            if not stock_re.match(path.name):
                continue
            try:
                size = path.stat().st_size
                if size < 32:
                    continue
                with path.open("rb") as fh:
                    fh.seek(size - size % 32 - 32)
                    raw = fh.read(4)
                value = struct.unpack("<I", raw)[0]
                if 20000101 <= value <= 20991231:
                    latest = max(latest or value, value)
            except (OSError, struct.error):
                continue
    return str(latest) if latest else None


def sync_raw(tdx_root: Path) -> dict:
    raw = ROOT / "database" / "raw" / "tdx"
    copied = 0
    for market in ("sh", "sz", "bj"):
        copied += _copy_tree(tdx_root / "vipdoc" / market / "lday", raw / market / "lday", f"{market}*.day")
    gbbq = tdx_root / "T0002" / "hq_cache" / "gbbq"
    if gbbq.exists():
        copied += _copy_tree(gbbq.parent, raw, "gbbq")
    meta = raw / "meta"
    cache = tdx_root / "T0002" / "hq_cache"
    for name in ("infoharbor_ex.code", "infoharbor_ex.name", "tdxhy.cfg"):
        copied += _copy_tree(cache, meta, name)
    return {"copied_files": copied, "source_latest_date": source_latest_date(tdx_root)}


def rebuild() -> None:
    env = os.environ.copy()
    for script in ("build_adjusted.py", "build_tdx_stock_basic.py"):
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if proc.returncode:
            raise RuntimeError((proc.stderr or proc.stdout).strip())


def run(tdx_root: Path = DEFAULT_TDX) -> dict:
    tdx_root = Path(tdx_root)
    if not (tdx_root / "vipdoc").exists():
        raise FileNotFoundError(f"未找到通达信 vipdoc 目录：{tdx_root / 'vipdoc'}")
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    info = sync_raw(tdx_root)
    rebuild()
    info.update({
        "status": "success",
        "source_root": str(tdx_root),
        "database_latest_date": database_latest_date(),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "started_at": started,
    })
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def database_latest_date() -> str | None:
    try:
        import pyarrow.parquet as pq
        path = ROOT / "database" / "processed" / "stock_daily.parquet"
        if not path.exists():
            return None
        parquet = pq.ParquetFile(path)
        index = parquet.schema_arrow.get_field_index("date")
        dates = [parquet.metadata.row_group(i).column(index).statistics.max
                 for i in range(parquet.metadata.num_row_groups)]
        return max(dates).strftime("%Y-%m-%d")
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="同步本地通达信日线并重建项目数据库")
    parser.add_argument("--tdx-root", type=Path, default=DEFAULT_TDX)
    args = parser.parse_args()
    print(json.dumps(run(args.tdx_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
