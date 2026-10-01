"""One-click launcher for the local K-line similarity UI."""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def start_service(host: str, port: int, data: Path) -> None:
    command = [sys.executable, "-m", "strategies.kline_similarity.app",
               "--host", host, "--port", str(port), "--data", str(data)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(command, cwd=ROOT, creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     close_fds=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the local K-line similarity UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "database" / "processed" / "stock_daily_qfq.parquet")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"
    if not port_open(args.host, args.port):
        start_service(args.host, args.port, args.data)
        for _ in range(120):
            if port_open(args.host, args.port):
                break
            time.sleep(0.25)
    if not args.no_browser:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
