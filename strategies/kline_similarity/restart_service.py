"""Restart the K-line UI after its old HTTP listener has closed."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path


def wait_for_port_release(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((bind_host, port))
                return
            except OSError:
                time.sleep(0.2)
    raise TimeoutError(f"服务端口 {port} 在 {timeout:g} 秒内未释放")


def main() -> int:
    parser = argparse.ArgumentParser(description="Restart the K-line similarity service")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--data", required=True, type=Path)
    args = parser.parse_args()
    wait_for_port_release(args.host, args.port)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [sys.executable, "-m", "strategies.kline_similarity.app",
         "--host", args.host, "--port", str(args.port), "--data", str(args.data)],
        cwd=Path(__file__).resolve().parents[2],
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
