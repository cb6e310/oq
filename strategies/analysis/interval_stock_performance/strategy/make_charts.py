"""Render the Top 30 interval stocks with matching Shanghai Composite daily K-lines."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "strategies" / "analysis" / "interval_stock_performance"
CSV = PROJECT / "results" / "leaders_2015-09-15_2015-12-22.csv"
DAILY = ROOT / "database" / "processed" / "stock_daily_qfq.parquet"
RAW = ROOT / "database" / "raw" / "tdx"
FIGURES = PROJECT / "figures"
REPORT = PROJECT / "docs" / "report.md"

sys.path.insert(0, str(ROOT / "scripts"))
from candlestick_svg import aggregate_bars, read_tdx_day, render_chart  # noqa: E402


SVG_CSS = (
    ".panel{display:block;width:100%;height:auto}.title{font-size:14px;font-weight:600;fill:#222}"
    ".sub,.axis,.mark{font-size:11px;fill:#666}.targetmark{fill:#9a6700}.signalmark{fill:#315d9e}"
    ".grid{stroke:#e6e6e6;stroke-width:1}.d0band{fill:#fff2a8;opacity:.72}"
    ".viewportband{fill:#fff2a8;opacity:.45}.sigline{stroke:#315d9e;stroke-width:1.2;stroke-dasharray:4 3}"
    ".volline{stroke:#bbb;stroke-width:1}"
)


def standalone_svg(svg: str) -> str:
    if "xmlns=" not in svg[:300]:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    if "<style>" not in svg[:1000]:
        first_close = svg.find(">")
        svg = svg[: first_close + 1] + f"<style>{SVG_CSS}</style>" + svg[first_close + 1 :]
    return svg


def combine_svgs(first: str, second: str, gap: int = 16) -> str:
    """Place two renderer SVGs into one valid, self-contained SVG."""
    def unpack(svg: str) -> tuple[float, float, str]:
        match = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
        if not match:
            raise ValueError("rendered SVG has no viewBox")
        width, height = float(match.group(1)), float(match.group(2))
        body = svg[svg.find(">") + 1 : svg.rfind("</svg>")]
        body = re.sub(r"<style>.*?</style>", "", body, flags=re.S)
        return width, height, body

    w1, h1, b1 = unpack(first)
    w2, h2, b2 = unpack(second)
    total_width = max(w1, w2)
    total_height = h1 + gap + h2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_width:.0f} {total_height:.0f}" '
        f'role="img"><style>{SVG_CSS}</style><g>{b1}</g>'
        f'<rect x="0" y="{h1:.0f}" width="{total_width:.0f}" height="{gap}" fill="#e8e8e8"/>'
        f'<g transform="translate(0,{h1 + gap:.0f})">{b2}</g></svg>'
    )


def chart_block(row: pd.Series, stock: pd.DataFrame, index: pd.DataFrame,
                start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, int, int, pd.Timestamp, pd.Timestamp]:
    code = str(row.ts_code)
    name = str(row["name"])
    title = f"{code} {name} 日线"
    subtitle = (
        f"区间 {start:%Y-%m-%d} 至 {end:%Y-%m-%d} · 区间涨幅 {row.interval_return:+.2%} · "
        f"最高 {row.interval_high_adj:.4f}（{row.interval_high_date}）· "
        f"最低 {row.interval_low_adj:.4f}（{row.interval_low_date}）"
    )
    stock_bars = aggregate_bars(stock.drop(columns=["ts_code"], errors="ignore"), "D")
    interval_rows = stock_bars.index[stock_bars.date.ge(start)]
    if len(interval_rows) == 0:
        raise ValueError(f"no stock bars in the interval for {code}")
    first_interval = int(interval_rows[0])
    stock_start = max(0, first_interval - 40)
    future_rows = stock_bars.index[stock_bars.date.gt(end)]
    stock_end = int(future_rows[min(19, len(future_rows) - 1)]) if len(future_rows) else len(stock_bars) - 1
    stock_bars = stock_bars.iloc[stock_start:stock_end + 1].reset_index(drop=True)
    bars_before = first_interval - stock_start
    bars_after = int(stock_bars.date.gt(end).sum())
    window_start = pd.Timestamp(stock_bars.date.iloc[0])
    window_end = pd.Timestamp(stock_bars.date.iloc[-1])
    index_bars = aggregate_bars(index[index.date.between(window_start, window_end)], "D")
    axis_dates = pd.DatetimeIndex(index_bars.date)
    stock_svg = render_chart(
        stock_bars, title, subtitle, period="D", width=1100, height=410,
        target_start=start, target_end=end, axis_dates=axis_dates,
    )
    index_svg = render_chart(
        index_bars, "上证指数 日线", f"同步区间 {start:%Y-%m-%d} 至 {end:%Y-%m-%d} · 对照指数",
        period="D", width=1100, height=410, target_start=start, target_end=end,
        axis_dates=axis_dates,
    )
    return combine_svgs(stock_svg, index_svg), bars_before, bars_after, window_start, window_end


def update_report(rows: pd.DataFrame, paths: list[Path]) -> None:
    report = REPORT.read_text(encoding="utf-8")
    marker = "\n## 区间日K线图\n"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    lines = [
        marker.rstrip("\n"), "",
        "每张宽版图上方为个股日 K 线，下方为上证指数日 K 线，两图共用同一交易日横坐标；停牌日保留空位，不补造 K 线。个股图在统计起点前展示最多 40 根、统计终点后展示 20 根日 K。黄色区域只标记统计区间。绘图复用之前分析项目的统一样式：对数价格轴、红涨绿跌、MA5/MA10/MA20 和成交量。", "",
        "完整长图页面：[interval-charts.html](../figures/interval-charts.html)", "",
    ]
    for row, path in zip(rows.itertuples(index=False), paths):
        lines.extend([
            f"### {int(row.rank):02d}. {row.ts_code} {row.name}", "",
            f"![{row.ts_code} 区间日K线](../{path.relative_to(PROJECT).as_posix()})", "",
        ])
    REPORT.write_text(report + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = pd.read_csv(CSV)
    if rows.empty:
        raise ValueError("ranking CSV is empty")
    start, end = pd.Timestamp("2015-09-15"), pd.Timestamp("2015-12-22")
    FIGURES.mkdir(parents=True, exist_ok=True)
    index = read_tdx_day(RAW / "sh" / "lday" / "sh000001.day")
    if not index.date.between(start, end).any():
        raise ValueError("Shanghai Composite index has no bars in the interval")
    codes = rows.ts_code.astype(str).tolist()
    stock = pd.read_parquet(
        DAILY,
        columns=["ts_code", "date", "open", "high", "low", "close", "vol"],
        filters=[("ts_code", "in", codes)],
    )
    stock["ts_code"] = stock.ts_code.astype(str)
    paths: list[Path] = []
    blocks = [
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>区间股票表现统计 · {start:%Y-%m-%d} 至 {end:%Y-%m-%d}</title>",
        "<style>body{margin:0;background:#f2f3f5;font-family:Arial,'Microsoft YaHei',sans-serif;color:#222}.wrap{width:min(1180px,96vw);margin:0 auto;background:#fff}.head{padding:20px 24px 14px;border-bottom:1px solid #ddd}.head h1{font-size:22px;margin:0 0 8px}.head p{font-size:14px;color:#666;margin:0;line-height:1.6}.sample{border-top:8px solid #e8e8e8}.sample-head{padding:14px 20px 8px;font-size:15px;font-weight:600;background:#fff}.panel{display:block;width:100%;height:auto;border-bottom:1px solid #ddd}</style></head><body><div class='wrap'>",
        "<div class='head'><h1>区间股票表现统计 · Top 30 日 K 线</h1><p>个股图包含 2015-09-15 前最多 40 根日 K；上方为个股，下方为同日期窗口的上证指数。个股使用前复权日线，指数使用原始指数日线。</p></div>",
    ]
    for _, row in rows.iterrows():
        code = str(row.ts_code)
        item = stock[stock.ts_code.eq(code)].sort_values("date").copy()
        if item.empty:
            raise ValueError(f"no stock bars found for {code}")
        combined, bars_before, bars_after, window_start, window_end = chart_block(row, item, index, start, end)
        rank = int(row["rank"])
        path = FIGURES / f"{rank:02d}_{code.replace('.', '_')}_interval.svg"
        path.write_text(combined, encoding="utf-8")
        paths.append(path)
        blocks.append(
            f"<div class='sample'><div class='sample-head'>排名 {rank:02d} · "
            f"{html.escape(code)} {html.escape(str(row['name']))} · 区间涨幅 {row.interval_return:+.2%} · "
            f"统计起点前 {bars_before} 根、终点后 {bars_after} 根日 K · "
            f"图表日期 {window_start:%Y-%m-%d} 至 {window_end:%Y-%m-%d}</div>"
            f"{combined}</div>"
        )
    gallery = FIGURES / "interval-charts.html"
    gallery.write_text("".join(blocks) + "</div></body></html>", encoding="utf-8")
    update_report(rows, paths)
    print(f"Saved {len(paths)} combined charts to {FIGURES}")
    print(f"Saved gallery to {gallery}")
    print(f"Updated report {REPORT}")


if __name__ == "__main__":
    main()
