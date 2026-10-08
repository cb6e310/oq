"""生成 ZT 回调公式样本的日/周/月/指数月线手机长图。"""
from pathlib import Path
import sys
import pandas as pd

STRATEGY = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
EVENTS = STRATEGY / "results" / "events.csv"
DAILY = ROOT / "database" / "processed" / "stock_daily.parquet"
RAW = ROOT / "database" / "raw" / "tdx"
OUT = STRATEGY / "figures" / "representative-samples.html"
sys.path.insert(0, str(ROOT / "scripts"))
from candlestick_svg import aggregate_bars, centered_window, mobile_css, read_tdx_day, render_chart

INDEX_FILES = {"创业板指": RAW / "sz" / "lday" / "sz399006.day", "上证指数": RAW / "sh" / "lday" / "sh000001.day"}

def pick_events(events):
    metric = "ret_close_to_best_close_25"
    events = events.dropna(subset=[metric]).copy()
    mean = events[metric].mean()
    events["mean_gap"] = (events[metric] - mean).abs()
    groups = [("高收益", events.nlargest(10, metric)), ("接近均值", events.nsmallest(10, "mean_gap")), ("低收益", events.nsmallest(10, metric))]
    rows = []
    for label, part in groups:
        for rank, (_, row) in enumerate(part.sort_values(metric, ascending=label == "低收益").iterrows(), 1):
            rows.append({"group": label, "rank": rank, **row.to_dict()})
    return pd.DataFrame(rows), mean

def path_daily_window(stock_daily, anchor, row):
    bars = aggregate_bars(stock_daily, "D")
    window = centered_window(bars, anchor, "D", 44, 44)
    if pd.isna(row.return_start_date) or pd.isna(row.return_end_date):
        return window
    positions = bars.index[bars.date.isin(pd.to_datetime([row.return_start_date, row.return_end_date]))]
    target = int(bars.index[bars.date.eq(anchor)][0])
    first = max(0, min(target - 44, int(positions.min()) - 3))
    last = min(len(bars), max(target + 45, int(positions.max()) + 4))
    return bars.iloc[first:last].reset_index(drop=True)

def path_markers(row, stock_daily):
    if pd.isna(row.return_start_date) or pd.isna(row.return_end_date):
        return []
    bars = stock_daily.set_index("date")
    return [{"date": row.return_start_date, "price": bars.loc[pd.Timestamp(row.return_start_date), "close"], "side": "buy"}, {"date": row.return_end_date, "price": bars.loc[pd.Timestamp(row.return_end_date), "close"], "side": "sell"}]

def chart_set(row, stock_daily, index_monthly):
    anchor, signal = pd.Timestamp(row.zt_date), pd.Timestamp(row.signal_date)
    subtitle = f"ZT收盘 {row.zt_close:.2f} · 信号收盘 {row.signal_close:.2f} · T={int(row.t)} · 回调 {row.pullback_pct:+.2%}"
    if pd.notna(row.return_start_date) and pd.notna(row.return_end_date):
        subtitle += f" · 信号日买入 {pd.Timestamp(row.return_start_date).date()} → {pd.Timestamp(row.return_end_date).date()} · 区间收益 {row.ret_close_to_best_close_25:+.2%} · 超额 {row.ex_close_to_best_close_25:+.2%}"
    else:
        subtitle += " · 后续行情不足，不计算T+25收益"
    daily_window = path_daily_window(stock_daily, anchor, row)
    daily_range = (daily_window.date.min(), daily_window.date.max())
    pieces = [f'<div class="sample"><div class="sample-head">{row["group"]} {int(row["rank"]):02d} · {row.ts_code} · ZT {anchor.date()} · 信号 {signal.date()}</div>']
    pieces += ["<div class=\"period-label\">日线 · ZT前后各44根，必要时延伸至收益起止点</div>", render_chart(daily_window, f"{row.ts_code} 日线", subtitle, anchor, signal, "D", price_markers=path_markers(row, stock_daily))]
    for period, label in (("W", "周线"), ("M", "个股月线")):
        window = centered_window(aggregate_bars(stock_daily, period), anchor, period, 22, 22)
        pieces += [f"<div class=\"period-label\">{label} · 45根</div>", render_chart(window, f"{row.ts_code} {label}", subtitle, anchor, signal, period, range_start=daily_range[0], range_end=daily_range[1])]
    for name, bars in index_monthly.items():
        window = centered_window(bars, anchor, "M", 22, 22)
        pieces += [f"<div class=\"period-label\">{name}月线 · 个股日线完整视窗</div>", render_chart(window, f"{name} 月线", f"同步显示 {row.ts_code} 日线视窗", anchor, signal, "M", range_start=daily_range[0], range_end=daily_range[1])]
    return "".join(pieces) + "</div>"

def main():
    events = pd.read_csv(EVENTS, parse_dates=["zt_date", "signal_date", "return_start_date", "return_end_date"])
    chosen, mean = pick_events(events)
    codes = chosen.ts_code.astype(str).unique().tolist()
    daily = pd.read_parquet(DAILY).query("ts_code in @codes").sort_values(["ts_code", "date"])
    indices = {name: aggregate_bars(read_tdx_day(path), "M") for name, path in INDEX_FILES.items()}
    blocks = [chart_set(row, daily[daily.ts_code.astype(str).eq(row.ts_code)].copy(), indices) for _, row in chosen.iterrows()]
    body = [f'<div class="wrap"><div class="head"><h1>ZT回调公式 · 多周期样本长图</h1><p>日线以ZT前后各44根为基础，必要时延伸至收益起止点；周线、个股月线和两项指数月线各展示45根。</p><p>红色为信号日收盘买入，蓝色为买入后T+1至T+25最高收盘卖出；最高卖点属于事后最优。区间收益均值 {mean:+.2%}。</p></div>']
    offset = 0
    for label in ("高收益样本（10）", "接近均值样本（10）", "低收益样本（10）"):
        body.append(f'<div class="section">{label}</div>')
        body.extend(blocks[offset:offset + 10])
        offset += 10
    OUT.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>' + mobile_css() + '</style><body>' + "".join(body) + "</div></body></html>", encoding="utf-8")
    print(OUT)

if __name__ == "__main__":
    main()
