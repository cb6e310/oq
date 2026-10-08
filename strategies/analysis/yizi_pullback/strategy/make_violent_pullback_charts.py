"""绘制全部暴力回调事件，D1-D3 出现跌停收盘的样本排在前面。"""
from pathlib import Path

import numpy as np
import pandas as pd

from make_representative_charts import DAILY, EVENTS, INDEX_FILES, chart_set
from candlestick_svg import aggregate_bars, mobile_css, read_tdx_day


OUT = Path(EVENTS).parent.parent / "figures" / "violent-pullback-samples.html"


def limit_down_dates(stock, d0):
    bars = stock.sort_values("date").reset_index(drop=True)
    positions = bars.index[bars.date.eq(pd.Timestamp(d0))]
    if len(positions) != 1:
        raise ValueError(f"D0 missing or duplicated: {d0}")
    start = int(positions[0]) + 1
    dates = []
    for position in range(start, min(start + 3, len(bars))):
        current = bars.iloc[position]
        previous = bars.iloc[position - 1]
        reference = previous.close * previous.adj_factor / current.adj_factor
        limit_pct = 0.20 if str(current.ts_code).startswith("30") and current.date >= pd.Timestamp("2020-08-24") else 0.10
        limit_cents = np.floor(reference * (1 - limit_pct) * 100 + 0.5 + 1e-6)
        if np.rint(current.close * 100) == limit_cents:
            dates.append(current.date)
    return dates


def pick_events(events, stocks):
    violent = events.loc[events.drop_pct.lt(-0.07)].copy()
    violent["limit_down_dates"] = [
        limit_down_dates(stocks[row.ts_code], row.yizi_date)
        for row in violent.itertuples()
    ]
    violent["has_limit_down"] = violent.limit_down_dates.map(bool)
    violent = violent.sort_values(
        ["has_limit_down", "signal_date", "ts_code"], ascending=[False, True, True]
    ).reset_index(drop=True)
    violent["group"] = violent.limit_down_dates.map(
        lambda dates: "D1-D3跌停 " + ",".join(date.strftime("%Y-%m-%d") for date in dates) if dates else "D1-D3无跌停"
    )
    violent["rank"] = range(1, len(violent) + 1)
    return violent


def main():
    events = pd.read_csv(EVENTS, parse_dates=["yizi_date", "d1_date", "signal_date"])
    violent = events.loc[events.drop_pct.lt(-0.07)]
    codes = violent.ts_code.astype(str).unique().tolist()
    daily = pd.read_parquet(DAILY, filters=[("ts_code", "in", codes)]).sort_values(["ts_code", "date"])
    stocks = {code: bars for code, bars in daily.groupby("ts_code", observed=True)}
    chosen = pick_events(events, stocks)
    index_monthly = {name: aggregate_bars(read_tdx_day(path), "M") for name, path in INDEX_FILES.items()}
    with_limit = int(chosen.has_limit_down.sum())
    without_entry = int(chosen.return_start_date.isna().sum())
    blocks = [chart_set(row, stocks[row.ts_code], index_monthly) for _, row in chosen.iterrows()]
    body = [
        '<div class="wrap"><div class="head"><h1>全部暴力回调样本 · 跌停在前</h1>',
        f'<p>暴力回调定义为信号日相对 D0 后复权收盘跌幅 &lt; -7%；共 {len(chosen)} 个事件。其中 D0 后前三根有效 K 线（D1-D3）有跌停收盘 {with_limit} 个，优先展示；其余 {len(chosen) - with_limit} 个在后。同组按信号日排序。</p>',
        f'<p>跌停按前一根收盘价与复权因子还原的当日参考价、相应板块 10%／20% 限幅及分位四舍五入判定。{without_entry} 个事件无有效买点，只展示形态，不计算路径收益。</p>',
        '<p>其余样本沿用 D1-D3 最低价 K 线收盘买入、买入后 T+1 至 T+25 最高后复权收盘价卖出的事后路径口径；红蓝浮框为原始收盘价，虚线指向对应 K 线。</p></div>',
        f'<div class="section">D1-D3 有跌停收盘（{with_limit} 个）</div>',
        *blocks[:with_limit],
        f'<div class="section">D1-D3 无跌停收盘（{len(chosen) - with_limit} 个）</div>',
        *blocks[with_limit:],
        '</div>',
    ]
    document = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>' + mobile_css() + '</style><body>' + ''.join(body) + '</body></html>'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(document, encoding="utf-8")
    print(OUT, len(chosen), with_limit, len(document))


if __name__ == "__main__":
    main()
