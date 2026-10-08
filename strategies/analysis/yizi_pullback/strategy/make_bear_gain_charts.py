"""绘制熊市阶段 T+25 路径收益严格超过 5% 的全部事件。"""
from pathlib import Path

import pandas as pd

from make_representative_charts import DAILY, INDEX_FILES, chart_set
from candlestick_svg import aggregate_bars, mobile_css, read_tdx_day


STRATEGY = Path(__file__).resolve().parent.parent
EVENTS = STRATEGY / "results" / "market_regime_events.csv"
OUT = STRATEGY / "figures" / "bear-gain-over-5pct-samples.html"
METRIC = "ret_close_to_best_close_25"


def pick_events(events):
    chosen = events.loc[events.t25_regime.eq("熊市") & events[METRIC].gt(0.05)].copy()
    chosen = chosen.sort_values([METRIC, "signal_date", "ts_code"], ascending=[False, True, True]).reset_index(drop=True)
    chosen["group"] = "熊市收益超过5%"
    chosen["rank"] = range(1, len(chosen) + 1)
    return chosen


def main():
    events = pd.read_csv(EVENTS, parse_dates=["yizi_date", "signal_date"])
    chosen = pick_events(events)
    codes = chosen.ts_code.astype(str).unique().tolist()
    daily = pd.read_parquet(DAILY, filters=[("ts_code", "in", codes)]).sort_values(["ts_code", "date"])
    stocks = {code: bars for code, bars in daily.groupby("ts_code", observed=True)}
    index_monthly = {name: aggregate_bars(read_tdx_day(path), "M") for name, path in INDEX_FILES.items()}
    blocks = [chart_set(row, stocks[row.ts_code], index_monthly) for _, row in chosen.iterrows()]
    body = [
        '<div class="wrap"><div class="head"><h1>熊市收益超过5%的全部样本</h1>',
        f'<p>共 {len(chosen)} 个事件，按区间收益从高到低排列。市场阶段为买入日至 T+25 观察边界全程熊市，收益严格大于 5%。同一股票的不同事件分别展示。</p>',
        '<p>D0 后 D1-D3 最低价 K 线收盘买入；买入后 T+1 至 T+25 内最高后复权收盘价为事后最佳卖出点。此收益表示路径潜力，不等于可预知实盘收益。</p>',
        '<p>红/蓝浮框分别标示原始买入/卖出收盘价，垂直虚线指向对应 K 线；日线必要时延伸至买卖点，价格坐标为对数刻度。</p></div>',
        f'<div class="section">熊市收益 &gt; 5%（{len(chosen)} 个）</div>',
        *blocks,
        '</div>',
    ]
    document = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>' + mobile_css() + '</style><body>' + ''.join(body) + '</body></html>'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(document, encoding="utf-8")
    print(OUT, len(chosen), len(document))


if __name__ == "__main__":
    main()
