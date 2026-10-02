"""展示全部暴力回调和固定抽样的中等回调多周期样本。"""
from pathlib import Path

import pandas as pd

from make_representative_charts import DAILY, EVENTS, INDEX_FILES, chart_set
from candlestick_svg import aggregate_bars, mobile_css, read_tdx_day


OUT = Path(EVENTS).parent.parent / "figures" / "pullback-samples.html"
SEED = 20261002


def pick_events(events):
    violent = events[events.drop_pct.lt(-0.07)].sort_values(["signal_date", "ts_code"]).copy()
    medium = events[events.drop_pct.ge(-0.07) & events.drop_pct.lt(-0.03)]
    sampled = medium.sample(n=min(14, len(medium)), random_state=SEED).copy()
    violent["group"] = "暴力回调"
    violent["rank"] = range(1, len(violent) + 1)
    sampled["group"] = "中等回调抽样"
    sampled["rank"] = range(1, len(sampled) + 1)
    return pd.concat([violent, sampled], ignore_index=True), len(violent), len(medium)


def main():
    events = pd.read_csv(EVENTS, parse_dates=["yizi_date", "d1_date", "signal_date"])
    chosen, violent_count, medium_total = pick_events(events)
    codes = chosen.ts_code.astype(str).unique().tolist()
    daily = pd.read_parquet(DAILY, filters=[("ts_code", "in", codes)]).sort_values(["ts_code", "date"])
    stocks = {code: bars for code, bars in daily.groupby("ts_code", observed=True)}
    index_monthly = {name: aggregate_bars(read_tdx_day(path), "M") for name, path in INDEX_FILES.items()}
    blocks = [chart_set(row, stocks[row.ts_code], index_monthly) for _, row in chosen.iterrows()]
    invalid_count = chosen.iloc[:violent_count].return_start_date.isna().sum()
    body = [
        '<div class="wrap"><div class="head"><h1>一字板回调 · 暴力与中等回调样本</h1>',
        f'<p>全部暴力回调（跌幅 &lt; -7%）{violent_count} 个；中等回调（-7% 至 -3%，含 -7%）共 {medium_total} 个，固定随机种子 {SEED} 抽取 {len(chosen) - violent_count} 个。</p>',
        f'<p>暴力回调中 {invalid_count} 个没有有效买点，仍展示形态，但不标买卖点或计算收益。其余样本按 D1-D3 最低价 K 线收盘买入、买入后 T+1 至 T+25 最高收盘卖出；最佳卖出点为事后最优。</p>',
        '<p>日线覆盖 D0 前后各 44 根并在必要时延伸至买卖点；周线、个股月线及两项指数月线各展示 45 根。价格为对数刻度。</p></div>',
        f'<div class="section">全部暴力回调（{violent_count} 个）</div>',
        *blocks[:violent_count],
        f'<div class="section">中等回调固定随机抽样（{len(chosen) - violent_count} 个）</div>',
        *blocks[violent_count:],
        '</div>',
    ]
    document = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>' + mobile_css() + '</style><body>' + ''.join(body) + '</body></html>'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(document, encoding="utf-8")
    print(OUT, len(chosen), len(document))


if __name__ == "__main__":
    main()
