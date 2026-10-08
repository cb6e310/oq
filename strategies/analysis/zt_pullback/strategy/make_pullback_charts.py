"""展示 ZT 回调公式的深度回调与中等回调多周期样本。"""
from pathlib import Path
import pandas as pd
from make_representative_charts import DAILY, EVENTS, INDEX_FILES, chart_set
from candlestick_svg import aggregate_bars, mobile_css, read_tdx_day

OUT = Path(EVENTS).parent.parent / "figures" / "pullback-samples.html"
SEED = 20261002

def pick_events(events):
    deep = events[events.pullback_pct.lt(-0.07)].sort_values(["signal_date", "ts_code"]).copy()
    medium = events[events.pullback_pct.ge(-0.07) & events.pullback_pct.le(-0.03)]
    sampled = medium.sample(n=min(14, len(medium)), random_state=SEED).copy()
    deep["group"] = "深度回调"; deep["rank"] = range(1, len(deep) + 1)
    sampled["group"] = "中等回调抽样"; sampled["rank"] = range(1, len(sampled) + 1)
    return pd.concat([deep, sampled], ignore_index=True), len(deep), len(medium)

def main():
    events = pd.read_csv(EVENTS, parse_dates=["zt_date", "signal_date", "return_start_date", "return_end_date"])
    chosen, deep_count, medium_total = pick_events(events)
    codes = chosen.ts_code.astype(str).unique().tolist()
    daily = pd.read_parquet(DAILY).query("ts_code in @codes").sort_values(["ts_code", "date"])
    indices = {name: aggregate_bars(read_tdx_day(path), "M") for name, path in INDEX_FILES.items()}
    blocks = [chart_set(row, daily[daily.ts_code.astype(str).eq(row.ts_code)].copy(), indices) for _, row in chosen.iterrows()]
    invalid = chosen.iloc[:deep_count].return_start_date.isna().sum()
    body = [f'<div class="wrap"><div class="head"><h1>ZT回调公式 · 回调样本</h1><p>深度回调（跌幅 &lt; -7%）{deep_count}个；边界回调（-7%至-3%，含-3%）{medium_total}个，固定随机种子{SEED}抽取{len(chosen)-deep_count}个。</p><p>由于公式的最低收盘约束，信号相对ZT收盘的回撤下限约为-3%，因此本页不预设暴力回调样本。信号日收盘买入，T+1至T+25最高收盘卖出；日线以ZT前后各44根为基础，价格使用对数刻度。</p></div>']
    if deep_count:
        body.append(f'<div class="section">全部深度回调（{deep_count}个）</div>')
        body.extend(blocks[:deep_count])
    body.append(f'<div class="section">边界回调固定抽样（{len(chosen)-deep_count}个）</div>')
    body.extend(blocks[deep_count:])
    body.append('</div>')
    OUT.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>' + mobile_css() + '</style><body>' + ''.join(body) + '</body></html>', encoding='utf-8')
    print(OUT, len(chosen))

if __name__ == "__main__": main()
