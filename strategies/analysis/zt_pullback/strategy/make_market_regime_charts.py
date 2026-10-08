"""生成牛市非正收益、熊市正收益各10个ZT回调样本的多周期长图。"""
from pathlib import Path
import pandas as pd
from make_representative_charts import DAILY, INDEX_FILES, chart_set
from candlestick_svg import aggregate_bars, mobile_css, read_tdx_day

ROOT = Path(__file__).resolve().parents[4]
STRATEGY = ROOT / "strategies" / "analysis" / "zt_pullback"
RESULTS = STRATEGY / "results"
OUT = STRATEGY / "figures" / "market-regime-samples.html"

def main():
    events = pd.read_csv(RESULTS / "market_regime_events.csv", parse_dates=["zt_date", "signal_date", "return_start_date", "return_end_date"])
    metric = "ret_close_to_best_close_25"
    bull = events[events.t25_regime.eq("牛市") & events[metric].le(0)].nsmallest(10, metric).copy()
    bear = events[events.t25_regime.eq("熊市") & events[metric].gt(0)].nlargest(10, metric).copy()
    bull["group"] = "牛市非正收益"; bull["rank"] = range(1, len(bull) + 1)
    bear["group"] = "熊市正收益"; bear["rank"] = range(1, len(bear) + 1)
    chosen = pd.concat([bull, bear], ignore_index=True)
    codes = chosen.ts_code.astype(str).unique().tolist()
    daily = pd.read_parquet(DAILY).query("ts_code in @codes").sort_values(["ts_code", "date"])
    indices = {name: aggregate_bars(read_tdx_day(path), "M") for name, path in INDEX_FILES.items()}
    blocks = [chart_set(row, daily[daily.ts_code.astype(str).eq(row.ts_code)].copy(), indices) for _, row in chosen.iterrows()]
    body = ['<div class="wrap"><div class="head"><h1>ZT回调公式 · 牛熊反例样本</h1><p>按正式T+25路径收益筛选牛市非正收益与熊市正收益样本；最佳卖出点为事后最优。</p></div>', '<div class="section">牛市非正收益</div>']
    body.extend(blocks[:10]); body.append('<div class="section">熊市正收益</div>'); body.extend(blocks[10:]); body.append('</div>')
    OUT.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>' + mobile_css() + '</style><body>' + ''.join(body) + '</body></html>', encoding='utf-8')
    print(OUT)

if __name__ == "__main__": main()
