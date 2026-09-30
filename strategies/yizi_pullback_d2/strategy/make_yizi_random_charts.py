"""生成随机正收益/负收益样本的多周期 K 线长图。"""
import os
import sys
import pandas as pd

STRATEGY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(STRATEGY_ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_yizi_mobile_charts import chart_set, INDEX_FILES  # noqa: E402
from candlestick_svg import aggregate_bars, read_tdx_day, mobile_css  # noqa: E402

EVENTS = os.path.join(STRATEGY_ROOT, "results", "yizi_pullback_d2_events.csv")
DAILY = os.path.join(ROOT, "database", "processed", "stock_daily.parquet")
OUT = os.path.join(STRATEGY_ROOT, "figures", "yizi-random-positive-negative-k-lines.html")
SEED = 20260930


def main():
    events = pd.read_csv(EVENTS, parse_dates=["yizi_date", "d1_date", "signal_date"])
    valid = events.dropna(subset=["ret_open_20"]).copy()
    positive = valid[valid.ret_open_20 > 0].sample(n=min(50, (valid.ret_open_20 > 0).sum()), random_state=SEED).copy()
    negative = valid[valid.ret_open_20 <= 0].sample(n=min(50, (valid.ret_open_20 <= 0).sum()), random_state=SEED).copy()
    positive["group"] = "随机正收益"; positive["rank"] = range(1, len(positive) + 1)
    negative["group"] = "随机负收益"; negative["rank"] = range(1, len(negative) + 1)
    chosen = pd.concat([positive, negative], ignore_index=True)
    codes = chosen.ts_code.astype(str).unique().tolist()
    daily = pd.read_parquet(DAILY, filters=[("ts_code", "in", codes)]).sort_values(["ts_code", "date"])
    index_monthly = {name: aggregate_bars(read_tdx_day(path), "M") for name, path in INDEX_FILES.items()}
    blocks = []
    for _, row in chosen.iterrows():
        stock = daily[daily.ts_code.astype(str).eq(row.ts_code)].copy()
        blocks.append(chart_set(row, stock, index_monthly))
    body = ['<div class="wrap"><div class="head"><h1>随机正收益与负收益样本</h1><p>当前正式策略｜固定随机种子 20260930｜次日开盘入场｜持有20根有效K线。</p><p>每个样本日线前后各44根（约89根），周线、个股月线、创业板指月线和上证指数月线各45根；价格坐标为对数坐标。</p></div>']
    body += ['<div class="section">随机正收益样本（50个）</div>'] + blocks[:len(positive)]
    body += ['<div class="section">随机负收益样本（50个）</div>'] + blocks[len(positive):] + ['</div>']
    doc = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>' + mobile_css() + '</style><body>' + ''.join(body) + '</body></html>'
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print(OUT, len(chosen), len(doc))


if __name__ == "__main__":
    main()
