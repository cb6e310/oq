"""生成牛市负收益、熊市正收益各10个样本的多周期 K 线长图。"""
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]; STRATEGY=ROOT/"strategies"/"analysis"/"yizi_pullback"; RESULTS=STRATEGY/"results"; OUT=STRATEGY/"figures"/"market-regime-samples.html"; DAILY=ROOT/"database"/"processed"/"stock_daily.parquet"; RAW=ROOT/"database"/"raw"/"tdx"
sys.path.insert(0,str(ROOT/"scripts")); from candlestick_svg import aggregate_bars,centered_window,mobile_css,read_tdx_day,render_chart
sys.path.insert(0,str(Path(__file__).resolve().parent)); from make_representative_charts import path_daily_window,path_markers
INDEX={"创业板指":RAW/"sz"/"lday"/"sz399006.day","上证指数":RAW/"sh"/"lday"/"sh000001.day"}

def pick(d):
    metric = "ret_close_to_best_close_25"
    return pd.concat([d[d.t25_regime.eq("牛市") & d[metric].le(0)].nsmallest(10,metric).assign(group="牛市非正收益"),d[d.t25_regime.eq("熊市") & d[metric].gt(0)].nlargest(10,metric).assign(group="熊市正收益")],ignore_index=True)

def one(row,stock,indices):
    d0,sig=pd.Timestamp(row.yizi_date),pd.Timestamp(row.signal_date); sub=f"D1-D3买入收盘 {row.return_start_close:.4f} · T+25最佳卖出收盘 {row.return_end_close:.4f} · 区间收益 {row.ret_close_to_best_close_25:+.2%}"; daily=path_daily_window(stock,d0,row); dr=(daily.date.min(),daily.date.max()); parts=[f'<div class="sample"><div class="sample-head">{row.group} · {row.ts_code} · D0 {d0.date()} · 信号 {sig.date()}</div>']
    for p,label in (("D","日线"),("W","周线"),("M","个股月线")):
        bars=aggregate_bars(stock,p); w=daily if p=="D" else centered_window(bars,d0,p,22,22); parts += [f'<div class="period-label">{label} · {"D0前后各44根，必要时延伸至收益起止点" if p=="D" else "45根"}</div>',render_chart(w,f"{row.ts_code} {label}",sub,d0,sig,p,range_start=dr[0] if p!="D" else None,range_end=dr[1] if p!="D" else None,price_markers=path_markers(row,stock) if p=="D" else None)]
    for name,bars in indices.items():
        w=centered_window(bars,d0,"M",22,22); parts += [f'<div class="period-label">{name}月线 · 日线完整视窗</div>',render_chart(w,f"{name}月线",sub,d0,sig,"M",range_start=dr[0],range_end=dr[1])]
    return "".join(parts)+"</div>"

def main():
    d=pd.read_csv(RESULTS/"market_regime_events.csv",parse_dates=["yizi_date","signal_date","t25_date"]); chosen=pick(d); codes=chosen.ts_code.astype(str).unique().tolist(); daily=pd.read_parquet(DAILY,filters=[("ts_code","in",codes)]).sort_values(["ts_code","date"]); indices={k:aggregate_bars(read_tdx_day(v),"M") for k,v in INDEX.items()}; blocks=[]
    for _,r in chosen.iterrows(): blocks.append(one(r,daily[daily.ts_code.astype(str).eq(r.ts_code)].copy(),indices))
    body=['<div class="wrap"><div class="head"><h1>牛市非正收益与熊市正收益样本（正式T+25口径）</h1><p>本页按D1-D3最低价K线收盘买入、T+1至T+25最高收盘卖出的正式收益筛选牛熊反例；最佳卖出日为事后最优。</p><p>日线以D0前后各44根为基础，必要时延伸至买卖点，周线、个股月线、创业板指月线和上证指数月线各45根。</p></div>']; body += ['<div class="section">牛市非正收益（最多10个）</div>']+blocks[:10]+['<div class="section">熊市正收益（最多10个）</div>']+blocks[10:]+['</div>']; OUT.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><style>'+mobile_css()+'</style><body>'+''.join(body)+'</body></html>',encoding='utf-8'); print(OUT)
if __name__=="__main__": main()
