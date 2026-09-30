"""生成一字板样本的日/周/月/指数月线手机长图。"""
import os, sys
import pandas as pd

STRATEGY_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT=os.path.dirname(os.path.dirname(STRATEGY_ROOT))
sys.path.insert(0,os.path.join(ROOT,"scripts"))
from candlestick_svg import aggregate_bars, centered_window, mobile_css, read_tdx_day, render_chart

EVENTS=os.path.join(STRATEGY_ROOT,"results","yizi_pullback_d2_events.csv")
DAILY=os.path.join(ROOT,"database","processed","stock_daily.parquet")
RAW=os.path.join(ROOT,"database","raw","tdx")
OUT=os.path.join(STRATEGY_ROOT,"figures","yizi-mobile-k-lines.html")
INDEX_FILES={"创业板指":os.path.join(RAW,"sz","lday","sz399006.day"),"上证指数":os.path.join(RAW,"sh","lday","sh000001.day")}

def pick_events(events):
    events=events.dropna(subset=["ret_open_20"]).copy(); mean=events.ret_open_20.mean(); events["mean_gap"]=(events.ret_open_20-mean).abs()
    groups=[("高收益",events.nlargest(10,"ret_open_20")),("平均收益",events.nsmallest(10,"mean_gap")),("低收益",events.nsmallest(10,"ret_open_20"))]; rows=[]
    for label,part in groups:
        for rank,(_,row) in enumerate(part.sort_values("ret_open_20",ascending=label!="低收益").iterrows(),1): rows.append({"group":label,"rank":rank,**row.to_dict()})
    return pd.DataFrame(rows),mean

def chart_set(row,stock_daily,index_monthly):
    d0,sig=pd.Timestamp(row.yizi_date),pd.Timestamp(row.signal_date); subtitle=f"D0价 {row.yizi_close:.2f} · 信号价 {row.close:.2f} · 回调 {row.drop_pct:+.2%} · D3开盘20根 {row.ret_open_20:+.2%} · 超额 {row.ex_open_20:+.2%}"; pieces=[f'<div class="sample"><div class="sample-head">{row.group} {int(row["rank"]):02d} · {row.ts_code} · D0 {d0.date()} · 信号 {sig.date()}</div>']
    daily_window=centered_window(aggregate_bars(stock_daily,"D"),d0,"D",44,44)
    weekly_window=centered_window(aggregate_bars(stock_daily,"W"),d0,"W",22,22)
    monthly_window=centered_window(aggregate_bars(stock_daily,"M"),d0,"M",22,22)
    daily_range=(daily_window.date.min(),daily_window.date.max())
    weekly_range=(weekly_window.date.min()-pd.Timedelta(days=6),weekly_window.date.max())
    pieces += [f'<div class="period-label">日线 · 前后各44根（约89根）</div>',render_chart(daily_window,f"{row.ts_code} 日线 · 前后各44根",subtitle,d0,sig,"D")]
    pieces += [f'<div class="period-label">周线 · 黄色范围对应上方日线图的完整视窗</div>',render_chart(weekly_window,f"{row.ts_code} 周线 · 45根",subtitle,d0,sig,"W",range_start=daily_range[0],range_end=daily_range[1])]
    pieces += [f'<div class="period-label">个股月线 · 黄色范围对应上方日线图的完整视窗</div>',render_chart(monthly_window,f"{row.ts_code} 个股月线 · 45根",subtitle,d0,sig,"M",range_start=daily_range[0],range_end=daily_range[1])]
    for name,bars in index_monthly.items():
        window=centered_window(bars,d0,"M",22,22); pieces += [f'<div class="period-label">{name}月线 · 黄色范围对应个股日线图的完整视窗</div>',render_chart(window,f"{name} 月线 · 45根",f"同步显示 {row.ts_code} 日线视窗所在月份",d0,sig,"M",range_start=daily_range[0],range_end=daily_range[1])]
    return "".join(pieces)+"</div>"

def main():
    events=pd.read_csv(EVENTS,parse_dates=["yizi_date","d1_date","signal_date"]); chosen,mean=pick_events(events); codes=chosen.ts_code.astype(str).unique().tolist(); daily=pd.read_parquet(DAILY,filters=[("ts_code","in",codes)]).sort_values(["ts_code","date"]); index_monthly={name:aggregate_bars(read_tdx_day(path),"M") for name,path in INDEX_FILES.items()}; blocks=[]
    for _,row in chosen.iterrows(): blocks.append(chart_set(row,daily[daily.ts_code.astype(str).eq(row.ts_code)].copy(),index_monthly))
    body=[f'<div class="wrap"><div class="head"><h1>一字板回调形态 · 多周期样本长图</h1><p>日线展示目标前后各44根（约89根）；周线、个股月线、创业板指月线和上证指数月线仍展示目标前后各22根（45根）。</p><p>所有价格坐标均为对数坐标；成交量保持线性坐标。周线、月线黄色范围均对应日线完整视窗｜蓝色虚线=信号所在周期｜全样本D3开盘20根均值 {mean:+.2%}</p></div>']; offset=0
    for label,count in (("高收益样本（10）",10),("平均收益样本（10）",10),("低收益样本（10）",10)): body.append(f'<div class="section">{label}</div>'); body.extend(blocks[offset:offset+count]); offset+=count
    body.append("</div>"); doc='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>'+mobile_css()+'</style><body>'+"".join(body)+"</body></html>"; os.makedirs(os.path.dirname(OUT),exist_ok=True); open(OUT,"w",encoding="utf-8").write(doc); print(OUT,len(blocks),len(doc))

if __name__=="__main__": main()
