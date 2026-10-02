"""生成一字板样本的日/周/月/指数月线手机长图。"""
import os, sys
import pandas as pd

STRATEGY_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(STRATEGY_ROOT)))
sys.path.insert(0,os.path.join(ROOT,"scripts"))
from candlestick_svg import aggregate_bars, centered_window, mobile_css, read_tdx_day, render_chart

EVENTS=os.path.join(STRATEGY_ROOT,"results","events.csv")
DAILY=os.path.join(ROOT,"database","processed","stock_daily.parquet")
RAW=os.path.join(ROOT,"database","raw","tdx")
OUT=os.path.join(STRATEGY_ROOT,"figures","representative-samples.html")
INDEX_FILES={"创业板指":os.path.join(RAW,"sz","lday","sz399006.day"),"上证指数":os.path.join(RAW,"sh","lday","sh000001.day")}

def pick_events(events):
    metric="ret_close_to_best_close_25"
    events=events.dropna(subset=[metric]).copy(); mean=events[metric].mean(); events["mean_gap"]=(events[metric]-mean).abs()
    groups=[("高收益",events.nlargest(10,metric)),("平均收益",events.nsmallest(10,"mean_gap")),("低收益",events.nsmallest(10,metric))]; rows=[]
    for label,part in groups:
        for rank,(_,row) in enumerate(part.sort_values(metric,ascending=label!="低收益").iterrows(),1): rows.append({"group":label,"rank":rank,**row.to_dict()})
    return pd.DataFrame(rows),mean

def path_daily_window(stock_daily, d0, row):
    bars = aggregate_bars(stock_daily, "D")
    window = centered_window(bars, d0, "D", 44, 44)
    if pd.isna(row.return_start_date) or pd.isna(row.return_end_date):
        return window
    positions = bars.index[bars.date.isin(pd.to_datetime([row.return_start_date, row.return_end_date]))]
    if len(positions) != len({pd.Timestamp(row.return_start_date), pd.Timestamp(row.return_end_date)}):
        raise ValueError(f"Path endpoints missing from daily bars: {row.ts_code} {d0:%Y-%m-%d}")
    target = int(bars.index[bars.date.eq(d0)][0])
    first = max(0, min(target - 44, int(positions.min()) - 3))
    last = min(len(bars), max(target + 45, int(positions.max()) + 4))
    return bars.iloc[first:last].reset_index(drop=True)

def path_markers(row, stock_daily):
    if pd.isna(row.return_start_date) or pd.isna(row.return_end_date):
        return []
    bars = stock_daily.set_index("date")
    return [
        {"date": row.return_start_date, "price": bars.loc[pd.Timestamp(row.return_start_date), "close"], "side": "buy"},
        {"date": row.return_end_date, "price": bars.loc[pd.Timestamp(row.return_end_date), "close"], "side": "sell"},
    ]

def chart_set(row,stock_daily,index_monthly):
    d0,sig=pd.Timestamp(row.yizi_date),pd.Timestamp(row.signal_date)
    subtitle=f"D0价 {row.yizi_close:.2f} · 信号价 {row.close:.2f} · 回调 {row.drop_pct:+.2%}"
    if pd.notna(row.return_start_date) and pd.notna(row.return_end_date):
        subtitle+=f" · D1-D3买入 {pd.Timestamp(row.return_start_date).date()} 后复权收盘{row.return_start_close:.4f} → T+1~25卖出 {pd.Timestamp(row.return_end_date).date()} 后复权收盘{row.return_end_close:.4f} · 区间收益 {row.ret_close_to_best_close_25:+.2%} · 超额 {row.ex_close_to_best_close_25:+.2%}"
    else:
        subtitle+=" · 无有效买点，不计算T+25收益"
    pieces=[f'<div class="sample"><div class="sample-head">{row.group} {int(row["rank"]):02d} · {row.ts_code} · D0 {d0.date()} · 信号 {sig.date()}</div>']
    daily_window=path_daily_window(stock_daily,d0,row)
    weekly_window=centered_window(aggregate_bars(stock_daily,"W"),d0,"W",22,22)
    monthly_window=centered_window(aggregate_bars(stock_daily,"M"),d0,"M",22,22)
    daily_range=(daily_window.date.min(),daily_window.date.max())
    weekly_range=(weekly_window.date.min()-pd.Timedelta(days=6),weekly_window.date.max())
    pieces += [f'<div class="period-label">日线 · D0前后各44根，必要时延伸至收益起止点</div>',render_chart(daily_window,f"{row.ts_code} 日线 · 路径起止点",subtitle,d0,sig,"D",price_markers=path_markers(row,stock_daily))]
    pieces += [f'<div class="period-label">周线 · 黄色范围对应上方日线图的完整视窗</div>',render_chart(weekly_window,f"{row.ts_code} 周线 · 45根",subtitle,d0,sig,"W",range_start=daily_range[0],range_end=daily_range[1])]
    pieces += [f'<div class="period-label">个股月线 · 黄色范围对应上方日线图的完整视窗</div>',render_chart(monthly_window,f"{row.ts_code} 个股月线 · 45根",subtitle,d0,sig,"M",range_start=daily_range[0],range_end=daily_range[1])]
    for name,bars in index_monthly.items():
        window=centered_window(bars,d0,"M",22,22); pieces += [f'<div class="period-label">{name}月线 · 黄色范围对应个股日线图的完整视窗</div>',render_chart(window,f"{name} 月线 · 45根",f"同步显示 {row.ts_code} 日线视窗所在月份",d0,sig,"M",range_start=daily_range[0],range_end=daily_range[1])]
    return "".join(pieces)+"</div>"

def main():
    events=pd.read_csv(EVENTS,parse_dates=["yizi_date","d1_date","signal_date"]); chosen,mean=pick_events(events); codes=chosen.ts_code.astype(str).unique().tolist(); daily=pd.read_parquet(DAILY,filters=[("ts_code","in",codes)]).sort_values(["ts_code","date"]); index_monthly={name:aggregate_bars(read_tdx_day(path),"M") for name,path in INDEX_FILES.items()}; blocks=[]
    for _,row in chosen.iterrows(): blocks.append(chart_set(row,daily[daily.ts_code.astype(str).eq(row.ts_code)].copy(),index_monthly))
    body=[f'<div class="wrap"><div class="head"><h1>一字板回调形态 · 多周期样本长图</h1><p>日线以D0前后各44根为基础，必要时延伸至买卖点；周线、个股月线、创业板指月线和上证指数月线仍各展示45根。</p><p>红色为D1-D3最低价K线的收盘买入，蓝色为买入后T+1至T+25最高收盘卖出；最高卖点属于事后最优，不代表可预知的交易结果。价格使用对数刻度｜蓝色虚线=信号所在周期｜区间收益均值 {mean:+.2%}</p></div>']; offset=0
    for label,count in (("高收益样本（10）",10),("平均收益样本（10）",10),("低收益样本（10）",10)): body.append(f'<div class="section">{label}</div>'); body.extend(blocks[offset:offset+count]); offset+=count
    body.append("</div>"); doc='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><style>'+mobile_css()+'</style><body>'+"".join(body)+"</body></html>"; os.makedirs(os.path.dirname(OUT),exist_ok=True); open(OUT,"w",encoding="utf-8").write(doc); print(OUT,len(blocks),len(doc))

if __name__=="__main__": main()
