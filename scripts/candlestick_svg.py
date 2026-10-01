"""通用 A 股多周期 K 线 SVG 绘图工具。

绘图约定：价格使用对数坐标，成交量使用线性坐标；支持日/周/月聚合、
目标前后固定根数窗口、跨周期视窗高亮、事件日虚线、MA5/10/20 和移动端 SVG。
"""
from __future__ import annotations
import html
from pathlib import Path
import numpy as np
import pandas as pd

DAY_DTYPE = np.dtype([("date","<u4"),("open","<u4"),("high","<u4"),("low","<u4"),("close","<u4"),("amount","<f4"),("vol","<u4"),("reserved","<u4")])

def esc(value): return html.escape(str(value), quote=True)

def read_tdx_day(path):
    raw=np.fromfile(Path(path),dtype=DAY_DTYPE)
    return pd.DataFrame({"date":pd.to_datetime(raw["date"].astype(str),format="%Y%m%d"),"open":raw["open"]/100.0,"high":raw["high"]/100.0,"low":raw["low"]/100.0,"close":raw["close"]/100.0,"vol":raw["vol"].astype(np.int64)})

def aggregate_bars(daily, period="D"):
    """聚合日线；周线按周五周期，月线按自然月，并重新计算 MA5/10/20。"""
    x=daily.sort_values("date").copy()
    if period=="D":
        out=x[["date","open","high","low","close","vol"]].copy(); out["period_key"]=out.date.dt.strftime("%Y-%m-%d")
    else:
        freq="W-FRI" if period=="W" else "ME"; key_freq="W-FRI" if period=="W" else "M"
        out=(x.set_index("date").resample(freq).agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),vol=("vol","sum")).dropna(subset=["open","close"]).reset_index())
        out["period_key"]=out.date.dt.to_period(key_freq).astype(str)
    for n in (5,10,20): out[f"ma{n}"]=out.close.rolling(n).mean()
    return out.reset_index(drop=True)

def target_period_key(date, period):
    d=pd.Timestamp(date)
    return d.strftime("%Y-%m-%d") if period=="D" else str(d.to_period("W-FRI" if period=="W" else "M"))

def centered_window(bars,target_date,period="D",before=22,after=22):
    """截取目标周期前后固定根数，默认最多 45 根。"""
    found=bars.index[bars.period_key.eq(target_period_key(target_date,period))]
    if not len(found): return bars.iloc[0:0].copy()
    pos=int(found[0]); return bars.iloc[max(0,pos-before):min(len(bars),pos+after+1)].reset_index(drop=True)

def render_chart(bars,title,subtitle="",target_date=None,signal_date=None,period="D",width=720,height=360,show_volume=True,range_start=None,range_end=None,target_start=None,target_end=None):
    """绘制单个周期 SVG；range_start/end 用于映射低周期完整视窗。"""
    bars=bars.reset_index(drop=True).copy()
    if bars.empty: return f'<svg viewBox="0 0 {width} 100" class="panel"><text x="12" y="50" class="title">{esc(title)}：无可用行情</text></svg>'
    left,right,top,bottom=48,12,62,34; vol_h=54 if show_volume else 0; chart_h=height-top-bottom-vol_h-(8 if show_volume else 0); n=len(bars); xstep=(width-left-right)/max(n,1); xs=left+xstep*(np.arange(n)+.5)
    # Price panels use a logarithmic coordinate system; volume remains linear.
    positive = bars[["low", "high"]].clip(lower=np.finfo(float).tiny)
    lo,hi=float(positive.low.min()),float(positive.high.max())
    log_lo,log_hi=np.log(lo),np.log(hi); pad=max((log_hi-log_lo)*.08,.02); log_lo,log_hi=log_lo-pad,log_hi+pad
    lo,hi=np.exp(log_lo),np.exp(log_hi)
    py=lambda p: top+(log_hi-np.log(max(float(p),np.finfo(float).tiny)))/(log_hi-log_lo)*chart_h
    volmax=max(float(bars.vol.max()),1.0); vol_top=top+chart_h+8; vy=lambda v: vol_top+vol_h-v/volmax*vol_h
    out=[f'<svg viewBox="0 0 {width} {height}" class="panel" role="img" aria-label="{esc(title)}">',f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',f'<text x="12" y="22" class="title">{esc(title)}</text>',f'<text x="12" y="42" class="sub">{esc(subtitle)}</text>']
    for frac in (0,.5,1):
        y=top+chart_h*frac; val=np.exp(log_hi-(log_hi-log_lo)*frac); out += [f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>',f'<text x="4" y="{y+4:.1f}" class="axis">{val:.2f}</text>']
    def locate(date):
        if date is None:return None
        hit=np.flatnonzero(bars.period_key.values==target_period_key(date,period)); return int(hit[0]) if len(hit) else None
    target_idx,signal_idx=locate(target_date),locate(signal_date)
    # A query/candidate can span multiple bars.  Keep target_date for backward
    # compatibility, while target_start/target_end shades the complete interval.
    if target_start is None: target_start=target_date
    if target_end is None: target_end=target_date
    target_positions=[]
    if target_start is not None and target_end is not None:
        ts,te=pd.Timestamp(target_start),pd.Timestamp(target_end)
        for i,bar_date in enumerate(bars.date):
            bd=pd.Timestamp(bar_date)
            if period=="D": bs,be=bd,bd
            elif period=="W": bs,be=bd-pd.Timedelta(days=6),bd
            else: bs,be=bd.to_period("M").start_time,bd.to_period("M").end_time.normalize()
            if bs<=te and be>=ts: target_positions.append(i)
    # Higher-timeframe panels shade the complete lower-timeframe viewport.
    # A weekly bar is represented by its Monday-Friday interval; a monthly bar
    # by its calendar-month interval. This makes the panels read like zoom levels.
    if range_start is not None and range_end is not None:
        rs,re=pd.Timestamp(range_start),pd.Timestamp(range_end)
        for i,bar_date in enumerate(bars.date):
            bd=pd.Timestamp(bar_date)
            if period=="D": bs,be=bd,bd
            elif period=="W": bs,be=bd-pd.Timedelta(days=6),bd
            else: bs,be=bd.to_period("M").start_time,bd.to_period("M").end_time.normalize()
            if bs<=re and be>=rs:
                out.append(f'<rect x="{left+i*xstep:.1f}" y="{top}" width="{xstep:.1f}" height="{chart_h+vol_h+8}" class="viewportband"/>')
    elif target_positions:
        first,last=target_positions[0],target_positions[-1]
        out.append(f'<rect x="{left+first*xstep:.1f}" y="{top}" width="{(last-first+1)*xstep:.1f}" height="{chart_h+vol_h+8}" class="d0band"/>')
    elif target_idx is not None:
        out.append(f'<rect x="{left+target_idx*xstep:.1f}" y="{top}" width="{xstep:.1f}" height="{chart_h+vol_h+8}" class="d0band"/>')
    if signal_idx is not None: out.append(f'<line x1="{xs[signal_idx]:.1f}" x2="{xs[signal_idx]:.1f}" y1="{top}" y2="{vol_top+vol_h}" class="sigline"/>')
    for col,color in (("ma5","#f39c12"),("ma10","#8e44ad"),("ma20","#2980b9")):
        if col not in bars:continue
        pts=" ".join(f"{xs[i]:.1f},{py(v):.1f}" for i,v in enumerate(bars[col].values) if np.isfinite(v)); out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    for i,r in bars.iterrows():
        x=xs[i]; color="#e74c3c" if r.close>=r.open else "#16a085"; out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{py(r.high):.1f}" y2="{py(r.low):.1f}" stroke="{color}" stroke-width="1"/>'); y0,y1=py(max(r.open,r.close)),py(min(r.open,r.close)); out.append(f'<rect x="{x-xstep*.31:.1f}" y="{y0:.1f}" width="{max(xstep*.62,1):.1f}" height="{max(y1-y0,1):.1f}" fill="{color}"/>')
        if show_volume: out.append(f'<rect x="{x-xstep*.31:.1f}" y="{vy(r.vol):.1f}" width="{max(xstep*.62,1):.1f}" height="{max(vol_top+vol_h-vy(r.vol),1):.1f}" fill="{color}" opacity=".45"/>')
    if show_volume:out.append(f'<line x1="{left}" x2="{width-right}" y1="{vol_top}" y2="{vol_top}" class="volline"/>')
    label_y=height-8
    if target_idx is not None:
        target_label = {"D": "D0一字", "W": "D0所在周", "M": "D0所在月"}[period]
        out.append(f'<text x="{xs[target_idx]:.1f}" y="{label_y}" class="mark targetmark" text-anchor="middle">{target_label}</text>')
    if signal_idx is not None and signal_idx!=target_idx: out.append(f'<text x="{xs[signal_idx]:.1f}" y="{label_y}" class="mark signalmark" text-anchor="middle">信号</text>')
    step=max(1,n//6)
    for i in range(0,n,step): out.append(f'<text x="{xs[i]:.1f}" y="{height-22}" class="axis" text-anchor="middle">{bars.iloc[i].date.strftime("%Y-%m" if period=="M" else "%m-%d")}</text>')
    return "".join(out)+"</svg>"

def mobile_css():
    return "body{margin:0;background:#f2f3f5;font-family:Arial,'Microsoft YaHei',sans-serif;color:#222}.wrap{width:min(720px,100vw);margin:0 auto;background:#fff}.head{padding:16px 14px 10px;background:#fff;border-bottom:1px solid #ddd}.head h1{font-size:18px;margin:0 0 6px}.head p{font-size:12px;color:#666;margin:0;line-height:1.5}.section{font-size:15px;font-weight:600;padding:12px;background:#f7f7f7;border-top:8px solid #e8e8e8}.sample{border-top:6px solid #d9dde3}.sample-head{padding:12px 12px 6px;font-size:14px;font-weight:600;background:#fff}.period-label{padding:7px 12px 0;color:#555;font-size:12px;font-weight:600}.panel{display:block;width:100%;height:auto;border-bottom:1px solid #ddd}.title{font-size:14px;font-weight:600;fill:#222}.sub,.axis,.mark{font-size:11px;fill:#666}.targetmark{fill:#9a6700}.signalmark{fill:#315d9e}.grid{stroke:#e6e6e6;stroke-width:1}.d0band{fill:#fff2a8;opacity:.72}.viewportband{fill:#fff2a8;opacity:.45}.sigline{stroke:#315d9e;stroke-width:1.2;stroke-dasharray:4 3}.volline{stroke:#bbb;stroke-width:1}"
