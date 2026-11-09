"""Local browser UI for historical formula screening."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .engine import DEFAULT_DATA, add_forward_returns, collect_events, load_market_data, load_names


PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"


def summarize_returns(result: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5, 10, 20)) -> list[dict]:
    """Summarize forward returns across all matched signals."""
    summary = []
    for horizon in horizons:
        values = result[f"return_{horizon}d"].dropna() if f"return_{horizon}d" in result else pd.Series(dtype=float)
        summary.append({"horizon": horizon,
                        "average": float(values.mean()) if not values.empty else None,
                        "samples": int(values.size)})
    return summary


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>公式历史筛选</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef1f5;color:#1f2937;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}.app{max-width:1180px;margin:0 auto;padding:22px}.card{background:#fff;border:1px solid #dfe5ec;border-radius:12px;box-shadow:0 3px 12px #1f29370d}.header{padding:22px 24px;margin-bottom:14px}.header h1{margin:0 0 5px;font-size:22px}.muted{color:#64748b}.form{padding:18px 20px;display:grid;grid-template-columns:minmax(0,1fr) 170px 170px auto;gap:12px;align-items:end}.field label{display:block;font-size:12px;color:#64748b;margin-bottom:5px}.field input,.field textarea{width:100%;border:1px solid #cbd5e1;border-radius:7px;padding:9px 10px;background:#fff;color:#1f2937;font:inherit}.field textarea{height:124px;resize:vertical;font-family:Consolas,monospace;font-size:13px}.field.formula{grid-column:1/-1}.field input:focus,.field textarea:focus{outline:2px solid #bfdbfe;border-color:#2563eb}.hint{margin-top:6px;font-size:12px;color:#64748b}.btn{height:38px;padding:0 18px;border:0;border-radius:7px;background:#2563eb;color:#fff;cursor:pointer;font-weight:600;white-space:nowrap}.btn:disabled{opacity:.5;cursor:wait}.status{margin:14px 0;padding:10px 12px;border-radius:7px;background:#f1f5f9;color:#475569}.results{padding:16px;min-height:360px}.results h2{font-size:16px;margin:0}.results-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.tablewrap{overflow:auto;margin-top:12px}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:right;border-bottom:1px solid #edf2f7;padding:9px 12px}th{font-size:11px;color:#64748b;background:#f8fafc;position:sticky;top:0}td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}.link{background:none;color:#2563eb;border:0;padding:0;text-decoration:underline;cursor:pointer}.empty{padding:38px 10px;color:#64748b;text-align:center}.chart{padding-top:12px;margin-top:12px;border-top:1px solid #e2e8f0}.chart svg{display:block;width:100%;height:auto;border:1px solid #e2e8f0;background:#fff}.error{color:#b91c1c}.hidden{display:none}@media(max-width:800px){.app{padding:10px}.header{padding:18px}.form{grid-template-columns:1fr 1fr}.field.formula{grid-column:1/-1}.run-field{grid-column:1/-1}.results{padding:12px}.tablewrap{max-height:65vh}.results-head{align-items:flex-start;flex-direction:column}}
.progress-row{display:flex;align-items:center;gap:12px;margin-top:8px}.progress-row.hidden{display:none}.progress-track{height:8px;flex:1;min-width:80px;overflow:hidden;border-radius:5px;background:#dbe3ed}.progress-fill{height:100%;width:0;background:#2563eb;transition:width .2s ease}.progress-value{min-width:42px;text-align:right;font-variant-numeric:tabular-nums;font-weight:600;color:#1d4ed8}
 .status-line{display:flex;justify-content:space-between;gap:12px}.elapsed{font-variant-numeric:tabular-nums;white-space:nowrap;color:#334155}.return-summary{display:flex;flex-wrap:wrap;gap:18px;padding:12px 0;border-bottom:1px solid #e2e8f0}.return-stat{min-width:105px}.return-stat strong{display:block;font-size:18px;font-variant-numeric:tabular-nums;color:#1f2937}.return-stat span{font-size:12px;color:#64748b}.positive{color:#15803d!important}.negative{color:#b91c1c!important}
</style></head><body><main class="app"><section class="card header"><h1>公式历史筛选</h1><div class="muted">输入同花顺或通达信选股公式，在全市场历史日线中定位每次信号，并查看信号后的走势。</div></section>
<section class="card form"><div class="field formula"><label for="formula">选股公式</label><textarea id="formula" spellcheck="false">XG: CROSS(CLOSE, MA(CLOSE, 20));</textarea><div class="hint">推荐将选股条件命名为 XG。非零结果视为命中；公式按每根历史 K 线逐日计算。</div></div><div class="field"><label for="start">信号开始日期</label><input type="date" id="start" value="2016-01-01"></div><div class="field"><label for="end">信号截止日期</label><input type="date" id="end"></div><div class="field run-field"><label>&nbsp;</label><button class="btn" id="run">运行历史筛选</button></div></section>
<div class="status"><div class="status-line"><div id="status">请输入条件后开始筛选。</div><div class="elapsed" id="elapsed">用时 00:00</div></div><div class="progress-row hidden" id="progressRow"><div class="progress-track"><div class="progress-fill" id="progressFill"></div></div><span class="progress-value" id="progressValue">0%</span></div></div><section class="card results"><div class="results-head"><h2>历史信号与后续收益统计</h2><span class="muted" id="count"></span></div><div id="body" class="empty">命中记录会显示在这里；后续收益按所有命中信号汇总。</div><div class="chart hidden" id="chart"></div></section></main>
<script>
const $=id=>document.getElementById(id);let poll=null,currentJob=null,currentPage=0,timer=null,startedAt=0;
function pct(v){return v==null?'—':(v*100).toFixed(2)+'%'}
function elapsedText(){const seconds=Math.floor((Date.now()-startedAt)/1000),hours=Math.floor(seconds/3600),minutes=Math.floor(seconds%3600/60),rest=seconds%60;return '用时 '+(hours?String(hours).padStart(2,'0')+':':'')+String(minutes).padStart(2,'0')+':'+String(rest).padStart(2,'0')}
function stopTimer(){if(timer){clearInterval(timer);timer=null}if(startedAt)$('elapsed').textContent=elapsedText()}
async function run(){if(poll)clearInterval(poll);if(timer)clearInterval(timer);startedAt=Date.now();$('elapsed').textContent=elapsedText();timer=setInterval(()=>$('elapsed').textContent=elapsedText(),1000);$('run').disabled=true;$('status').textContent='正在启动筛选';$('count').textContent='';$('progressRow').classList.remove('hidden');setProgress(0);$('body').className='empty';$('body').textContent='正在读取行情并计算公式';$('chart').classList.add('hidden');try{const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({formula:$('formula').value,start:$('start').value,end:$('end').value})});const j=await r.json();if(!r.ok)throw Error(j.error);poll=setInterval(()=>check(j.job_id),700)}catch(e){stopTimer();$('status').textContent='启动失败：'+e.message;$('count').textContent='';$('progressRow').classList.add('hidden');$('run').disabled=false}}
function setProgress(value){const percent=Math.max(0,Math.min(100,Math.round(Number(value)||0)));$('progressFill').style.width=percent+'%';$('progressValue').textContent=percent+'%'}
async function check(id){try{const r=await fetch('/api/jobs/'+id),j=await r.json();$('status').textContent=j.message||j.status;if(j.status==='running'){setProgress(j.progress);$('progressRow').classList.remove('hidden')}if(j.status==='done'){clearInterval(poll);poll=null;stopTimer();setProgress(100);$('run').disabled=false;currentJob=id;currentPage=0;await loadPage()}if(j.status==='error'){clearInterval(poll);poll=null;stopTimer();$('progressRow').classList.add('hidden');$('run').disabled=false;$('count').textContent='';$('body').innerHTML='<span class="error"></span>';$('body').firstChild.textContent=j.error}}catch(e){}}
async function loadPage(){if(!currentJob)return;const r=await fetch('/api/jobs/'+currentJob+'?page='+currentPage),j=await r.json();render(j.rows||[],j.count||0,j.page||0,j.return_stats||[])}
function render(rows,count,page,stats){const summary='<div class="return-summary">'+stats.map(s=>'<div class="return-stat"><span>'+s.horizon+' 日平均涨幅 · '+s.samples.toLocaleString()+' 只样本</span><strong class="'+(s.average>0?'positive':s.average<0?'negative':'')+'">'+pct(s.average)+'</strong></div>').join('')+'</div>';if(!count){$('body').className='';$('body').innerHTML=summary+'<div class="empty">筛选完成，未找到符合条件的历史信号。</div>';$('count').textContent='0 个信号';return}const cols=['signal_date','ts_code','name','signal_close'];const labels=['信号日期','代码','名称','信号收盘'];let html=summary+'<div class="tablewrap"><table><thead><tr>'+labels.map(x=>'<th>'+x+'</th>').join('')+'<th>走势</th></tr></thead><tbody>';for(const r of rows){html+='<tr>'+cols.map((c,i)=>'<td>'+(i===3?Number(r[c]).toFixed(2):esc(r[c]))+'</td>').join('')+'<td><button class="link" onclick="showChart('+JSON.stringify(r.ts_code)+','+JSON.stringify(r.signal_date)+')">查看</button></td></tr>'}html+='</tbody></table></div><div class="results-head"><span class="muted">第 '+(page+1)+' / '+Math.max(1,Math.ceil(count/200))+' 页 · <a class="link" href="/api/results/'+currentJob+'">下载 CSV</a></span><span><button id="prev" class="btn" '+(page===0?'disabled':'')+'>上一页</button> <button id="next" class="btn" '+((page+1)*200>=count?'disabled':'')+'>下一页</button></span></div>';$('body').className='';$('body').innerHTML=html;$('body').querySelector('#prev').onclick=()=>{currentPage--;loadPage()};$('body').querySelector('#next').onclick=()=>{currentPage++;loadPage()};$('count').textContent=count.toLocaleString()+' 个信号';$('status').textContent=`筛选完成 · ${count.toLocaleString()} 个信号`}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function showChart(symbol,date){const r=await fetch('/api/chart?symbol='+encodeURIComponent(symbol)+'&date='+date);const j=await r.json();if(!r.ok)return;$('chart').innerHTML=j.svg;$('chart').classList.remove('hidden');$('chart').scrollIntoView({behavior:'smooth',block:'nearest'})}
$('end').value=new Date().toISOString().slice(0,10);$('run').addEventListener('click',run);
</script></body></html>'''


class ScreenerApp:
    def __init__(self, data_path: Path = DEFAULT_DATA):
        self.data_path = data_path
        self.jobs: dict[str, dict] = {}

    def start(self, payload: dict) -> str:
        formula = str(payload.get("formula", "")).strip()
        if not formula:
            raise ValueError("请填写通达信选股公式")
        start, end = str(payload.get("start", "")), str(payload.get("end", ""))
        if start:
            date.fromisoformat(start)
        if end:
            date.fromisoformat(end)
        if start and end and start > end:
            raise ValueError("信号开始日期不能晚于截止日期")
        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {"status": "running", "message": "正在载入日线数据", "progress": 0}
        threading.Thread(target=self._run, args=(job_id, formula, start or None, end or None), daemon=True).start()
        return job_id

    def _run(self, job_id: str, formula: str, start: str | None, end: str | None) -> None:
        try:
            market = load_market_data(self.data_path)
            total = market.ts_code.nunique()
            self.jobs[job_id].update(message=f"公式计算中 · 0/{total:,} 只股票", progress=1)
            def progress(done: int, total: int):
                percent = 1 + 90 * done / max(total, 1)
                self.jobs[job_id].update(message=f"公式计算中 · {done:,}/{total:,} 只股票",
                                         progress=percent)
            found = collect_events(market, formula, start, end, progress)
            self.jobs[job_id].update(message="正在整理历史信号和后续收益", progress=94)
            events = pd.DataFrame(found)
            if not events.empty:
                events["date"] = pd.to_datetime(events.date)
                events = events[(events.date <= pd.Timestamp(end)) if end else (events.date <= market.date.max())]
            result = add_forward_returns(events, market)
            names = load_names()
            horizons = (1, 3, 5, 10, 20)
            return_stats = []
            if not result.empty:
                result["name"] = result.ts_code.map(names).fillna("")
                result = result.sort_values(["signal_date", "ts_code"], ascending=[False, True])
                return_stats = summarize_returns(result, horizons)
                result = result.drop(columns=[f"return_{horizon}d" for horizon in horizons])
            else:
                return_stats = summarize_returns(result, horizons)
            RESULTS.mkdir(parents=True, exist_ok=True)
            output = RESULTS / f"signals_{job_id[:8]}.csv"
            self.jobs[job_id].update(message="正在保存筛选结果", progress=98)
            result.to_csv(output, index=False, encoding="utf-8-sig")
            count = len(result)
            self.jobs[job_id] = {"status": "done", "message": f"完成 · {count:,} 个历史信号",
                                 "progress": 100, "count": count, "output": str(output),
                                 "return_stats": return_stats}
        except Exception as exc:
            self.jobs[job_id] = {"status": "error", "message": "筛选失败", "error": str(exc)}


class Handler(BaseHTTPRequestHandler):
    app: ScreenerApp

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = self.app.jobs.get(job_id)
            if not job:
                self._json(404, {"error": "任务不存在"})
            elif job.get("status") == "done":
                try:
                    page = max(0, int(parse_qs(parsed.query).get("page", ["0"])[0]))
                    offset = page * 200
                    if job["count"] == 0:
                        rows = pd.DataFrame()
                    else:
                        rows = pd.read_csv(job["output"], skiprows=range(1, offset + 1), nrows=200)
                    clean = rows.astype(object).where(pd.notna(rows), None)
                    self._json(200, {"status": "done", "count": job["count"], "page": page,
                                     "rows": clean.to_dict("records"),
                                     "return_stats": job.get("return_stats", [])})
                except Exception as exc:
                    self._json(400, {"error": str(exc)})
            else:
                self._json(200, job)
            return
        if parsed.path.startswith("/api/results/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = self.app.jobs.get(job_id)
            if not job or job.get("status") != "done":
                self._json(404, {"error": "结果文件不存在"})
                return
            output = Path(job["output"])
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{output.name}"')
            self.send_header("Content-Length", str(output.stat().st_size))
            self.end_headers()
            with output.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)
            return
        if parsed.path == "/api/chart":
            query = parse_qs(parsed.query)
            symbol, signal_date = query.get("symbol", [""])[0], query.get("date", [""])[0]
            try:
                frame = pd.read_parquet(self.app.data_path, filters=[("ts_code", "==", symbol)])
                frame["date"] = pd.to_datetime(frame.date)
                center = pd.Timestamp(signal_date)
                ix = frame.date.searchsorted(center)
                view = frame.iloc[max(0, ix - 35):min(len(frame), ix + 46)]
                import sys
                scripts = str(Path(__file__).resolve().parents[2] / "scripts")
                if scripts not in sys.path:
                    sys.path.insert(0, scripts)
                from candlestick_svg import aggregate_bars, render_chart
                from strategies.kline_similarity.visualize import standalone_svg
                bars = aggregate_bars(view, "D")
                chart = standalone_svg(render_chart(
                    bars, f"{symbol} 日线", f"信号日期 {signal_date} · 前复权", 
                    signal_date=signal_date, period="D", width=980, height=390,
                ))
                self._json(200, {"svg": chart})
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/search":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            job_id = self.app.start(payload)
            self._json(202, {"job_id": job_id})
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="通达信公式历史筛选")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    Handler.app = ScreenerApp(args.data)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Formula Screener: {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
