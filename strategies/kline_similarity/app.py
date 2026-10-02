"""Local browser UI for K-line similarity search.

Run from the repository root with ``python -m strategies.kline_similarity.app``.
The UI uses only Python's standard library, so no web framework is required.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .engine import KlineSimilarityEngine, ParquetDataProvider, SimilarityConfig
from .visualize import build_visual_results, standalone_svg


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "database" / "processed" / "stock_daily_qfq.parquet"
DEFAULT_BASIC = ROOT / "database" / "processed" / "stock_basic_tdx.parquet"
DEFAULT_TDX_NAMES = ROOT / "database" / "raw" / "tdx" / "meta" / "infoharbor_ex.code"
DEFAULT_OUTPUT = ROOT / "strategies" / "kline_similarity" / "ui_results"
HISTORY_FILE = ROOT / "strategies" / "kline_similarity" / "query_history.json"


def load_stock_names(basic_path: Path = DEFAULT_BASIC, tdx_path: Path = DEFAULT_TDX_NAMES) -> list[dict]:
    """Load code/name/exchange metadata, preferring TDX's GBK name file."""
    exchange: dict[str, str] = {}
    if basic_path.exists():
        basic = pd.read_parquet(basic_path, columns=["symbol", "exchange", "ts_code"])
        exchange = {str(row.symbol).zfill(6): str(row.exchange) for row in basic.itertuples()}
    names: dict[str, str] = {}
    if tdx_path.exists():
        for line in tdx_path.read_bytes().decode("gbk", errors="replace").splitlines():
            parts = line.split("|")
            if len(parts) >= 2 and len(parts[0]) == 6 and parts[0].isdigit():
                names[parts[0]] = parts[1].strip()
    rows = []
    for symbol, name in names.items():
        ex = exchange.get(symbol)
        if not ex:
            ex = "SH" if symbol.startswith(("6", "68")) else "BJ" if symbol.startswith(("4", "8")) else "SZ"
        rows.append({"symbol": symbol, "name": name, "exchange": ex, "ts_code": f"{symbol}.{ex}"})
    return sorted(rows, key=lambda x: x["symbol"])


def resolve_symbol(value: str, stocks: list[dict]) -> dict:
    query = str(value).strip().upper()
    if "." in query:
        for stock in stocks:
            if stock["ts_code"] == query:
                return stock
    digits = query.zfill(6) if query.isdigit() else query
    exact = [x for x in stocks if x["symbol"] == digits or x["name"] == value.strip()]
    if exact:
        return exact[0]
    contains = [x for x in stocks if digits in x["symbol"] or query in x["name"].upper()]
    if len(contains) == 1:
        return contains[0]
    raise ValueError(f"无法唯一识别股票：{value}")


class SimilarityApp:
    def __init__(self, data_path: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUTPUT):
        self.provider = ParquetDataProvider(data_path)
        self.engine = KlineSimilarityEngine(self.provider, SimilarityConfig(top_k=20, recall_n=1000))
        self.stocks = load_stock_names()
        self.stock_by_code = {x["symbol"]: x for x in self.stocks}
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def history(self) -> list[dict]:
        if not HISTORY_FILE.exists():
            return []
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def save_history(self, item: dict) -> None:
        records = self.history()
        records = [x for x in records if x.get("id") != item.get("id")]
        records.insert(0, item)
        HISTORY_FILE.write_text(json.dumps(records[:50], ensure_ascii=False, indent=2), encoding="utf-8")

    def submit(self, payload: dict) -> str:
        stock = resolve_symbol(payload.get("stock", ""), self.stocks)
        timeframe = str(payload.get("timeframe", "1d"))
        if timeframe not in {"1d", "1w", "1m"}:
            raise ValueError("周期必须是 1d、1w 或 1m")
        start, end = pd.Timestamp(payload.get("start")), pd.Timestamp(payload.get("end"))
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("日期区间无效")
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "status": "queued", "message": "等待执行"}
        thread = threading.Thread(target=self._run, args=(job_id, stock, start, end, timeframe), daemon=True)
        thread.start()
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a queued/running job.

        The search worker is deliberately not force-killed because it may be
        inside pandas/NumPy code.  It checks this flag before publishing
        results, so a cancelled search never appears in history or replaces
        the current result panel.
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job.get("status") in {"done", "error", "cancelled"}:
                return False
            job.update(status="cancelled", message="已取消搜索")
            return True

    def _cancelled(self, job_id: str) -> bool:
        with self.lock:
            return self.jobs.get(job_id, {}).get("status") == "cancelled"

    def _run(self, job_id: str, stock: dict, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> None:
        try:
            with self.lock:
                if self.jobs.get(job_id, {}).get("status") == "cancelled":
                    return
                self.jobs[job_id].update(status="running", message="正在扫描全市场历史行情…")
            folder = self.output_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}"
            results = build_visual_results(self.engine, stock["ts_code"], start, end,
                                           timeframe=timeframe, output_dir=folder, top_k=20,
                                           history_only=True)
            if self._cancelled(job_id):
                return
            public_results = []
            for item in results:
                item = dict(item)
                item["image_paths"] = {key: _web_path(Path(value)) for key, value in item["image_paths"].items()}
                item["image_path"] = item["image_paths"]["daily"]
                item["html_path"] = _web_path(Path(item["html_path"]))
                public_results.append(item)
            record = {"id": job_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "stock": stock, "start": start.strftime("%Y-%m-%d"),
                      "end": end.strftime("%Y-%m-%d"), "timeframe": timeframe,
                      "result_count": len(public_results), "results": public_results,
                      "html_path": public_results[0]["html_path"] if public_results else "",
                      "search_stats": dict(self.engine.last_search_stats)}
            if self._cancelled(job_id):
                return
            self.save_history(record)
            with self.lock:
                self.jobs[job_id].update(status="done", message="完成", record=record)
        except Exception as exc:  # surface errors in UI rather than killing worker
            with self.lock:
                self.jobs[job_id].update(status="error", message=str(exc))


def _web_path(path: Path) -> str:
    try:
        return "/files/" + path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>K线历史相似形态搜索</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#eef1f5;color:#1f2937;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}.app{max-width:1180px;margin:0 auto;padding:22px}.card{background:#fff;border:1px solid #dfe5ec;border-radius:12px;box-shadow:0 3px 12px #1f29370d}.header{padding:22px 24px;margin-bottom:14px}.header h1{margin:0 0 5px;font-size:22px}.muted{color:#64748b}.form{padding:18px 20px;display:grid;grid-template-columns:2fr 1fr 1fr 1fr auto auto;gap:12px;align-items:end}.field label{display:block;font-size:12px;color:#64748b;margin-bottom:5px}.field input,.field select{width:100%;height:38px;border:1px solid #cbd5e1;border-radius:7px;padding:0 10px;background:#fff}.stockbox{position:relative}.suggestions{position:absolute;z-index:5;left:0;right:0;top:64px;background:#fff;border:1px solid #cbd5e1;border-radius:7px;max-height:230px;overflow:auto}.suggestion{padding:8px 10px;cursor:pointer}.suggestion:hover{background:#eff6ff}.btn{height:38px;padding:0 18px;border:0;border-radius:7px;background:#2563eb;color:#fff;cursor:pointer;font-weight:600}.btn.cancel{background:#dc2626}.btn:disabled{opacity:.5;cursor:wait}.status{margin:14px 20px;padding:10px 12px;border-radius:7px;background:#f1f5f9;color:#475569}.layout{display:grid;grid-template-columns:265px 1fr;gap:14px;margin-top:14px}.history{padding:15px}.history h2,.results h2{font-size:16px;margin:0 0 10px}.history-item{padding:10px 8px;border-bottom:1px solid #edf2f7;cursor:pointer}.history-item:hover{background:#f8fafc}.results{padding:16px}.result{border:1px solid #e2e8f0;border-radius:10px;margin:10px 0;overflow:hidden}.result-head{padding:8px 12px;display:grid;grid-template-columns:minmax(220px,1fr) 260px 145px;gap:12px;align-items:center;cursor:pointer;background:#f8fafc}.result-meta{min-width:0}.result-score{text-align:right;white-space:nowrap}.thumb{width:260px;height:104px;object-fit:cover;object-position:center;border:1px solid #dbe3ec;border-radius:6px;background:#fff;margin:0}.badge{color:#1d4ed8;font-weight:700}.result-body{padding:12px;display:none}.result.open .result-body{display:block}.result-body img{display:block;width:100%;border:1px solid #e2e8f0;margin:8px 0;background:#fff}.empty{padding:38px 10px;color:#64748b;text-align:center}@media(max-width:800px){.form{grid-template-columns:1fr 1fr}.form .stockbox{grid-column:1/-1}.layout{grid-template-columns:1fr}.result-head{grid-template-columns:1fr}.thumb{width:100%;height:130px}.result-score{text-align:left}.header{padding:18px}.app{padding:10px}}
</style></head><body><div class="app">
<div class="card header"><h1>K线历史相似形态搜索</h1><div class="muted">全市场历史相似区间 · 日线前后90根 · 周/月线前后22根 · 支持过往查询记录</div></div>
<div class="card form"><div class="field stockbox"><label>股票名称或代码</label><input id="stock" placeholder="如：601567 或名称" autocomplete="off"><div id="suggestions" class="suggestions" hidden></div></div><div class="field"><label>周期</label><select id="timeframe"><option value="1d">日 K</option><option value="1w">周 K</option><option value="1m">月 K</option></select></div><div class="field"><label>开始日期</label><input id="start" type="date"></div><div class="field"><label>结束日期</label><input id="end" type="date"></div><button id="search" class="btn">开始搜索</button><button id="cancel" class="btn cancel" hidden>取消搜索</button></div>
<div id="status" class="status">请输入条件后开始搜索。</div><div class="layout"><aside class="card history"><h2>过往查询</h2><div id="history" class="empty">暂无记录</div></aside><main class="card results"><h2>相似度 Top20</h2><div id="results" class="empty">搜索结果会显示在这里。</div></main></div></div>
<script>
const $=id=>document.getElementById(id), stock=$('stock'), sugg=$('suggestions'), status=$('status'), results=$('results'), searchButton=$('search'), cancelButton=$('cancel');
let pollTimer=null, elapsedTimer=null, searchStartedAt=0, activeJobId=null;
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function stocks(q){if(!q){sugg.hidden=true;return}const r=await fetch('/api/stocks?q='+encodeURIComponent(q));const xs=await r.json();sugg.innerHTML=xs.map(x=>`<div class="suggestion" data-value="${esc(x.ts_code)}"><b>${esc(x.symbol)}</b> ${esc(x.name)} <span class="muted">${esc(x.exchange)}</span></div>`).join('');sugg.hidden=!xs.length;sugg.querySelectorAll('.suggestion').forEach(x=>x.onclick=()=>{stock.value=x.dataset.value;sugg.hidden=true})}
stock.oninput=()=>stocks(stock.value);document.onclick=e=>{if(!e.target.closest('.stockbox'))sugg.hidden=true};
function renderRecord(record){const xs=record.results||[];results.innerHTML=xs.length?xs.map(x=>`<div class="result"><div class="result-head"><div class="result-meta"><span class="badge">#${x.rank}</span> ${esc(x.symbol)}<br><span class="muted">${esc(x.start_date)} ~ ${esc(x.end_date)}</span></div><img class="thumb" loading="lazy" src="${x.image_paths.monthly}" alt="${esc(x.symbol)} 月线缩略图"><div class="result-score">相似度 ${Number(x.score).toFixed(4)}<br><span class="muted">点击展开</span></div></div><div class="result-body"><div class="muted">特殊板事件距离 ${Number(x.event_distance||0).toFixed(4)} · DTW ${Number(x.dtw_distance).toFixed(4)}</div><img loading="lazy" src="${x.image_paths.daily}" alt="日线"><img loading="lazy" src="${x.image_paths.weekly}" alt="周线"><img loading="lazy" src="${x.image_paths.monthly}" alt="月线"></div></div>`).join(''):'<div class="empty">没有找到可展示的历史候选。</div>';results.querySelectorAll('.result-head').forEach(x=>x.onclick=()=>x.parentElement.classList.toggle('open'))}
function elapsedText(){return Math.max(0,Math.floor((Date.now()-searchStartedAt)/1000))+' 秒'}
function setSearching(value){searchButton.disabled=value;searchButton.textContent=value?'搜索中…':'开始搜索';cancelButton.hidden=!value;if(!value){clearTimeout(pollTimer);clearInterval(elapsedTimer);pollTimer=null;elapsedTimer=null;activeJobId=null}}
function showProgress(message){status.textContent=(message||'正在搜索')+' · 已用时 '+elapsedText()}
function finishSearch(message){const elapsed=elapsedText();setSearching(false);status.textContent=message+' · 用时 '+elapsed}
async function poll(id){
  if(id!==activeJobId)return;
  try{
    const r=await fetch('/api/jobs/'+id),j=await r.json();
    if(!r.ok)throw new Error(j.message||'任务状态读取失败');
    if(j.status==='done'){renderRecord(j.record);finishSearch('搜索完成');await loadHistory()}
    else if(j.status==='error'){finishSearch('搜索失败：'+(j.message||'未知错误'))}
    else{showProgress(j.message||j.status);pollTimer=setTimeout(()=>poll(id),1000)}
  }catch(err){finishSearch('搜索失败：'+err.message)}
}
searchButton.onclick=async()=>{
  if(searchButton.disabled)return;
  sugg.hidden=true;searchStartedAt=Date.now();setSearching(true);showProgress('提交搜索…');
  elapsedTimer=setInterval(()=>showProgress('正在扫描全市场历史行情…'),1000);
  results.innerHTML='<div class="empty">正在扫描全市场，请稍候…</div>';
  try{
    const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stock:stock.value,timeframe:$('timeframe').value,start:$('start').value,end:$('end').value})});
    const j=await r.json();
    if(!r.ok)throw new Error(j.error||'提交搜索失败');
    activeJobId=j.id;poll(j.id);
  }catch(err){finishSearch('输入或网络错误：'+err.message)}
};
cancelButton.onclick=async()=>{
  const id=activeJobId;
  if(!id)return;
  clearTimeout(pollTimer);clearInterval(elapsedTimer);pollTimer=null;elapsedTimer=null;activeJobId=null;
  try{await fetch('/api/jobs/'+id+'/cancel',{method:'POST'})}catch(_err){}
  setSearching(false);status.textContent='已取消搜索 · 用时 '+elapsedText();
};
async function loadHistory(){const r=await fetch('/api/history'),xs=await r.json();$('history').innerHTML=xs.length?xs.map(x=>`<div class="history-item" data-id="${x.id}"><b>${esc(x.stock.symbol)} ${esc(x.stock.name)}</b><br><span class="muted">${esc(x.start)} ~ ${esc(x.end)} · ${esc(x.timeframe)} · ${x.result_count} 条</span></div>`).join(''):'<div class="empty">暂无记录</div>'; $('history').querySelectorAll('.history-item').forEach(el=>el.onclick=async()=>{const x=xs.find(y=>y.id===el.dataset.id);if(x){
  // Restore the original query parameters so the loaded result can be
  // edited and searched again immediately.
  stock.value=x.stock.ts_code||x.stock.symbol||'';
  $('timeframe').value=x.timeframe||'1d';
  $('start').value=x.start||'';
  $('end').value=x.end||'';
  renderRecord(x);status.textContent='已加载历史查询：'+x.created_at+'（参数已回填）';
}})}
loadHistory();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    app: SimilarityApp

    def log_message(self, fmt, *args):
        return

    def send_json(self, value: object, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if parsed.path == "/api/stocks":
            q = parse_qs(parsed.query).get("q", [""])[0].upper(); rows = [x for x in self.app.stocks if q in x["symbol"] or q in x["name"].upper()][:20]; self.send_json(rows); return
        if parsed.path == "/api/history": self.send_json(self.app.history()); return
        if parsed.path.startswith("/api/jobs/"):
            with self.app.lock: job = dict(self.app.jobs.get(parsed.path.rsplit("/", 1)[-1], {"status": "missing"}))
            self.send_json(job, 404 if job.get("status") == "missing" else 200); return
        if parsed.path.startswith("/files/"):
            rel = parsed.path.removeprefix("/files/"); path = (ROOT / rel).resolve()
            if not str(path).startswith(str(ROOT.resolve())) or not path.exists() or not path.is_file(): self.send_error(404); return
            if path.suffix.lower() == ".svg":
                body = standalone_svg(path.read_text(encoding="utf-8")).encode("utf-8")
            else:
                body = path.read_bytes()
            self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-cache"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
            job_id = self.path.split("/")[-2]
            if self.app.cancel(job_id):
                self.send_json({"status": "cancelled", "message": "已取消搜索"})
            else:
                self.send_json({"error": "任务已结束或不存在"}, 409)
            return
        if self.path != "/api/search": self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
            self.send_json({"id": self.app.submit(payload)}, 202)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def run(host: str = "127.0.0.1", port: int = 8765, data_path: Path = DEFAULT_DATA) -> None:
    app = SimilarityApp(data_path)
    Handler.app = app
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"K-line similarity UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start local K-line similarity UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    run(args.host, args.port, args.data)
