# K 线历史相似形态搜索

给定股票、日期区间和周期，在全市场历史行情中寻找相似区间。支持日线、周线和自然月线；默认使用兼容旧结果的等长 `fixed` 模式。

新增“最新走势选股”：仍用指定股票的历史区间作为查询形态，但每只候选股票只取数据库中的最新末端窗口，按原有匹配内核精排并展示前 10 只。自动排除查询股票，以及末根 K 线明显落后于全市场最新日期的停牌/退市股票。网页选择“最新走势选股 · Top10”，或运行：

```powershell
python -m strategies.kline_similarity.visualize 600519.SH 2024-01-01 2024-03-31 --latest-only
python -m strategies.kline_similarity.search 600519.SH 2024-01-01 2024-03-31 --latest-only --top-k 10
```

结果中的开始/结束日期是候选股票**最近走势**的日期，数据库截止日会显示在网页顶部。最新模式支持 `fixed` 和 `elastic`；后者允许末端窗口在既定长度范围内变化。原有历史搜索示例：

```powershell
python -m strategies.kline_similarity.search 600519.SH 2024-01-01 2024-03-31 --timeframe 1d --top-k 20
python -m strategies.kline_similarity.search 600519.SH 2022-01-01 2024-12-31 --timeframe 1w
python -m strategies.kline_similarity.search 600519.SH 2018-01-01 2020-12-31 --timeframe 1m
```

需要允许候选区间比查询多或少几根 K 线时，使用有边界的 `elastic` 模式。它会进行多长度粗召回，再使用带时间带宽、重复步上限和跳过代价的弹性 DTW 精排，不会把任意长度窗口直接拉伸为相似：

```powershell
python -m strategies.kline_similarity.search 600519.SH 2024-01-01 2024-03-31 `
  --timeframe 1d --match-mode elastic --min-length 25 --max-length 35
```

默认读取 `database/processed/stock_daily_qfq.parquet`（前复权 OHLC）。默认 `history_only=True`，只返回在查询区间开始日以前结束的窗口，避免未来数据泄漏。传入 `--include-future` 可关闭该限制。

Python API：

```python
from strategies.kline_similarity import KlineSimilarityEngine, ParquetDataProvider

engine = KlineSimilarityEngine(ParquetDataProvider())
matches = engine.search("600519.SH", "2024-01-01", "2024-03-31", timeframe="1d")
# elastic mode is opt-in; fixed remains the default for compatibility
elastic_matches = engine.search(
    "600519.SH", "2024-01-01", "2024-03-31", timeframe="1d",
    match_mode="elastic", min_length=25, max_length=35,
)
for match in matches:
    print(match.to_dict())
```

标准化 Top-10 + 日/周/月三周期 SVG 图（日线相似区间前后各补 90 根，周线和月线前后各补 22 根）：

```powershell
python -m strategies.kline_similarity.visualize `
  600519.SH 2024-01-01 2024-03-31 `
  --timeframe 1d --output-dir strategies/kline_similarity/results
```

可视化入口同样支持 `--match-mode elastic --min-length 25 --max-length 35`。返回结果和图表副标题会显示 `window_length`、`query_length`、`length_ratio`、`match_mode`，以及 elastic 的状态/锚点距离和跳过 K 线诊断；HTML 汇总页还会显示本次搜索的长度模式。

返回的每个 item 包含 `symbol`、`timeframe`、`interval`、`score`、`image_paths` 和 `html_path`。`image_paths` 分别提供 `daily`、`weekly`、`monthly` 三张图。图表沿用项目统一的对数价格坐标、红涨绿跌、成交量、MA5/10/20 和黄色目标区间样式；周线/月线黄色范围映射日线完整视窗，HTML 汇总页可直接浏览 Top-10 三周期图表。

实现采用 close-shape 的 MASS 等价快速召回、OHLCV 特征的 Sakoe-Chiba 约束 DTW、成交量变化、涨停/跌停/一字板事件特征、指标 rank fusion 和全局 NMS。高开涨停、T 字板、一字板分别有明确事件通道；一字板与 T 字板同时属于高开涨停。查询区间含有这些特殊 K 线时，召回阶段为同类形态保留名额，精排时优先展示包含相应形态的候选；若历史中缺少完全一致的形态，仍展示部分匹配，并在结果中显示匹配比例和缺少的特殊板数量。自定义数据库时实现 `MarketDataProvider.get_symbols()` 和 `get_symbol_data()` 即可。

性能基准：

```powershell
python -m strategies.kline_similarity.benchmark 601567.SH 2026-09-14 2026-09-29 --timeframe 1d
```

引擎已按 `kline_similarity_optimization_core.md` 做了优化：非法窗口在 local Top-N 前屏蔽；Stage 1 只保留 metadata 并限制全局召回数量；elastic 粗筛使用 NumPy/Numba 批量计算，并按稀疏步长扫描，再对候选做弹性 DTW 精排；DTW 使用 Sakoe-Chiba rolling-row 低内存实现；NMS 按 bar 区间重叠率判断。完整 5909 只股票库上，elastic 粗筛窗口量由约 6080 万降至约 1000 万，固定模式不受影响。未安装 Numba 时会自动回退到 NumPy 实现。

## 本地 UI 小软件

启动：

```powershell
python -m strategies.kline_similarity.app
```

浏览器打开 `http://127.0.0.1:8765`。UI 支持股票名称/代码联想（日常名称来自 TDX 的 `infoharbor_ex.code`）、日/周/月周期、`fixed`/`elastic` 匹配模式、日期区间、Top20 三周期图表和过往查询记录。搜索在后台线程执行，页面会显示进度状态；历史记录保存在 `strategies/kline_similarity/query_history.json`。

修改服务代码后，可在页面顶部点击“重启服务”。页面会等待新进程接管原端口并自动刷新；数据同步或搜索正在运行时，重启会提示先等待任务结束。

Windows 用户可以直接双击 `start_ui.bat`：它会在后台启动服务并自动打开浏览器，不需要手动输入命令或网址。

启动脚本和开始菜单快捷方式会优先使用 `%USERPROFILE%\.conda\envs\stock_quant` 中的 Python；该环境需安装根目录 README 中列出的依赖（包括 `pytdx`）。如果该环境不存在，则使用 PATH 中的 `pythonw.exe`。环境变更后重新运行 `install_start_menu.bat`，以更新已有快捷方式。

如需固定到开始菜单，首次双击 `install_start_menu.bat` 安装快捷方式；以后从开始菜单搜索“**KLine Similarity Search**”即可一键启动。重复点击不会启动多个服务，只会打开已有页面。
