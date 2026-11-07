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

实现采用 close-shape 的 MASS 等价快速召回、OHLCV 特征的 Sakoe-Chiba 约束 DTW、成交量变化、涨跌停及特殊板事件特征、指标 rank fusion 和全局 NMS。涨停、跌停、一字板、高开涨停和 T 字板均计入事件覆盖、粗召回、精排特征和最终排序；一字板与 T 字板也分别保留专属事件通道。查询区间出现任意一种事件时，召回阶段会为事件完整的同类候选保留名额，最终排序分成三档：事件种类和数量齐全、事件在相对位置上也一致的候选获得最高档；种类和数量齐全但事件位置或额外事件有差异的候选进入中间档；缺少查询事件的候选进入低档。形态分数仍参与各档内部排序。结果显示事件覆盖率、缺失事件项和事件序列距离。自定义数据库时实现 `MarketDataProvider.get_symbols()` 和 `get_symbol_data()` 即可。

### 涨跌停与特殊板的排序规则

- 六个事件通道的相对权重：涨停 `1`、跌停 `1`、涨停一字板 `3`、跌停一字板 `3`、高开涨停 `1`、T 字板 `4`。同一根 K 线可命中多个通道，因此“缺少事件项”不是缺少的 K 线根数。
- 查询含任意事件时，全局精排预算最多一半预留给完整覆盖查询事件种类和最低数量的候选；其余预算继续由价格形态和结构召回竞争。
- 事件距离按种类比较数量和归一化后的发生位置（数量差异占 `65%`，位置差异占 `35%`），不会被区间中的大量无事件 K 线稀释。弹性模式按各窗口自身长度归一化事件位置。
- 默认 `hard_event_match=True`：事件完全一致的候选最终分数在 `[0.5, 1]`；同类覆盖完整但位置或额外事件不同的候选在 `[0.25, 0.5)`；缺少查询事件的候选在 `[0, 0.25)`。最终结果始终按页面显示分数降序排列；同档仍比较事件质量与原始形态分数。分数是排序指标，不是匹配概率。
- 查询不含任何事件时不启用分档。设置 `hard_event_match=False` 只关闭最终分档，事件仍参与召回和距离计算。`special_event_weight` 默认 `0.20`，`event_recall_weight` 默认 `0.75`。
- `fixed` / `elastic` 与历史 / 最新候选均使用同一规则。结果保留旧版 `special_event_*` 字段，并提供 `event_required`、`event_match`、`missing_events`、`base_score`、`event_count_distance`、`event_order_distance`；历史查询不会自动重算，需要重新搜索。

### 日线与周线背景联合匹配

日线搜索增加第二层周线背景评分：取日线区间起点所在周之前最多 **8 根有交易的周线**，加上从起点所在周到日线区间终点的周线。遵照“不使用后续行情”的设置，终点后的 8 根周线不参与评分。终点落在周中时，从该周周初到终点的日线重建当周 OHLCV，不能使用已经包含周四、周五行情的完整历史周线。查询和每个候选都以各自区间终点截断。

- 默认联合分 `75% × 日线原始分 + 25% × 周线背景分`，随后再应用原有涨跌停/特殊板分档，周线背景不会越过事件优先级。`daily_score` 保留融合前的日线分，`base_score` 是融合后、事件分档前的分数。
- 周线背景分分别比较前置段和区间段，前置段占 `40%`，区间段占 `60%`；分别重采样到 16 个比较点，避免把前期走势与目标区间互相错配。比较相对区间起始周开盘价的对数 OHLC 走势和相对成交量，不要求股价绝对水平相同。
- 缺少前置历史时，只比较双方共同可用的最近周线，并通过覆盖率降低背景分；不使用未来行情补足。查询合计不足 3 根周线时跳过本层。
- 支持 `fixed`、`elastic` 和历史/最新候选，仅在已有精排预算内计算，每只入围股票的周分组只准备一次，不增加全市场周线扫描。周线和月线查询保持原评分。
- 配置 `weekly_context_bars=8`、`weekly_context_weight=0.25`；设权重为 `0` 可关闭本层。这是初始可调参数，尚未进行检索质量校准。结果提供 `weekly_context_score/distance/coverage`、实际比较周数和启用状态，诊断提供 `weekly_context_s`；图表仍保留原有前后行情展示，后续展示行情不参与本层评分。

2026-10-24 对 61 只股票的历史搜索抽样中，100 个精排候选的周线层耗时约 `0.30–0.38` 秒。1000 个重复候选、4 只不同股票的额外检查耗时约 `0.34` 秒；该检查不能代表1000只不同股票的开销。原始结果见 [抽样证据](performance_evidence/2026-10-24/weekly_context_sample.json)，不含图表耗时。

性能基准：

```powershell
python -m strategies.kline_similarity.benchmark 601567.SH 2026-09-14 2026-09-29 --timeframe 1d
python -m strategies.kline_similarity.benchmark 601567.SH 2026-09-14 2026-09-29 --match-mode elastic
python -m strategies.kline_similarity.benchmark 601567.SH 2026-09-14 2026-09-29 --latest-only
```

引擎已按 `kline_similarity_optimization_core.md` 做了优化：非法窗口在 local Top-N 前屏蔽；Stage 1 只保留 metadata 并限制全局召回数量；elastic 粗筛使用 NumPy/Numba 批量计算，并按稀疏步长扫描，再对候选做弹性 DTW 精排；DTW 使用 Sakoe-Chiba rolling-row 低内存实现；NMS 按 bar 区间重叠率判断。完整 5909 只股票库上，elastic 粗筛窗口量由约 6080 万降至约 1000 万，固定模式不受影响。未安装 Numba 时会自动回退到 NumPy 实现。

2026-10-24 性能检查见 [实测报告](performance_review_2026-10-24.md)。NumPy 弹性粗筛现在共享状态/事件累计计数；诊断新增 `coarse_backend`、`dtw_backend`、`recall_s`、`candidate_selection_s`、`rerank_s`、`total_s`，便于区分计算后端和阶段耗时。

需要编译加速时，在**启动服务实际使用的环境**安装 Numba，再重启服务：

```powershell
& "$env:USERPROFILE\.conda\envs\stock_quant\python.exe" -m pip install numba
```

注意：当前 NumPy 回退路径扫描步长为 1，Numba 路径默认采用 `elastic_coarse_step=8`，因此安装 Numba 后默认候选窗口可能变化。比较相同召回精度时，在 Python API 使用 `SimilarityConfig(elastic_coarse_step=1)`，或用以下基准命令；网页目前没有步长设置。

```powershell
python -m strategies.kline_similarity.benchmark 603767.SH 2026-07-21 2026-10-24 --match-mode elastic --elastic-coarse-step 1
```

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
