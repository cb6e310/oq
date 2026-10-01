# K 线历史相似形态搜索

给定股票、日期区间和周期，在全市场历史行情中寻找等长的相似区间。支持日线、周线和自然月线：

```powershell
python -m strategies.kline_similarity.search 600519.SH 2024-01-01 2024-03-31 --timeframe 1d --top-k 20
python -m strategies.kline_similarity.search 600519.SH 2022-01-01 2024-12-31 --timeframe 1w
python -m strategies.kline_similarity.search 600519.SH 2018-01-01 2020-12-31 --timeframe 1m
```

默认读取 `database/processed/stock_daily_qfq.parquet`（前复权 OHLC）。默认 `history_only=True`，只返回在查询区间开始日以前结束的窗口，避免未来数据泄漏。传入 `--include-future` 可关闭该限制。

Python API：

```python
from strategies.kline_similarity import KlineSimilarityEngine, ParquetDataProvider

engine = KlineSimilarityEngine(ParquetDataProvider())
matches = engine.search("600519.SH", "2024-01-01", "2024-03-31", timeframe="1d")
for match in matches:
    print(match.to_dict())
```

标准化 Top-10 + 日/周/月三周期 SVG 图（日线相似区间前后各补 90 根，周线和月线前后各补 22 根）：

```powershell
python -m strategies.kline_similarity.visualize `
  600519.SH 2024-01-01 2024-03-31 `
  --timeframe 1d --output-dir strategies/kline_similarity/results
```

返回的每个 item 包含 `symbol`、`timeframe`、`interval`、`score`、`image_paths` 和 `html_path`。`image_paths` 分别提供 `daily`、`weekly`、`monthly` 三张图。图表沿用项目统一的对数价格坐标、红涨绿跌、成交量、MA5/10/20 和黄色目标区间样式；周线/月线黄色范围映射日线完整视窗，HTML 汇总页可直接浏览 Top-10 三周期图表。

实现采用 close-shape 的 MASS 等价快速召回、OHLCV 特征的 Sakoe-Chiba 约束 DTW、成交量变化、涨停/跌停/一字板事件特征、指标 rank fusion 和全局 NMS；当查询区间包含涨停、跌停或一字板时，默认优先返回事件签名一致的历史窗口。不依赖 stumpy/tslearn，方便直接在现有环境运行。自定义数据库时实现 `MarketDataProvider.get_symbols()` 和 `get_symbol_data()` 即可。

性能基准：

```powershell
python -m strategies.kline_similarity.benchmark 601567.SH 2026-09-14 2026-09-29 --timeframe 1d
```

引擎已按 `kline_similarity_optimization_core.md` 做了首轮优化：非法窗口在 local Top-N 前屏蔽；Stage 1 只保留 metadata 并限制全局召回数量；DTW 使用 Sakoe-Chiba rolling-row 低内存实现；NMS 按 bar 区间重叠率判断。基准机上 5908 只股票、recall 1000 的日线示例从约 92~101 秒降至约 63 秒，三周期 Top10 纯渲染约 1 秒。
