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

标准化 Top-10 + 单幅 SVG 图（相似区间前后各补 44 根同周期 K 线）：

```powershell
python -m strategies.kline_similarity.visualize `
  600519.SH 2024-01-01 2024-03-31 `
  --timeframe 1d --output-dir strategies/kline_similarity/results
```

返回的每个 item 包含 `symbol`、`timeframe`、`interval`、`score`、`image_path` 和 `html_path`。单幅图沿用项目统一的对数价格坐标、红涨绿跌、成交量、MA5/10/20 和黄色目标区间样式；HTML 汇总页可直接浏览 Top-10 图表。

实现采用 close-shape 的 MASS 等价快速召回、OHLCV 特征的 Sakoe-Chiba 约束 DTW、指标 rank fusion 和全局 NMS；不依赖 stumpy/tslearn，方便直接在现有环境运行。自定义数据库时实现 `MarketDataProvider.get_symbols()` 和 `get_symbol_data()` 即可。
