# 区间股票表现统计

统计指定日期区间内的领涨股票，并计算区间复权最高价、最低价及对应日期。

默认分析 2015-09-15 至 2015-12-22：

```powershell
python strategies/analysis/interval_stock_performance/strategy/analyze.py
```

自定义区间或数量：

```powershell
python strategies/analysis/interval_stock_performance/strategy/analyze.py --start 2015-09-15 --end 2015-12-22 --top 30
```

结果写入 `docs/report.md` 与 `results/`。前复权行情来自仓库本地 `database/processed/stock_daily_qfq.parquet`，股票名称读取 `database/raw/tdx/meta/infoharbor_ex.code`。

为报告中的 Top 30 生成区间个股日 K 与上证指数对照图：

```powershell
python strategies/analysis/interval_stock_performance/strategy/make_charts.py
```

绘图复用 `scripts/candlestick_svg.py`，输出个股与上证指数上下排列的宽版 SVG；两图共用交易日横坐标，个股在区间开始前展示最多 40 根、结束后展示 20 根日 K。连续浏览页面为 `figures/interval-charts.html`，并会更新 `docs/report.md` 内嵌图表。
