# 下跌前起点创两年新高的腰斩股票：T+25 表现

筛选前 63 根个股交易日内从后复权收盘高点首次下跌超过 50% 的股票，要求下跌前起点创此前 504 根个股交易日的后复权收盘新高。分别统计 T+1 至 T+25 最高后复权价的涨幅和 T+25 收盘终点收益；两者均相对腰斩信号日 T0 收盘价计算。

运行：

```powershell
python strategies/analysis/three_month_halving/strategy/analyze.py
```

结果见 [最终报告](docs/report.md) 和 `results/` 内的事件明细、收益分布及年度统计。数据来自 `database/processed/stock_daily.parquet`；后复权价格为原始价格乘 `adj_factor`。参数 `--window`、`--horizon` 默认分别为 63、25。
