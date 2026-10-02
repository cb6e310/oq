# 区间日均成交额前 N 只股票

本策略使用 `database/processed/stock_daily.parquet` 的 `amount` 字段（成交额，单位：元），统计每只股票在指定日期区间内实际日线记录的算术平均值，并按均值降序输出。

默认参数是 `2021-12-01` 至 `2022-04-30`、前 50 名。缺失交易日不补零；结果同时保留实际交易天数和相对区间交易日覆盖率。若只想比较基本覆盖完整区间的股票，可增加 `--min-days 80` 等过滤条件。

```powershell
python strategies/analysis/average_daily_amount_top50/strategy/average_daily_amount_top50.py
```

结果写入 `results/top50_average_daily_amount.csv`，其中 `average_daily_amount_yi` 为亿元，`average_daily_amount` 为原始元值。`vol`（成交量）不参与计算。
