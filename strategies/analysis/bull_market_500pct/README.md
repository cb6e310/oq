# 牛市区间 500% 涨幅股票研究

使用项目在 `scripts/market_regimes.py` 和根目录 README 中约定的最近两段固定牛市：2019-02-01—2021-11-29、2024-09-23—2026-06-30。

运行：

```powershell
python strategies/analysis/bull_market_500pct/strategy/analyze.py
```

脚本读取 `database/processed/stock_daily.parquet` 和 `stock_basic_tdx.parquet`，并从 Baostock 读取对应牛市结束日可查询到的证监会行业分类。行业快照存入 `results/`；之后可用 `--no-baostock` 离线复算。首次联网运行需安装 `baostock`。

主统计为每只股票在区间内某交易日低点到其后交易日高点的最大后复权涨幅，严格大于 500% 算入名单。低点必须在高点的前一个或更早交易日，同日高低价的先后顺序无法由日线确认。对照统计为区间内首个有效交易日后复权收盘价到区间最高后复权价。区间中新上市及退市股票均按有行情的日期纳入。复权价格采用原始日线的 `adj_factor` 计算，不使用前复权文件的四位小数价格。

输出：

- `results/summary.csv`：两个口径的总体数量。
- `results/industry_summary.csv`：主口径超 500% 名单按行业分组的数量、最大值和均值。
- `results/board_summary.csv`：主板、创业板、科创板、北交所的超 500% 数量、板块内占比和入选股票的涨幅中位数。
- `results/board_over_200pct_summary.csv`：上述板块超 200% 的数量、板块内占比和入选股票的涨幅中位数。
- `results/牛市*_over_500pct.csv`：主口径逐股名单，含名称、行业、低点和高点日期、两个涨幅口径。
- `results/牛市*_all_stocks.csv`：所有有行情股票的计算结果。
- `docs/report.md`：结论与口径说明。

Baostock 未覆盖的股票保留通达信行业代码，两种行业分类不可直接混合解释；`industry_display` 会显式标为“通达信代码 …”。历史行业按区间结束日快照，不代表牛市期间每一天的行业归属。

**数据完整性：**2026-10-19 同步新下载的通达信数据后，第一段牛市起点的股票日线从 230 条增至 3,553 条，天齐锂业已覆盖并入选。外部在交易名单仍有 17 只沪深 A 股在起点无本地行情（其中 1 只全库未收录），因此第一段统计仍可能少量低估。逐股详情见完整性报告。

运行 `python strategies/analysis/bull_market_500pct/strategy/audit_data.py` 可重新核对原始 `.day` 与 Parquet、逐股起止日期及两段牛市的起点覆盖。结果保存在 `docs/data_completeness.md` 与 `results/coverage_by_stock.csv`。
