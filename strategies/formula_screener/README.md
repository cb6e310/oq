# 公式历史筛选

在全 A 股本地日线历史中执行同花顺/通达信风格的选股公式，列出每次历史信号，并汇总信号后 1、3、5、10、20 个该股交易日的平均收盘涨幅；也可查看命中日前后 K 线。筛选时显示百分比进度和已用时间。

公式解析和执行基于 [formula-go](https://github.com/DTrader-store/formula-go)。每只股票按时间顺序逐根输入日线，逐日读取公式输出；未来引用 `REFX` / `REFXV` 会由解析器拒绝。标准选股公式请将选股条件写为最后一个输出，推荐命名为 `XG`：

```text
XG: CROSS(CLOSE, MA(CLOSE, 20));
```

## 运行

需要 Python 3.12 环境 `stock_quant`、项目根目录的数据依赖（`pandas`、`pyarrow`）和 Go 1.25.4 或更新版本。首次运行会由 Go Modules 下载并校验 `formula-go` 模块。

```powershell
conda activate stock_quant
python -m strategies.formula_screener.app
```

服务打开 `http://127.0.0.1:8766`，默认读取 `database/processed/stock_daily_qfq.parquet`。可指定其他行情文件或端口：

```powershell
python -m strategies.formula_screener.app --data database/processed/stock_daily_qfq.parquet --port 8766
```

每次筛选的命中明细保存为 `strategies/formula_screener/results/signals_<任务号>.csv`。页面对所有命中事件汇总前复权收盘收益：信号日收盘至后续第 N 根个股日线收盘，并显示有效样本数；停牌日不计，缺少足够未来数据的信号不纳入该周期平均值。

## 范围与解释

- 公式历史计算会载入信号起始日之前的数据作为指标预热，再仅展示指定日期范围内的命中。
- 输入使用本地前复权日线，字段包括 `OPEN/HIGH/LOW/CLOSE/VOLUME/AMOUNT` 及常见别名；支持的函数和语法以 formula-go 文档为准。
- 非零选股输出视为命中，优先识别 `XG`、`选股`、`SELECT`、`选股条件`，否则取最后一个输出项。若普通指标公式的最后一项不代表选股条件，应先改成明确的 `XG:` 输出。
- 后续收益是收盘到收盘的价格变化，不包含交易成本、滑点、涨跌停可成交性或分红现金流，适合历史筛选与研究，不等同于可实现回测收益。
