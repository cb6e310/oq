# open_quant

基于通达信本地日线数据的 A 股量化研究项目。

## 环境

conda 环境 `stock_quant`（Python 3.12）：

```bash
conda activate stock_quant
pip install tushare pandas pyarrow tqdm pytdx==1.72
```

## 目录结构

```
open_quant/
├── database/
│   ├── raw/tdx/                 # 通达信原始数据（只读，不要手改）
│   │   ├── sh/lday/*.day        # 上交所：股票、指数、基金、债券等，一个文件一只证券
│   │   ├── sz/lday/*.day        # 深交所
│   │   ├── bj/lday/*.day        # 北交所
│   │   └── gbbq                 # 股本变迁/权息文件（加密，用 pytdx 解码）
│   └── processed/               # 脚本生成，可随时重建
│       ├── stock_daily.parquet
│       ├── stock_daily_qfq.parquet
│       └── xdxr.parquet
├── scripts/
│   └── build_adjusted.py        # 解析 .day + gbbq，生成复权数据
└── strategies/
    ├── yizi_pullback_d2/        # 一字板回调策略
    │   ├── strategy/            # 一字板回调策略代码
    │   ├── results/             # 最终事件明细与统计
    │   ├── figures/             # 图表素材
    │   └── docs/                # 唯一完整回测与可视化报告
    └── ten_year_return/         # 长期收益研究
        ├── strategy/            # 长期收益脚本
        └── results/             # 长期收益结果
```

项目结构说明：

```text
open_quant/
├── database/
│   ├── raw/tdx/                         # 通达信原始行情与股本变迁数据
│   └── processed/                       # 标准化、复权后的 Parquet 数据
│       ├── stock_daily.parquet
│       ├── stock_daily_qfq.parquet
│       └── xdxr.parquet
├── scripts/
│   ├── build_adjusted.py                # 构建复权行情数据
│   ├── download_stock_basic.py          # 下载全 A 股名称、行业、上市状态等基础信息
│   ├── candlestick_svg.py               # 通用 K 线 SVG/手机长图绘图模块
│   └── market_regimes.py                 # 跨策略统一牛熊时段划分
├── strategies/
│   ├── yizi_pullback_d2/                # 一字板回调策略
│   │   ├── strategy/                    # 形态识别、事件研究与图表脚本
│   │   ├── results/                     # 最终事件明细与统计 CSV
│   │   ├── figures/                     # 收益分布、K 线长图等素材
│   │   └── docs/                        # 唯一完整回测与可视化报告
│   └── ten_year_return/                 # 长期收益研究
│       ├── strategy/ten_year_return.py
│       └── results/ten_year_return_by_stock.csv
├── README.md
└── .gitignore
```

目录约定：

- `database/` 只存放行情数据，不存放策略结果。
- `scripts/` 只保留跨策略的数据构建工具。
- 每个独立策略在 `strategies/<strategy_name>/` 下自包含代码、结果、图表和文档。
- 项目根目录不保留 `results/`，策略结果仅保存在对应策略目录内。
- `strategies/test/` 已删除。

## 跨策略研究约定

### 固定牛熊时段

所有策略统一使用以下市场阶段划分，日期边界均包含当日：

| 市场阶段 | 日期范围 |
|---|---|
| 熊市 | 数据起始日—2019-01-31 |
| 牛市 | 2019-02-01—2021-11-29 |
| 熊市 | 2021-11-30—2024-09-22 |
| 牛市 | 2024-09-23—2026-06-30 |
| 熊市 | 2026-07-01 及以后 |

该划分是项目的固定研究口径，不根据后续行情自动调整。跨策略回测、分组统计和可视化需要引用
`scripts/market_regimes.py`，不得在单个策略内另行复制日期边界。

```python
from scripts.market_regimes import market_regime, market_regime_series

market_regime("2024-09-22")  # 熊市
market_regime("2024-09-23")  # 牛市
market_regime("2026-06-30")  # 牛市
market_regime("2026-07-01")  # 熊市
events["market_regime"] = market_regime_series(events["signal_date"])
```

## 统计结果可视化

### K 线图绘制规范

项目中的样本图由 `scripts/candlestick_svg.py` 统一绘制，策略脚本只负责选择样本、准备事件日期和加载行情。当前一字板策略的入口是
`strategies/yizi_pullback_d2/strategy/make_yizi_mobile_charts.py`，输出到
`strategies/yizi_pullback_d2/figures/yizi-mobile-k-lines.html`。

绘图逻辑固定如下：

- 每个样本依次绘制个股日线、周线、月线，以及创业板指（`399006.SZ`）和上证指数（`000001.SH`）月线。
- 日线围绕 D0 前后各取 44 根（最多约 89 根 K 线），周线、个股月线及两个指数月线仍各取前后 22 根，形成更宽的日线视窗与日线 → 周线 → 月线的缩放阅读效果。
- 周线、个股月线和两个指数月线的黄色辅助范围，都映射到该样本日线图的完整展示区间；蓝色虚线标记信号周期。
- 价格坐标使用对数坐标，成交量使用线性坐标；K 线采用红涨绿跌，并叠加 MA5、MA10、MA20 和成交量柱。
- 黄色区域表示 D0 所在周期（或跨周期映射后的目标范围）；日线标注 `D0一字`，周线/月线分别标注 `D0所在周`、`D0所在月`。
- 周、月 K 线由日线聚合：周线按周五收盘周期，月线按自然月；均线在各自周期内重新计算。
- 指数日线从通达信 `database/raw/tdx/{sz,sh}/lday/*.day` 读取后聚合，不混入个股复权因子。

通用绘图模块提供以下接口：

| 接口 | 用途 |
|---|---|
| `read_tdx_day()` | 读取通达信 `.day` 指数/行情文件 |
| `aggregate_bars()` | 将日线聚合为日、周、月 K 线并计算均线 |
| `centered_window()` | 截取目标周期前后固定根数 |
| `render_chart()` | 绘制单个周期 SVG，支持对数价格轴、视窗映射和事件标记 |
| `mobile_css()` | 输出适合手机纵向滚动的长图样式 |

其他策略可以复用该模块，只需提供标准字段 `date/open/high/low/close/vol`，以及目标日期、信号日期和跨周期视窗范围。

本策略另提供固定随机种子的抽样长图：`strategies/yizi_pullback_d2/strategy/make_yizi_random_charts.py` 输出
`strategies/yizi_pullback_d2/figures/yizi-random-positive-negative-k-lines.html`，从有效 T+20 样本中随机抽取正收益 50 个与负收益 50 个；抽样种子为 `20260930`，结果可复现。

## 数据来源与更新

- `.day` 来自通达信 `D:\software\tdx\vipdoc\{sh,sz,bj}\lday`，`gbbq` 来自 `D:\software\tdx\T0002\hq_cache\gbbq`。
- 更新流程：在通达信客户端里下载盘后数据，把上面两处文件覆盖复制到 `database/raw/tdx/`，然后重跑脚本：

```bash
python scripts/build_adjusted.py   # 约 2 分钟
```

当前数据：1990-12-19 ~ 2026-09-29，共 5,908 只 A 股（含已退市），1,819 万行。

### 整合通达信基础资料

项目也支持从本机通达信目录导入基础资料。原始文件放在
`database/raw/tdx/meta/`，解析结果放在 `database/processed/`，与行情数据保持相同的
`raw/processed` 分层：

```text
database/raw/tdx/meta/
├── base.dbf             # 股本、财务数据、上市日期等
├── infoharbor_ex.code   # 股票代码、名称、实际控制人等
├── infoharbor_ex.name   # 通达信证券名称缓存
└── tdxhy.cfg            # 股票与通达信行业分类代码

database/processed/
├── stock_basic_tdx.parquet
└── stock_basic_tdx.csv
```

重新从已复制的通达信文件构建：

```powershell
python scripts/build_tdx_stock_basic.py
```

输出表可通过 `ts_code` 与 `stock_daily.parquet` 连接。主要字段有
`name`、`list_date`、`tdx_area_code`、`tdx_industry_no`、
`tdx_industry_code`、`total_shares`、`float_shares`、`total_assets`、
`net_assets`、`operating_income` 和 `net_profit`。通达信行业字段是分类代码，
不是直接的行业中文名称；名称覆盖取决于本机通达信缓存，退市股和历史名称可能缺失。

### 下载股票基础信息

行情源数据不含股票名称、行业、上市/退市日期等字段。项目使用 Tushare
`stock_basic` 接口补充，默认下载正常上市（L）股票；额度允许时可加
`--include-delisted`，再下载退市（D）和暂停上市（P）股票，便于与包含退市股的历史行情连接。
该接口通常有每分钟或每小时调用限制。

PowerShell：

```powershell
$env:TUSHARE_TOKEN = "你的 Tushare token"
python scripts/download_stock_basic.py
```

也可以临时传入 token，或只生成 Parquet：

```powershell
python scripts/download_stock_basic.py --token "你的 token"
python scripts/download_stock_basic.py --no-csv
python scripts/download_stock_basic.py --include-delisted
```

默认生成：

- `database/processed/stock_basic.parquet`：供 Python/回测程序高效读取。
- `database/processed/stock_basic.csv`：UTF-8 BOM 编码，便于 Excel 直接查看。

主要字段包括 `ts_code`、`symbol`、`name`、`industry`、`area`、`market`、
`exchange`、`list_status`、`list_date`、`delist_date`、`is_hs`、实际控制人等。
与行情表通过 `ts_code` 连接，例如：

```python
import pandas as pd

daily = pd.read_parquet("database/processed/stock_daily.parquet")
basic = pd.read_parquet("database/processed/stock_basic.parquet")
daily = daily.merge(basic[["ts_code", "name", "industry"]], on="ts_code", how="left")
```

## 输出文件

### stock_daily.parquet：不复权行情 + 复权因子

| 列 | 说明 |
|---|---|
| date | 交易日 |
| ts_code | 代码，tushare 格式，如 `600000.SH`、`920000.BJ` |
| open/high/low/close | 不复权价格（元） |
| vol | 成交量（股） |
| amount | 成交额（元）。源文件是 float32，约 7 位有效数字 |
| adj_factor | 后复权因子，上市首日为 1.0 |

- 后复权价 = 价格 × adj_factor
- 前复权价 = 价格 × adj_factor / 该股最后一天的 adj_factor

### stock_daily_qfq.parquet：前复权 OHLC

以每只股票最后一根 K 线为基准，价格保留 4 位小数，vol 和 amount 不做调整。每次更新数据后，历史前复权价都会变。回测时建议用 `stock_daily.parquet` 里的 adj_factor 自己算后复权价。

### xdxr.parquet：除权除息事件及处理结果

从 gbbq 中解出的 category=1 事件，字段有 `cash_per10`（每 10 股派现）、`rights_price`（配股价）、`bonus_per10`（每 10 股送转）、`rights_per10`（每 10 股配股），另外记录了实际落在哪个交易日（`trade_date`）、复权比例（`ratio`）和处理状态（`status`）：

- `applied`：已计入复权因子
- `before_listing`：发生在数据开始之前。绝大多数是北交所公司在新三板时期的分红，北交所行情从 2020-07-27 才有
- `after_data_end`：已公告、尚未到除权日的事件，数据更新后会自动生效

## 分析脚本

策略研究输出位于各策略目录的 `results/`（该目录不进 git）。

| 脚本 | 用途 |
|---|---|
| `scripts/download_stock_basic.py` | 从 Tushare 下载全 A 股名称、行业、上市状态、上市/退市日期等基础信息 |
| `strategies/ten_year_return/strategy/ten_year_return.py` | 全部股票近 N 年的持有收益（后复权、包含退市股）；每日再平衡的等权组合对比沪深300、上证指数 |
| `strategies/yizi_pullback_d2/strategy/event_study.py` | 事件研究通用模块 |
| `strategies/yizi_pullback_d2/strategy/pattern_yizi_pullback.py` | 一字板回调形态识别与回测（当前正式策略口径） |
| `strategies/yizi_pullback_d2/strategy/regime_t20_analysis.py` | 按固定牛熊区间分析 T+20 正负收益相关性，并生成图表 |

事件研究的统一口径：
- 一字板回调策略额外要求：D0 后复权收盘价不高于此前约 3 个月（63 根有效日线）最低后复权低点的 1.2 倍，即涨幅不超过 20%；历史窗口不含 D0。
- 收益：按后复权价计算，持有 h 根该股自己的 K 线，停牌日不计。设 1、3、5、10、20 五档，20 根约等于一个月。
- 买入价两种：
  - `close`：信号日收盘价买入。
  - `open`：次日开盘价买入。次日是一字涨停、买不进的样本会剔除。
- 超额收益：相对同期全市场等权组合（每日再平衡）。用随机抽取的 5 万个股票日做过检验，平均超额收益约为 0（-0.09%）。
- 涨跌停价：用不复权价，基于除权后的参考前收，按四舍五入到分计算。
  - 主板 10%
  - 创业板：2020-08-24 之前 10%，之后 20%
  - 科创板 20%
  - 北交所 30%
  - 数据里没有 ST 标记。主板上正好涨 5% 的一字涨停，按疑似 ST（`st_like`）处理。
- 新股期：上市后到第一次开板之前，以及上市不满 60 根 K 线的日子。
- 已知漏判：近十年大约 39 次一字涨停没被识别出来，主要是重整转增等除权日，复权参考价和交易所的口径不一致。

## 复权方法

等比复权。除权日的参考价计算公式：

```
ref = (前收 - 派现/10 + 配股价×配股/10) / (1 + 送转/10 + 配股/10)
```

除权日及以后的每根 K 线都乘以 `前收 / ref`。如果除权日停牌，就顺延到复牌后的第一个交易日。

### 验证结果（2026-09-30）

- 全市场 5.6 万个除权日中，不复权时有 11,766 个日收益超出涨跌停限制，复权后降到 639 个，其中绝大部分是长期停牌后复牌、2005–2008 年股改复牌或 1996 年以前（当时还没有涨跌停制度）。
- 与新浪前复权（akshare `stock_zh_a_daily`）对照了 600000、000001、600519、300750、688981、002594：2010 年以来的除权日收益差 ≤ 0.16%（新浪只保留 2 位小数），最新收盘价完全一致。

## 已知限制

- 破产重整中的资本公积转增：交易所会用特殊公式调整除权参考价，按 gbbq 的标准公式算会有偏差，例如 600518 在 2021-12-15、000564、000981、600179。研究这类股票时建议剔除，或者用 tushare `adj_factor` 校正。
- 1997 年以前的数据质量一般，不建议用于回测。
- 北交所旧代码（43/83/87 开头）在 2025-09-30 之后不再更新，它们的完整历史已经在对应的 920 开头新代码文件里，所以脚本只处理 920 开头的代码。
- 停牌日在源数据里没有记录，做截面分析时需要按交易日历对齐。
- 源数据里没有股票名称、行业、ST 标记、上市/退市日期，需要另外从 tushare 补充。
- 脚本目前只处理 A 股股票。指数、ETF、可转债的价格缩放倍数不同（ETF 是 /1000），还没有纳入。

