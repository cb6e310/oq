# K 线搜索与事件扫描性能优化指导

> 适用项目：`cb6e310/oq`  
> 适用场景：全市场历史 K 线事件扫描、规则策略回测、后续相似 K 线检索能力建设  
> 目标：在保持结果正确性的前提下，显著降低重复计算、内存占用和全历史扫描成本，并为后续扩展更多策略建立统一的数据访问与特征计算架构。

---

## 1. 文档目标

当前项目已经采用了较多向量化方式进行事件扫描，真正的性能瓶颈并不主要位于策略条件本身，而集中在：

- 每次运行重复读取全量历史行情；
- 每次运行重复执行全表排序；
- 每次运行重复计算 `rolling / groupby / merge` 特征；
- 月度特征被扩散回全部日线；
- 市场指数等公共数据被重复生成；
- 数据文件组织方式不利于按时间区间裁剪；
- 一些只对少数候选事件需要的昂贵特征，却提前对全市场全部 K 线计算。

因此，优化重点应从：

```text
“让某个 NumPy 表达式更快”
```

转变为：

```text
“让绝大多数不必要的计算根本不要发生”
```

推荐的总体方向是：

```text
全量原始数据
    ↓
一次性标准化 / 特征构建
    ↓
按时间分区保存
    ↓
策略运行时只加载必要列和必要年份
    ↓
便宜条件粗筛
    ↓
只对少量候选执行昂贵条件
    ↓
事件结果
```

---

# 2. 当前性能瓶颈概览

当前策略运行的大致流程可以抽象为：

```text
读取约 1819 万行日线
        ↓
全表排序
        ↓
groupby / cumcount
        ↓
复权价格计算
        ↓
涨跌停价格计算
        ↓
63 根历史窗口 rolling
        ↓
月线聚合
        ↓
月线 merge 回日线
        ↓
涨停累计次数
        ↓
查找一字板
        ↓
D1 / D2 事件判断
        ↓
市场指数构造
        ↓
forward return
```

其中：

```text
find_events()
```

本身已经主要使用 NumPy 布尔向量和位置索引，属于较高效的线性扫描。

真正值得优先优化的是：

```text
load_bars()
feature construction
monthly merge
market index construction
full-history I/O
```

---

# 3. 优化原则

## 3.1 优先减少数据量，而不是先换技术栈

优先级应当是：

```text
减少读取
    >
减少计算
    >
减少 join / merge
    >
减少中间对象
    >
换 Polars / Numba / Cython
```

例如：

```text
最快的 rolling
不如
根本不对 1819 万行执行 rolling
```

同理：

```text
最快的 merge
不如
只对 5000 个候选事件做 merge
```

---

## 3.2 区分三类特征

建议把策略使用的特征分为三类。

### A. 静态特征

一旦股票或交易规则确定，基本不会改变。

例如：

```text
sid
ts_code
board
上市顺序
```

这些特征应该提前保存，不应每次策略运行重新计算。

---

### B. 行情派生特征

只要底层行情没有变化，其结果就不会变化。

例如：

```text
bar_no
new_stock
close_c
is_yizi
hfq_close
prior_63_low
is_limit_up
limit_ups_prev3
```

适合一次性预计算并缓存。

---

### C. 策略特有特征

只有某个策略需要。

例如：

```text
特定长度的历史最低点
特殊形态结构
某种动态阈值
```

这类特征需要判断：

```text
很多策略都会使用
    → 预计算

只有单个策略使用
    → 候选事件出现后再计算
```

---

# 4. P0：立即可以做的低风险优化

## 4.1 删除策略运行阶段的全表排序

当前类似：

```python
df = pd.read_parquet(DAILY)

df = (
    df.sort_values(
        ["ts_code", "date"],
        kind="stable",
    )
    .reset_index(drop=True)
)
```

对于 1800 万级数据，全表排序会产生明显 CPU 和内存开销。

更合理的方式是在数据构建阶段保证：

```text
同一股票的数据连续存储
且
每只股票内部 date 单调递增
```

即维护 invariant：

```text
sid contiguous
date ascending within sid
```

策略运行阶段直接依赖这个约束。

---

## 4.2 给数据增加整数型 `sid`

不建议在热路径大量使用：

```text
"000001.SZ"
"600519.SH"
```

作为 group key 和比较对象。

推荐建立：

```text
sid: int32
```

例如：

```text
1
2
3
...
5908
```

股票基础信息单独维护：

```text
sid
ts_code
board
list_date
```

好处包括：

- groupby 更快；
- 比较更快；
- 内存占用更低；
- NumPy 索引更自然；
- 后续构建稠密数组更加方便。

---

## 4.3 Parquet 只读取策略真正需要的列

不要：

```python
df = pd.read_parquet(path)
```

而应使用：

```python
df = pd.read_parquet(
    path,
    columns=[
        "sid",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_factor",
    ],
)
```

如果策略不需要：

```text
vol
amount
```

就不应让它们进入内存。

对于 1800 万行数据，即使少加载两列，也可能减少数百 MB 的内存压力。

---

## 4.4 避免重复 `cumcount`

如果已经存在：

```python
df["bar_no"] = g.cumcount()
```

后续不要再：

```python
same = g.cumcount() > 0
```

直接：

```python
same = df["bar_no"].to_numpy() > 0
```

类似原则：

```text
任何已经计算好的列
不要在同一次运行中重新 groupby 得到
```

---

# 5. P1：建立一次性 Daily Feature Cache

这是整个优化方案中最重要的一项。

建议新增：

```text
database/processed/
    stock_daily_features/
```

把与策略无关、但每次运行都重复生成的特征提前保存。

推荐字段：

```text
sid
date

bar_no
new_stock

open_c
high_c
low_c
close_c

adj_factor

hfq_open
hfq_high
hfq_low
hfq_close

board_id

is_yizi
is_limit_up
is_limit_up_5

limit_ups_prev3

prior_63_low
```

运行方式由：

```text
策略启动
  ↓
重新计算 1819 万行所有基础特征
```

改成：

```text
行情更新
  ↓
build_features.py
  ↓
只计算一次
  ↓
所有策略共用
```

---

# 6. OHLC 建议改为整数“分”存储

当前很多涨停判断最终都会执行类似：

```python
np.round(close * 100)
```

因此更适合直接保存：

```text
open_c
high_c
low_c
close_c
```

数据类型可以优先考虑：

```text
int32
```

例如：

```text
10.23 元
→
1023
```

优点：

- 避免浮点价格比较误差；
- 一字板判断更直接；
- 涨跌停价格计算更自然；
- 压缩率通常更好；
- 内存占用更可控。

例如：

```python
is_yizi = (
    (high_c - low_c <= 1)
    & (abs(close_c - open_c) <= 1)
    & (abs(close_c - low_c) <= 1)
)
```

只有复权计算时再转浮点：

```python
hfq_close = close_c * 0.01 * adj_factor
```

---

# 7. P1：月度特征独立保存，不要扩散回全部日线

当前常见做法是：

```text
日线
 ↓
groupby 股票 + 月份
 ↓
生成月线
 ↓
merge 回 1819 万日线
```

这是非常昂贵的。

而策略真正需要：

```text
current_month_return
prev_month_return
```

通常只是在最终事件判断阶段使用。

推荐建立：

```text
stock_monthly_features.parquet
```

字段：

```text
sid
month

month_close
prev_month_close
prev_month_return
```

策略执行流程：

```text
全市场日线
    ↓
先找 D0 / D1 / D2 候选
    ↓
得到少量事件
    ↓
event × monthly feature lookup
```

而不是：

```text
monthly feature × 全部 1819 万日线
```

---

# 8. P1：市场等权指数预计算

如果 forward return 需要：

```text
全市场每日等权指数
```

那么这个指数属于：

```text
公共基础数据
```

而不是某个策略独有的数据。

建议保存：

```text
database/processed/
    equal_weight_market.parquet
```

字段例如：

```text
date
open_idx
close_idx
market_return
overnight_return
```

策略运行时只需要按日期 lookup。

这样可以避免每次策略运行重新执行：

```text
groupby stock
groupby date
cumprod
```

---

# 9. P2：按年份对 Parquet 分区

一个大的：

```text
stock_daily.parquet
```

不利于历史区间裁剪。

推荐：

```text
stock_daily_features/
    year=1995/
    year=1996/
    year=1997/
    ...
    year=2025/
    year=2026/
```

策略如果只研究：

```text
2023 ~ 2026
```

则只扫描：

```text
year=2022
year=2023
year=2024
year=2025
year=2026
```

其中多读一部分历史，是为了满足：

```text
63-bar rolling
前 3 根 K 线
上一月
其他 lookback
```

---

# 10. 日期裁剪时必须保护 lookback

不能简单：

```python
df = read(
    date >= start_date
)
```

然后才计算：

```text
rolling 63
previous 3 bars
previous month
```

否则研究区间开头的数据会错误。

正确模式有两个。

---

## 方案 A：全历史预计算 feature

```text
全历史
    ↓
预计算所有基础 feature
    ↓
保存
    ↓
策略运行时随便裁日期
```

这是推荐方式。

---

## 方案 B：读取额外 lookback

例如：

```text
研究起点：2024-01-01
最大 lookback：63 bars
```

则加载：

```text
2023 年末额外若干交易日
+
2024 之后的数据
```

但是需要谨慎处理：

```text
停牌
上市不足
跨月
跨年
```

因此如果存储空间允许，更推荐方案 A。

---

# 11. P2：采用“两阶段扫描”

不要所有昂贵特征都提前对 1819 万行计算。

建议：

```text
Phase 1
便宜条件
    ↓
生成 candidate

Phase 2
昂贵条件
    ↓
最终事件
```

例如对于一字板策略：

```text
全部日线
   ↓
便宜判断：
一字形态 + 涨停
   ↓
D0 candidates
   ↓
检查 D1 / D2
   ↓
少量候选
   ↓
计算：
prior 63 low
月度动量
其他复杂条件
```

---

# 12. `prior_63_low` 是否应预计算

这是一个需要按策略数量决定的问题。

## 情况 1：很多策略都会用历史区间低点

推荐：

```text
预计算 prior_63_low
```

一次计算，所有策略复用。

---

## 情况 2：只有一个策略使用

可以采用 candidate-first。

例如已经得到：

```python
candidate_pos
```

构造：

```python
idx = (
    candidate_pos[:, None]
    - np.arange(1, 64)
)
```

然后：

```python
prior_63_low = hfq_low[idx].min(axis=1)
```

如果：

```text
candidate = 20,000
```

则处理的数据规模约：

```text
20,000 × 63
≈ 126 万
```

远小于：

```text
1819 万行 rolling
```

---

# 13. 事件扫描建议保持 NumPy 向量化

当前这一类写法是合理的：

```python
pos = np.arange(n)

code = df["sid"].to_numpy()

prev_idx = pos - 1

same_stock = (
    (prev_idx >= 0)
    & (code[prev_idx] == code)
)
```

规则型事件搜索建议继续遵循：

```text
布尔数组
+
位置索引
+
整数 sid
```

尽量避免：

```python
for stock in stocks:
    for bar in bars:
        ...
```

后者在 Python 层循环会明显更慢。

---

# 14. 数据层建议增加 `MarketDataStore`

为了避免每个策略重复写：

```text
读 Parquet
裁日期
选择列
加载月线
加载指数
处理 lookback
```

建议抽象一个统一的数据访问层。

例如：

```python
class MarketDataStore:

    def load_daily(
        self,
        start,
        end,
        columns,
        lookback_bars=0,
    ):
        ...

    def load_monthly(
        self,
        start,
        end,
        columns,
    ):
        ...

    def load_market_index(
        self,
        start,
        end,
    ):
        ...
```

策略只声明：

```python
bars = store.load_daily(
    start=start,
    end=end,
    columns=[
        "sid",
        "date",
        "close_c",
        "is_yizi",
        "is_limit_up",
        "hfq_close",
    ],
    lookback_bars=63,
)
```

这样未来新策略不需要关心：

```text
Parquet 文件位置
年份分区
日期 predicate
lookback
股票 metadata
```

---

# 15. 推荐目标架构

```text
                     TDX .day
                        │
                        ▼
                build_adjusted
                        │
                        ▼
               normalized daily
                        │
                        ▼
                 build_features
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
 daily_features     monthly       market index
  year partition    features
          │
          │
          ▼
     MarketDataStore
          │
          ▼
      strategy scan
          │
          ▼
      cheap filter
          │
          ▼
       candidate
          │
          ▼
    expensive filter
          │
          ▼
        events
          │
          ▼
   forward-return lookup
```

---

# 16. 推荐目录结构

```text
database/
├── raw/
│   └── ...
│
├── normalized/
│   └── stock_daily/
│
└── processed/
    ├── stock_daily_features/
    │   ├── year=1995/
    │   ├── year=1996/
    │   ├── ...
    │   └── year=2026/
    │
    ├── stock_monthly_features.parquet
    │
    ├── equal_weight_market.parquet
    │
    └── securities.parquet
```

其中：

```text
securities.parquet
```

建议包括：

```text
sid
ts_code
board
list_date
```

---

# 17. 是否需要换 Polars

推荐顺序：

```text
先改架构
    ↓
再 benchmark
    ↓
如果仍然 CPU / memory bound
    ↓
考虑 Polars
```

Polars 比较适合：

```text
Parquet scan
predicate pushdown
column pruning
group/window
大量列运算
```

例如：

```python
import polars as pl

df = (
    pl.scan_parquet(
        "stock_daily_features/year=*/data.parquet"
    )
    .filter(
        pl.col("date") >= start
    )
    .select([
        "sid",
        "date",
        "close_c",
        "is_yizi",
        "is_limit_up",
    ])
    .collect()
)
```

优势包括：

- Lazy execution；
- predicate pushdown；
- column pruning；
- 多线程执行；
- 较低的对象开销。

但要注意：

```text
Polars 不是第一优先级
```

如果仍然每次：

```text
扫描全历史
重新 rolling
重新 merge
```

仅仅把 Pandas 改成 Polars，仍然是在执行大量本可以避免的工作。

---

# 18. Numba / Cython / GPU 的优先级

当前并不建议优先投入。

原因：

```text
瓶颈首先是 I/O + 全量重复 feature engineering
```

而不是单个 Python 数值循环。

推荐顺序：

```text
1. column pruning
2. feature cache
3. year partition
4. candidate-first
5. monthly / market 独立缓存
6. Polars
7. Numba / Cython
8. GPU
```

只有 profiling 明确显示：

```text
某个纯数值计算函数占用了大量 CPU
```

才值得用 Numba/Cython。

---

# 19. 推荐的实施路线

## Phase 0：建立性能基线

在改代码之前记录：

```text
总运行时间
Peak RSS
Parquet read 时间
sort 时间
feature 时间
find_events 时间
forward_returns 时间
event 数量
```

建议至少测试：

```text
1 年
3 年
10 年
全历史
```

---

## Phase 1：低风险优化

完成：

```text
[ ] Parquet columns 裁剪
[ ] 删除重复 cumcount
[ ] build 阶段保证排序
[ ] 删除运行阶段 sort_values
[ ] board 转静态 metadata
```

---

## Phase 2：公共 feature cache

完成：

```text
[ ] sid
[ ] bar_no
[ ] new_stock
[ ] cent-based OHLC
[ ] limit-up feature
[ ] limit_ups_prev3
[ ] hfq prices
```

重新 benchmark。

---

## Phase 3：拆分低频数据

完成：

```text
[ ] monthly feature dataset
[ ] equal-weight market dataset
[ ] event 后再 lookup monthly
```

重新 benchmark。

---

## Phase 4：Parquet partition

完成：

```text
[ ] year partition
[ ] date predicate
[ ] column pruning
[ ] configurable lookback
```

重新 benchmark。

---

## Phase 5：candidate-first

识别所有：

```text
昂贵但只对少量 candidate 需要
```

的计算。

优先考虑：

```text
prior_63_low
复杂 rolling
特殊统计量
```

重新 benchmark。

---

## Phase 6：Polars

只有在前面完成后仍需要进一步提升时：

```text
[ ] scan_parquet
[ ] LazyFrame
[ ] predicate pushdown
[ ] parallel group/window
```

---

# 20. Benchmark 设计

优化不能只看：

```text
“感觉快了”
```

建议使用固定测试集。

例如：

```text
dataset:
2016-01-01 ~ 2026-09-30

strategy:
yizi_pullback_d2

参数固定
```

每次记录：

| 指标 | Before | After |
|---|---:|---:|
| Parquet read |  |  |
| sort |  |  |
| base feature |  |  |
| rolling |  |  |
| monthly |  |  |
| candidate scan |  |  |
| forward returns |  |  |
| total |  |  |
| peak memory |  |  |
| events |  |  |

---

# 21. 正确性验收

所有性能优化必须首先保证：

```text
事件结果不变
```

建议保存一个 golden dataset。

例如：

```text
tests/golden/
    yizi_pullback_d2_events.parquet
```

每次优化后检查：

```python
assert optimized_events.equals(
    baseline_events
)
```

或者至少比较：

```text
ts_code
D0 date
signal date
signal type
```

---

## 21.1 必须特别测试的边界

### 股票停牌

确保：

```text
previous 3 bars
```

指的是前 3 根该股票 K 线，而不是前 3 个 calendar days。

---

### 新股

确保：

```text
bar_no
new_stock
```

没有因为日期裁剪而从 0 重新开始。

---

### 跨年

例如：

```text
2025-12
→
2026-01
```

month feature 必须连续。

---

### 除权除息

确保：

```text
涨跌停判断
```

仍然基于实际交易价格和正确参考价，而不是直接使用后复权价格判断。

---

### 研究区间边缘

例如：

```text
start = 2024-01-01
```

必须确保：

```text
prior 63 bars
previous month
previous 3 bars
```

仍然完整。

---

# 22. 推荐增加 profiling

建议给主要阶段增加简单 timer：

```python
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer(name):
    t0 = perf_counter()

    yield

    dt = perf_counter() - t0

    print(
        f"{name}: {dt:.3f}s"
    )
```

使用：

```python
with timer("read parquet"):
    df = load_daily()

with timer("build features"):
    df = build_features(df)

with timer("find events"):
    events = find_events(df)
```

不要在没有 profiler 的情况下猜瓶颈。

---

# 23. 一个更合理的策略接口

最终建议让策略变成：

```python
class YiziPullbackD2:

    daily_columns = [
        "sid",
        "date",
        "close_c",
        "hfq_close",
        "is_yizi",
        "is_limit_up",
        "limit_ups_prev3",
    ]

    lookback_bars = 63

    def find_candidates(
        self,
        bars,
    ):
        ...

    def apply_expensive_filters(
        self,
        candidates,
        store,
    ):
        ...
```

运行框架：

```python
strategy = YiziPullbackD2()

bars = store.load_daily(
    start=start,
    end=end,
    columns=strategy.daily_columns,
    lookback_bars=strategy.lookback_bars,
)

candidates = strategy.find_candidates(
    bars
)

events = strategy.apply_expensive_filters(
    candidates,
    store,
)
```

这样：

```text
数据管理
和
策略逻辑
```

被彻底分离。

---

# 24. 如果未来增加“相似 K 线搜索”

当前规则扫描属于：

```text
满足条件
→ True / False
```

未来如果增加：

```text
输入任意一段 K 线
→ 搜全市场历史最相似窗口
```

建议不要复用当前的事件扫描逻辑硬做，而是新增独立模块：

```text
similarity/
    feature_encoder.py
    candidate_index.py
    distance.py
    search.py
```

流程：

```text
K 线窗口
    ↓
无量纲归一化
    ↓
粗特征
    ↓
ANN / coarse search
    ↓
候选窗口
    ↓
精确距离
    ↓
Top-K
```

推荐比较特征：

```text
log return
OHLC relative position
body ratio
upper wick ratio
lower wick ratio
range
volume z-score
```

固定长度窗口优先考虑：

```text
cosine distance
correlation distance
weighted Euclidean distance
```

不要一开始就使用 DTW，除非明确希望允许时间轴伸缩。

---

# 25. 最值得优先执行的三个改动

如果当前只准备做三个优化，建议按下面顺序。

## 第一项

把：

```text
全表 sort
```

从策略运行阶段移到数据构建阶段。

---

## 第二项

建立：

```text
daily feature cache
```

至少预计算：

```text
sid
bar_no
new_stock
is_limit_up
limit_ups_prev3
hfq prices
```

---

## 第三项

把：

```text
monthly features
market index
```

拆成独立数据集。

策略只在候选事件上 lookup。

---

# 26. 预期效果

优化前：

```text
每次策略运行

读取全历史
    ↓
排序全历史
    ↓
重新计算基础 feature
    ↓
重新 rolling
    ↓
重新 monthly merge
    ↓
重新 market index
    ↓
事件扫描
```

优化后：

```text
行情更新时一次性：

build base feature
build monthly
build market index


策略运行时：

读取目标年份
    ↓
读取必要列
    ↓
cheap candidate scan
    ↓
少量 expensive lookup
    ↓
events
```

真正的性能提升来源不是：

```text
某个函数快 20%
```

而是：

```text
让 90% 以上的重复工作从每次策略运行中消失
```

---

# 27. 最终建议

当前事件扫描本身不需要大改。

最值得优化的是：

```text
数据组织
+
公共 feature 计算
+
查询裁剪
+
候选优先
```

建议最终形成如下原则：

```text
Build once.
Store once.
Scan only what is needed.
Compute expensive features only for candidates.
Keep strategy logic independent from storage logic.
```

如果后续策略数量会持续增加，建议尽早建设：

```text
MarketDataStore
+
Feature Store
+
统一 Benchmark
+
Golden Result Test
```

这样性能优化不仅服务于当前的一字板策略，也会直接提升整个 `oq` 策略研究框架的可维护性和扩展能力。
