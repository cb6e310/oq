# K线历史相似形态搜索引擎 --- 实现规格 v1.0

> 目标：给定任意股票的一段历史日 K 区间，在全 A 股历史日 K
> 数据中寻找形态最相似的历史区间，并返回 Top-K。\
> 约束：无需训练模型；本地已具备全 A 股历史日 K
> 数据；数据接口由现有项目自行适配。\
> 推荐架构：**MASS 快速全市场召回 → 多维 OHLCV Constrained-DTW 精排 →
> Rank Fusion → Global NMS**。

------------------------------------------------------------------------

## 1. 目标

输入概念：

``` text
query_symbol = "600519"
query_start  = "2024-01-01"
query_end    = "2024-03-31"
top_k = 50
```

假设本地数据至少包含：

``` text
symbol
date
open
high
low
close
volume
```

可选：

``` text
amount
turnover
adj_factor
```

输出：

``` python
[
    {
        "symbol": "000001",
        "start_date": "2018-03-05",
        "end_date": "2018-06-01",
        "score": 0.9312,
        "mass_score": ...,
        "dtw_score": ...,
        "return_score": ...,
        "candle_score": ...,
        "volume_score": ...,
        "window_length": 60
    },
    ...
]
```

最终按 `score DESC` 排序。

------------------------------------------------------------------------

## 2. 核心原则

不要直接比较 raw close。不同股票价格尺度完全不同，但形态可能完全一致。

所有比较必须基于**尺度无关 representation**。

同时，不建议只比较 Close。真正的 K 线形态至少包含：

-   趋势
-   每日涨跌
-   实体长度
-   上影线
-   下影线
-   振幅
-   跳空
-   成交量变化

总体架构：

``` text
Stage 1
Close shape MASS
    ↓
候选 Top-N
    ↓
Stage 2
Multivariate OHLCV constrained DTW
    ↓
Top-K
```

------------------------------------------------------------------------

## 3. 数据预处理

### 3.1 复权

优先使用前复权 OHLC，或者任何保证历史价格序列连续的复权方式。

`open/high/low/close` 必须采用相同复权因子。

如果本地数据已经复权，不重复处理。Volume 不进行价格复权。

------------------------------------------------------------------------

## 4. Query 定义

例如：

``` text
symbol = 600519
start  = 2024-01-01
end    = 2024-03-31
```

提取实际交易日：

``` python
query_df
m = len(query_df)
```

历史候选默认使用完全相同的交易日长度：

``` python
candidate = stock_df[i:i+m]
```

------------------------------------------------------------------------

## 5. 避免未来数据泄漏

任何 `feature[t]` 只能依赖 `<= t` 的数据。

允许：

``` python
rolling(20)
ewm(...)
pct_change()
```

不允许：

``` python
center=True
```

如果用于严格历史研究，默认启用：

``` python
history_only = True
```

此时必须满足：

``` text
candidate.end_date < query.start_date
```

------------------------------------------------------------------------

## 6. Stage 1：MASS 全市场快速召回

推荐使用 **STUMPY**：

-   GitHub: https://github.com/stumpy-dev/stumpy
-   MASS API: https://stumpy.readthedocs.io/en/latest/api.html

核心：

``` python
stumpy.mass(Q, T)
```

若 Query 长度为 `m`，历史序列长度为 `n`：

``` python
D = stumpy.mass(Q, T)
```

则：

``` text
len(D) = n - m + 1
```

其中：

``` text
D[i]
```

表示 Query 与：

``` text
T[i:i+m]
```

之间的 z-normalized Euclidean distance。

因此 `argmin(D)` 就是该股票历史中与 Query 最接近的等长窗口。

### 6.1 MASS 输入

Stage 1 使用：

``` python
log_close = np.log(close)
```

调用：

``` python
D = stumpy.mass(
    query_log_close,
    historical_log_close
)
```

MASS 的 z-normalization 会消除绝对价格 level
和整体尺度，主要比较价格轨迹形状。

### 6.2 数学含义

对于 Query `Q` 和 Candidate `C`：

``` text
Z(Q) = (Q - mean(Q)) / std(Q)
Z(C) = (C - mean(C)) / std(C)
```

距离：

``` text
D_mass = sqrt(Σ (Z(Q)t - Z(C)t)^2)
```

越小越相似。

对于等长 z-normalized 序列，该距离与 Pearson correlation 高度对应，因此
Stage 1 没必要再重复计算 Pearson。

------------------------------------------------------------------------

## 7. Stage 1 全市场扫描

伪代码：

``` python
global_candidates = []

for symbol in universe:

    df = load_stock(symbol)

    if len(df) < m:
        continue

    T = np.log(df.close.values)

    D = stumpy.mass(
        Q=query_log_close,
        T=T
    )

    candidates = local_minima(D)

    for idx in candidates:
        global_candidates.append({
            "symbol": symbol,
            "start_idx": idx,
            "end_idx": idx + m - 1,
            "mass_distance": D[idx]
        })
```

不要保存所有窗口。

每只股票只保留：

``` python
LOCAL_TOP_N = 20
```

全市场汇总后按 `mass_distance ASC` 排序，只让：

``` python
GLOBAL_RECALL_N = 1000
```

进入第二阶段。

建议允许配置 `500~2000`。

------------------------------------------------------------------------

## 8. Exclusion Zone / Non-Maximum Suppression

这是必须实现的。

假设最佳结果是：

``` text
2018-01-01 → 2018-03-31
```

附近偏移一两个交易日的窗口必然也很相似。如果不处理，Top-K
会被同一段行情占满。

默认：

``` python
exclusion_radius = max(
    5,
    int(m * 0.25)
)
```

每次选出：

``` python
idx = argmin(D)
```

后：

``` python
D[idx-radius : idx+radius+1] = np.inf
```

继续寻找下一个局部最佳结果，直到获得 `LOCAL_TOP_N`。

------------------------------------------------------------------------

## 9. 排除 Query 自身

如果：

``` text
candidate.symbol == query.symbol
```

并且 candidate 与 query 的日期区间显著重叠，则排除。

普通模式可使用：

``` text
overlap_ratio > 0.2
```

严格历史研究模式：

``` python
history_only = True
```

要求：

``` text
candidate.end_date < query.start_date
```

推荐默认 `history_only=True`。

------------------------------------------------------------------------

## 10. Stage 2：K线 Feature Representation

Stage 1 是价格曲线召回；Stage 2 才真正比较 K 线结构。

定义：

``` python
prev_close = close.shift(1)
```

### Feature 1：Close Return

``` python
r_close = log(close / prev_close)
```

### Feature 2：Open Gap

``` python
gap = log(open / prev_close)
```

### Feature 3：Intraday Body Return

``` python
body_return = log(close / open)
```

### Feature 4：Upper Wick

``` python
upper_wick = (
    high - maximum(open, close)
) / prev_close
```

### Feature 5：Lower Wick

``` python
lower_wick = (
    minimum(open, close) - low
) / prev_close
```

应满足：

``` text
lower_wick >= 0
```

### Feature 6：Daily Range

``` python
range_pct = (high - low) / prev_close
```

### Feature 7：Volume Change

``` python
volume_log_change =
log((volume + eps) / (volume.shift(1) + eps))
```

每根 K 线最终表示为：

``` python
X[t] = [
    r_close,
    gap,
    body_return,
    upper_wick,
    lower_wick,
    range_pct,
    volume_log_change
]
```

因此：

``` text
X.shape = (m, 7)
```

------------------------------------------------------------------------

## 11. Feature Robust Normalization

不同 feature 数值尺度不同，不能直接进入 Euclidean/DTW。

不要针对每个 candidate 单独 z-score 所有 feature。

建议从全市场历史数据预计算每个 feature 的：

``` text
median
Q25
Q75
IQR = Q75 - Q25
```

然后：

``` python
X[:, j] = (
    X[:, j] - median_j
) / (IQR_j + eps)
```

最后：

``` python
X = np.clip(X, -5, 5)
```

目的：防止涨跌停、复牌、异常成交量和脏数据完全支配距离。

这只是 normalization，不属于模型训练。

------------------------------------------------------------------------

## 12. Feature 权重

默认：

``` python
FEATURE_WEIGHTS = {
    "r_close":           2.0,
    "gap":               0.8,
    "body_return":       1.2,
    "upper_wick":        0.8,
    "lower_wick":        0.8,
    "range_pct":         1.0,
    "volume_log_change": 0.6,
}
```

原则：

``` text
趋势/收益 > K线实体 > 波动/影线 > Volume
```

Volume 权重不宜过高，因为不同股票、年代、市值和流动性差异很大。

------------------------------------------------------------------------

## 13. Stage 2：Multivariate DTW

推荐使用 **tslearn**：

-   GitHub: https://github.com/tslearn-team/tslearn
-   DTW 文档: https://tslearn.readthedocs.io/

Query：

``` text
Q.shape = (m, 7)
```

Candidate：

``` text
C.shape = (m, 7)
```

使用 constrained DTW：

``` python
dtw(
    Q,
    C,
    global_constraint="sakoe_chiba",
    sakoe_chiba_radius=r
)
```

------------------------------------------------------------------------

## 14. 必须使用 Constrained DTW

禁止 unrestricted DTW。

否则 DTW 可能通过过度扭曲时间轴，把实际结构明显不同的行情强行匹配起来。

默认：

``` python
sakoe_chiba_radius = max(
    1,
    round(m * 0.08)
)
```

例如：

``` text
m=20  → radius≈2
m=60  → radius≈5
m=120 → radius≈10
```

即允许约 ±8% 的局部时间伸缩。

------------------------------------------------------------------------

## 15. Weighted DTW Ground Distance

时间点 `q_t` 和 `c_s` 的 ground distance：

``` text
d(q,c)^2 =
Σ_j w_j * (q_j - c_j)^2
```

实现时可以预先：

``` python
Q_weighted = Q * np.sqrt(weights)
C_weighted = C * np.sqrt(weights)
```

再调用标准多维 DTW。

------------------------------------------------------------------------

## 16. Aligned Distance

不能只使用 DTW。

DTW 会主动解释时间上的差异，因此还需要一个不允许 warp 的逐交易日距离：

``` python
aligned_rmse = sqrt(
    mean(
        sum(
            weights * (Q - C)**2,
            axis=1
        )
    )
)
```

两者含义：

``` text
DTW
= 允许轻微时间伸缩后有多像

Aligned RMSE
= 原始交易日一一对应有多像
```

------------------------------------------------------------------------

## 17. Total Return Difference

z-normalization 会弱化涨跌幅度。

例如 +20% 与 +5% 的走势可能拥有几乎相同的轮廓，但投资意义并不完全相同。

因此额外计算：

``` python
query_total_return = np.log(
    query.close[-1] / query.close[0]
)

candidate_total_return = np.log(
    candidate.close[-1] / candidate.close[0]
)

return_diff = abs(
    query_total_return -
    candidate_total_return
)
```

------------------------------------------------------------------------

## 18. Volatility Difference

计算：

``` python
query_vol = std(
    log(close_t / close_t_minus_1)
)

candidate_vol = std(...)
```

定义：

``` python
vol_diff = abs(
    log(
        (candidate_vol + eps)
        /
        (query_vol + eps)
    )
)
```

用于区分走势轮廓相同但波动幅度完全不同的情况。

------------------------------------------------------------------------

## 19. 最终距离与 Score

Stage 2 每个 candidate 得到：

``` text
D_mass
D_dtw
D_aligned
D_return
D_vol
```

不要直接相加原始值，因为不同 metric 的尺度不同。

在 Stage 2 候选集合内部，对每个 metric 做 rank normalization：

``` python
rank_metric = rank(distance) / (N - 1)
```

其中：

``` text
0 = best
1 = worst
```

得到：

``` text
R_mass
R_dtw
R_aligned
R_return
R_vol
```

最终：

``` python
D_final = (
      0.20 * R_mass
    + 0.35 * R_dtw
    + 0.25 * R_aligned
    + 0.15 * R_return
    + 0.05 * R_vol
)

score = 1 - D_final
```

默认：

``` text
MASS          20%
DTW           35%
Aligned       25%
Total Return  15%
Volatility     5%
```

`score` 越大越相似。

MASS 保留在最终 score 中，是因为它提供整体价格轨迹 similarity，而 DTW
更偏局部 candle behavior，二者并非完全重复。

------------------------------------------------------------------------

## 20. 完整 Pipeline

``` text
                 QUERY
                   │
                   ▼
          extract OHLCV window
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
      log(close)       OHLCV features
          │                 │
          ▼                 ▼
        MASS          robust normalize
          │
          ▼
   ALL STOCK HISTORY
          │
          ▼
  sliding distance profile
          │
          ▼
 local minima + exclusion
          │
          ▼
   per-stock Top 20
          │
          ▼
    global Top 1000
          │
          ├──────────────────┐
          │                  │
          ▼                  ▼
 Multivariate DTW       aligned RMSE
          │                  │
          └────────┬─────────┘
                   │
            return diff
                   │
             volatility diff
                   │
                   ▼
           rank normalization
                   │
                   ▼
             weighted score
                   │
                   ▼
              global NMS
                   │
                   ▼
                 Top K
```

------------------------------------------------------------------------

## 21. Global NMS

Stage 1 已做 per-stock exclusion，但最终结果仍需要 Global NMS。

对于同一 symbol 的两个候选，如果：

``` text
intersection / min(lengthA, lengthB) > 0.5
```

只保留 score 更高的候选。

避免输出大量仅偏移一两个交易日的重复结果。

------------------------------------------------------------------------

## 22. 数据质量过滤

Candidate 必须满足：

``` text
OHLC finite
close > 0
open > 0
high >= max(open, close)
low <= min(open, close)
high >= low
volume >= 0
```

如果窗口存在缺失 K 线，默认直接丢弃。

不要 interpolation K 线。

------------------------------------------------------------------------

## 23. 停牌处理

如果本地数据中停牌日不存在，直接按实际交易日序列处理。

如果停牌日被人工填充为：

``` text
OHLC = previous close
volume = 0
```

建议删除 synthetic bars 后再搜索，否则 MASS
会把停牌形成的长直线当成真实形态。

------------------------------------------------------------------------

## 24. 涨跌停处理

不要删除涨跌停。

涨跌停本身属于有效 K 线形态信息。

Robust scaler + clipping 用于控制极端值对 DTW 的支配程度。

------------------------------------------------------------------------

## 25. Query 长度

建议：

``` text
MIN_WINDOW = 10
MAX_WINDOW = 250
```

最佳常用范围：

``` text
20 ~ 120 trading days
```

过短容易出现偶然匹配；过长会增加 DTW
成本，也会让"形态相似"的定义变得模糊。

------------------------------------------------------------------------

## 26. 性能策略

禁止直接：

``` python
for stock:
    for window:
        calculate_dtw()
```

Stage 1 必须使用 MASS 或等价的高效 subsequence search。

原则：

``` text
MASS = retrieval / recall engine
DTW  = reranker
```

全市场所有窗口只跑快速召回，昂贵 DTW 只作用于有限候选。

------------------------------------------------------------------------

## 27. Parallelism

股票之间完全独立，可以使用：

``` python
ProcessPoolExecutor
```

或：

``` text
joblib
multiprocessing
Ray
```

推荐起点：

``` python
ProcessPoolExecutor(
    max_workers=max(1, os.cpu_count() - 1)
)
```

每个 worker：

``` text
load one stock
↓
MASS
↓
extract local Top-N
↓
return candidate metadata
```

注意 benchmark STUMPY/Numba 自身线程与多进程组合，避免 CPU
oversubscription。

------------------------------------------------------------------------

## 28. 内存策略

除非 RAM 足够，不要一次性把全市场所有数据载入内存。

推荐：

``` text
stock
↓
ndarray
↓
MASS
↓
candidate metadata
↓
release
```

Stage 1 只保存 candidate metadata。

Stage 2 再读取 Top-N 对应 OHLCV 窗口。

------------------------------------------------------------------------

## 29. 建议模块结构

``` text
kline_similarity/
│
├── engine.py
├── features.py
├── mass_recall.py
├── dtw_rerank.py
├── scoring.py
├── filters.py
├── normalization.py
├── types.py
├── config.py
│
└── tests/
    ├── test_mass.py
    ├── test_features.py
    ├── test_dtw.py
    ├── test_nms.py
    └── test_engine.py
```

------------------------------------------------------------------------

## 30. API 设计

核心 similarity engine 必须与数据源解耦：

``` python
class KlineSimilarityEngine:

    def search(
        self,
        query_symbol: str,
        start_date: str,
        end_date: str,
        *,
        top_k: int = 50,
        history_only: bool = True,
        recall_n: int = 1000,
    ) -> list["SimilarityMatch"]:
        ...
```

数据读取抽象：

``` python
class MarketDataProvider:

    def get_symbol_data(
        self,
        symbol: str
    ) -> pd.DataFrame:
        ...

    def get_symbols(
        self
    ) -> list[str]:
        ...
```

Similarity engine 不应该关心底层是 SQL、Parquet、CSV、DuckDB、ClickHouse
或其他存储。

------------------------------------------------------------------------

## 31. Result Dataclass

``` python
@dataclass
class SimilarityMatch:

    symbol: str

    start_date: datetime
    end_date: datetime

    score: float

    mass_distance: float
    dtw_distance: float
    aligned_distance: float

    total_return_diff: float
    volatility_diff: float

    mass_rank: float
    dtw_rank: float
    aligned_rank: float
    return_rank: float
    volatility_rank: float
```

必须保留中间指标，方便解释排名原因和后续调参。

------------------------------------------------------------------------

## 32. Config

所有算法参数必须配置化：

``` python
@dataclass
class SimilarityConfig:

    local_top_n: int = 20
    recall_n: int = 1000
    top_k: int = 50

    exclusion_ratio: float = 0.25
    sakoe_chiba_ratio: float = 0.08
    clip_value: float = 5.0

    mass_weight: float = 0.20
    dtw_weight: float = 0.35
    aligned_weight: float = 0.25
    return_weight: float = 0.15
    volatility_weight: float = 0.05
```

Feature weights：

``` python
feature_weights = np.array([
    2.0,  # close return
    0.8,  # gap
    1.2,  # body
    0.8,  # upper wick
    0.8,  # lower wick
    1.0,  # range
    0.6,  # volume change
])
```

------------------------------------------------------------------------

## 33. Volume 开关

必须支持：

``` python
use_volume = True
```

以及：

``` python
use_volume = False
```

建议 benchmark：

``` text
OHLC only
vs
OHLCV
```

默认可以启用 Volume，但保持较低权重。

------------------------------------------------------------------------

## 34. 可选搜索模式

未来可支持：

``` python
mode = "price" | "candle" | "balanced"
```

### price

偏重走势轮廓：

``` text
MASS ↑
DTW ↓
```

### candle

偏重蜡烛图结构：

``` text
MASS ↓
DTW ↑
```

### balanced

默认：

``` text
MASS          20%
DTW           35%
Aligned       25%
Return        15%
Volatility     5%
```

第一版只实现 `balanced` 即可。

------------------------------------------------------------------------

## 35. 测试要求

必须实现 synthetic tests。

### Test A：完全一样

``` text
Q == C
```

距离应接近 0，并成为第一名。

### Test B：价格尺度不同

``` text
Q = [10, 11, 12, 11, 13]
C = [100, 110, 120, 110, 130]
```

Stage 1 应认为高度相似。

### Test C：形状相反

``` text
Q 上涨
C 下跌
```

应显著降低排名。

### Test D：轻微时间错位

构造：

``` text
Q：上涨 → 横盘 → 下跌
C：上涨 → 稍长横盘 → 下跌
```

Constrained DTW 应认为较相似。

### Test E：过度 Warp

构造时间结构明显不同的数据。

由于 Sakoe-Chiba constraint，不应获得异常高 similarity。

### Test F：重复窗口

同一股票附近：

``` text
i
i+1
i+2
```

NMS 后只保留代表性窗口。

------------------------------------------------------------------------

## 36. Benchmark

实现：

``` text
benchmark.py
```

记录：

``` text
query length
stock count
total bars
MASS runtime
Stage2 runtime
total runtime
peak RAM
```

测试：

``` text
m = 20
m = 60
m = 120
```

以及：

``` text
recall_n = 100
recall_n = 500
recall_n = 1000
recall_n = 2000
recall_n = 5000
```

确认默认 `1000` 是否在本地数据规模和硬件上合适。

------------------------------------------------------------------------

## 37. Recall Quality Benchmark

比纯 runtime 更重要。

随机抽取真实历史窗口 `Q`，人工生成：

``` text
Q_noise
Q_scaled
Q_time_shifted
Q_vol_modified
```

插入测试 universe。

验证：

``` text
Recall@100
Recall@500
Recall@1000
```

Stage 1 的目标不是负责最终排名，而是尽量保证真正优秀的 candidate 能进入
Stage 2。

宁可适度增加 `recall_n`，也不要让高质量候选在召回阶段被误杀。

------------------------------------------------------------------------

## 38. 第一版明确不要实现

不要第一版实现：

``` text
CNN
AutoEncoder
Transformer
FAISS
LanceDB
learned metric
GPU neural network
Soft-DTW
shapeDTW
MACD similarity
RSI similarity
KDJ similarity
```

技术指标本质上通常仍是 OHLCV 的 deterministic
transform，第一版加入大量指标容易重复计算同一信息。

第一版保持：

``` text
Raw price structure
+
Raw candle structure
+
Volume
```

------------------------------------------------------------------------

## 39. 依赖与算法来源

### STUMPY / MASS

GitHub：

https://github.com/stumpy-dev/stumpy

Documentation：

https://stumpy.readthedocs.io/

MASS API：

https://stumpy.readthedocs.io/en/latest/api.html

用途：

``` text
Query vs 长历史序列中的所有等长 subsequence
```

作为 Stage 1 全市场快速召回算法。

### tslearn / DTW

GitHub：

https://github.com/tslearn-team/tslearn

Documentation：

https://tslearn.readthedocs.io/

用途：

``` text
Multivariate DTW
+
Sakoe-Chiba global constraint
```

作为 Stage 2 精排算法。

### SciPy

Documentation：

https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

用于距离定义、辅助数值计算和验证。

### 核心依赖

``` text
numpy
pandas
scipy
stumpy
tslearn
```

可选：

``` text
numba
joblib
pyarrow
```

------------------------------------------------------------------------

## 40. 开发顺序

### Phase 1：正确性

``` text
单进程 MASS
↓
候选召回
↓
DTW rerank
↓
Rank Fusion
↓
NMS
↓
Top-K
```

先肉眼确认搜索结果合理。

### Phase 2：性能

``` text
multiprocessing
↓
cache
↓
Numba
↓
memory optimization
```

### Phase 3：Profiling

``` text
profiling
↓
定位真实瓶颈
↓
针对性优化
```

不要为了提前优化牺牲算法可验证性。

------------------------------------------------------------------------

## 41. 最终验收标准

给定：

``` text
股票 A
日期 T0 ~ T1
```

程序必须：

1.  自动得到 Query 的实际交易日数量 `m`。
2.  使用 `log(close)` + MASS 搜索全市场所有历史等长窗口。
3.  对每只股票执行 exclusion-zone 去重。
4.  汇总全市场并保留约 Top 1000 recall candidates。
5.  为 Query / Candidate 构造：
    -   return
    -   gap
    -   body
    -   upper wick
    -   lower wick
    -   range
    -   volume change
6.  使用全市场统计量进行 robust normalization。
7.  执行 weighted multivariate constrained DTW。
8.  计算 aligned distance。
9.  计算 total return difference。
10. 计算 volatility difference。
11. 对各距离执行 rank normalization。
12. 执行 weighted rank fusion。
13. 执行 global NMS。
14. 返回 Top-K。

输出至少包含：

``` text
rank
symbol
start_date
end_date
score
mass_distance
dtw_distance
aligned_distance
return_diff
volatility_diff
```

------------------------------------------------------------------------

## 42. 给 Codex 的架构约束

**必须保持两阶段搜索设计：**

``` text
MASS = retrieval / recall engine
DTW  = reranker
```

禁止对全市场所有滑动窗口直接执行 DTW。

推荐基线：

``` text
全A历史日K
        │
        ▼
    log(close)
        │
        ▼
MASS / z-normalized Euclidean
        │
        ▼
   全市场 Top 1000
        │
        ▼
7-dimensional OHLCV representation
        │
        ▼
    Robust Scaling
        │
        ├──────────────┐
        ▼              ▼
Constrained DTW    Aligned RMSE
        │              │
        └──────┬───────┘
               │
      Return + Volatility
               │
               ▼
          Rank Fusion
               │
               ▼
          Global NMS
               │
               ▼
             Top K
```

这套架构的关键优势是：

-   无需模型训练；
-   全流程可解释；
-   MASS 负责大规模快速召回；
-   DTW 只处理有限候选；
-   同时保留价格轮廓、蜡烛结构、收益幅度与波动信息；
-   每一层都可独立 benchmark；
-   后续可单独替换 recall 或 reranker，而无需重构整个搜索系统。
