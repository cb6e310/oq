
# oq K 线搜索算法改进引导：从等长窗口到形态、结构与可变时长匹配

> 适用仓库：cb6e310/oq
> 适用目录：strategies/kline_similarity/
> 基线提交：964e4b1
> 目标：让“拉高/打压后的洗盘”这类行情按阶段、方向、幅度、波动和事件顺序匹配，而不是要求候选区间必须拥有与查询区间完全相同的 K 线根数。

## 1. 先明确要改什么

当前引擎已经具备合理的两阶段框架：

~~~text
快速召回（close-shape / candle-structure / event）
        ↓
OHLCV 特征 + 约束 DTW 精排
        ↓
rank fusion + NMS
~~~

但几乎每一层都把查询长度 m = len(query) 当成候选长度：

- search() 用 idx : idx + m 枚举候选窗口；
- _mass_distances() 只产生长度为 m 的窗口；
- _structure_distances() 和 _event_window_distances() 也只比较等长切片；
- _constrained_dtw() 虽然允许轻微时间扭曲，但输入矩阵仍是 (m, features) 对 (m, features)；
- SimilarityMatch.window_length 最终总是等于查询长度。

这会漏掉下面几类本来应该相似的历史区间：

1. 同一个拉升—回撤—洗盘结构，因为某次洗盘多持续了 1～3 根 K 线而被漏掉。
2. 同一个阶段顺序，但某个阶段内部少一根低波动 K 线，被等长召回挡在候选集之外。
3. 走势拐点、回撤深度、反弹方向和成交量变化都相近，但局部节奏快慢不同，被严格的一一对应距离判为不相似。
4. 收盘价轨迹不够接近，但实体、影线、跳空、放量和事件顺序相同，只因第一阶段偏重 close 而没有进入 DTW。

本改造的核心不是“把 DTW 放宽到任意变形”，而是把匹配单位从“固定根数”改成：

> 阶段序列 + 结构关系 + 有边界的时间伸缩。

允许低信息 K 线插入或删除，保留关键拐点和事件顺序；允许同一阶段快一点或慢一点，不允许通过无限重复把完全不同的行情硬对齐。

## 2. 目标行为和非目标行为

### 2.1 应该被视为相似

- 上攻 → 高位钝化 → 回撤/洗盘 → 再次转强的阶段顺序一致；
- 关键高点、低点、回撤深度和反弹方向接近；
- 候选比查询多一两根低波动 K 线，或少一两根低波动 K 线；
- 同一阶段内的成交量放大、缩小、长上影或长下影出现顺序相近；
- 价格绝对值不同，但归一化收益、波动和结构关系相近。

### 2.2 不应该被误判为相似

- 上攻之后直接崩跌，却被拉伸成“上攻—洗盘—再转强”；
- 关键涨停、跌停、长上影拐点被跳过；
- 通过大量一对多、多对一配对把一段横盘硬配成急涨急跌；
- 只有收盘价曲线相似，实体、影线和事件结构完全不同；
- 候选长度差异过大，已经不是同一个行情阶段。

## 3. 总体架构

保持现有“召回便宜、精排昂贵”的原则，把等长 Stage 1 改成多长度粗召回，把 Stage 2 改成阶段约束的弹性对齐。

~~~mermaid
flowchart TD
    A[查询 K 线] --> B[特征与阶段提取]
    B --> C[多长度粗召回]
    C --> D[阶段约束弹性对齐]
    D --> E[绝对质量分与 NMS]
~~~

建议的完整流程：

~~~text
Query
  ├─ 连续特征：收益、实体、影线、波动、量能
  ├─ 离散状态：方向、波动档位、K 线类型、事件
  └─ 锚点/阶段：拐点、拉升、回撤、洗盘、反转
        ↓
每只股票、多个候选长度 L
  ├─ close shape 粗距离
  ├─ candle structure 粗距离
  ├─ 状态/事件签名粗距离
  └─ 锚点顺序与长度合法性过滤
        ↓ 只保留有限 metadata
有限候选
  ├─ 阶段序列对齐
  ├─ 带插入/删除代价的状态序列距离
  ├─ 带斜率限制的多变量弹性 DTW
  ├─ 回撤、反弹、波动和成交量摘要
  └─ 长度差、关键事件错位惩罚
        ↓
校准后的综合距离
        ↓
同股票区间 NMS + 全局 Top-K
~~~

## 4. 形态表示：不要只保存收盘价

### 4.1 连续特征

沿用 engine.py 中的 OHLCV 特征，但建议把影线和波动从“前收盘价归一化”升级为同时使用 ATR 归一化。一个因果、可解释的特征向量可以是：

~~~python
r_close       = log(close / prev_close)
gap           = log(open / prev_close)
body_return   = log(close / open)
upper_wick    = (high - max(open, close)) / (atr + eps)
lower_wick    = (min(open, close) - low) / (atr + eps)
range_atr     = (high - low) / (atr + eps)
volume_change = robust_z(log1p(volume) - log1p(prev_volume))
~~~

其中 atr 使用只依赖当前及历史数据的滚动或 EWM 统计；搜索模式若要求严格因果，禁止 center=True。价格、成交量和波动分别做 robust scaling，避免涨跌停或异常量柱支配距离。

再增加三类摘要特征，用于区分“轮廓像但实际幅度不同”的候选：

~~~python
total_return   = log(close[-1] / close[0])
max_drawdown   = max(previous_peak - close) / close_at_peak
max_runup      = max(close - previous_trough) / close_at_trough
pullback_depth = drawdown_from_last_impulse_high
~~~

这些摘要不替代逐根特征，而是作为最终评分的独立项。

### 4.2 离散状态

为每根 K 线生成低维状态，方便用编辑距离处理“多一根/少一根”：

~~~text
direction  = {up, flat, down}
volatility = {low, normal, high}
body_type  = {wide_body, normal_body, doji_like}
wick_type  = {upper_rejection, lower_rejection, balanced}
volume     = {low_effort, normal_effort, high_effort}
~~~

状态不应使用过细的标签。第一版建议把状态编码成一个 tuple，并提供代价矩阵：同方向替换代价低，up ↔ down 代价高，normal ↔ high volatility 的代价低于方向反转。

### 4.3 关键锚点和阶段

形态匹配需要少量稳定锚点，不需要把每根 K 线都当作等价位置。可采用：

- 以 ATR 为阈值的因果拐点检测，或离线模式下使用带确认延迟的 ZigZag；
- 局部高点、低点、放量柱、涨跌停、长影线、波动突变；
- 斜率、波动和成交量状态的变点。

每个锚点保存：

~~~python
Anchor(
    index,                  # 原始 K 线位置
    kind,                   # peak / trough / impulse / rejection / event
    t_norm,                 # 在窗口中的归一化位置 [0, 1]
    level_norm,             # 相对窗口起点或 ATR 的价格位置
    salience,               # 关键程度
    state,
)
~~~

阶段可以先使用少量可解释标签：

~~~text
impulse_up / impulse_down
exhaustion
pullback
washout
rebound
range
~~~

这些标签不建议一开始用训练模型生成。先用规则和连续特征组合，便于调试、回测和人工检查。规则不确定时产生“候选阶段 + 置信度”，不要立即硬过滤。

## 5. 可变长度搜索契约

### 5.1 长度范围

对查询长度 m，生成有限长度集合，而不是枚举所有可能长度：

~~~python
delta = min(max(2, round(0.20 * m)), 20)
min_len = max(3, m - delta)
max_len = m + delta
lengths = sorted(set(
    [min_len, max_len, m] +
    [round(m * r) for r in (0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35)]
))
~~~

实际使用时要限制长度数量，例如最多 9 个。短查询使用绝对偏移，长查询使用比例；不要让 5 根 K 线的查询被无意中扩展到 20 根。

推荐配置：

~~~python
@dataclass(frozen=True)
class SimilarityConfig:
    length_mode: str = "elastic"       # elastic / fixed
    min_length_ratio: float = 0.75
    max_length_ratio: float = 1.35
    max_length_delta: int = 20
    max_length_candidates: int = 9
    min_window_length: int = 8
    max_window_length: int = 250
    normalized_warp_band: float = 0.10
    max_repeat_step: int = 2
    gap_penalty: float = 0.35
    salient_gap_multiplier: float = 2.5
~~~

length_mode="fixed" 必须保留，作为线上回退、性能对照和回归基线。

### 5.2 候选元数据

Stage 1 不复制窗口 DataFrame，只保存：

~~~python
RecallCandidate(
    symbol,
    start_idx,
    end_idx,
    length,
    coarse_distance,
    structure_distance,
    state_distance,
    event_distance,
    phase_signature,
    recall_paths,
)
~~~

只有进入全局 recall_n 的候选才重新读取 OHLCV 窗口并计算完整特征。长度放进元数据后，后续评分和 NMS 才能正确区分 m-2、m 和 m+3 的候选。

## 6. Stage 1：多长度、低成本召回

### 6.1 粗粒度时间归一化

对于候选原始长度 L，先把候选窗口按固定数量的时间桶聚合到 K 个桶，再与查询的 K 个桶比较。这个步骤只用于召回，不用于最终展示和精排。

OHLCV 聚合规则：

~~~text
open   = 首根 open
high   = 桶内 high 最大值
low    = 桶内 low 最小值
close  = 末根 close
volume = 桶内 volume 求和
~~~

建议 K = clip(round(m / 2), 8, 32)。候选仍然保留原始 K 线，最终 DTW 不使用插值后的伪 K 线。

粗表示至少包含：

1. 累积对数收益曲线；
2. 每桶实体、影线、波动和量能摘要；
3. 锚点方向和阶段签名。

### 6.2 三条独立召回路径

每只股票、每个候选长度 L 至少保留三条路径：

~~~text
shape path     = 累积收益 / close shape 的快速距离
structure path = 实体、影线、跳空、波动、量能的粗距离
state path     = 阶段和离散状态签名的近似距离
~~~

若查询包含高置信度涨跌停、长影线或明显放量事件，再增加 event path。不能让 close-shape 的候选独占所有局部召回名额；每条路径都要保留最低配额，再合并去重。

### 6.3 局部最优和全局 bounded heap

~~~python
for symbol in provider.get_symbols():
    frame = valid_bars(get_bars(symbol, timeframe))
    for length in candidate_lengths(query_length):
        if len(frame) < length:
            continue

        shape_dist = coarse_shape_profile(query, frame, length)
        struct_dist = coarse_structure_profile(query, frame, length)
        state_dist = coarse_state_profile(query, frame, length)

        legal = build_legal_mask(
            frame=frame,
            query_start=query_start,
            history_only=history_only,
            exclude_query=exclude_query,
        )

        candidates = union_local_minima(
            shape_dist, struct_dist, state_dist,
            legal=legal,
            per_path_top_n=config.local_top_n,
            exclusion_radius=max(3, round(length * config.exclusion_ratio)),
        )

        bounded_global_heap.push_many(
            metadata_only(candidates),
            limit=config.recall_n,
        )
~~~

现有 _mass_distances() 可以先保留为正确性参考实现，再增加真正的 distance-profile 实现。多长度版本需要逐步验证：小数据上与暴力实现一致，大数据上再考虑滚动均值/方差、FFT 或分块扫描。

### 6.4 锚点条件召回

对“拉高后洗盘”这类结构，建议再增加一条锚点条件召回：

1. 从查询提取高点、回撤低点、反弹起点等锚点；
2. 在候选股票上提取同类型锚点；
3. 只枚举锚点顺序一致、归一化位置差在容忍范围内的窗口；
4. 窗口长度可在锚点间距上下浮动，不要求每段长度相等。

锚点条件召回的作用是提高覆盖率，不能直接取代连续距离。锚点检测失败时必须允许回退到 shape/structure/state 三条路径。

## 7. Stage 2：阶段约束的弹性对齐

### 7.1 不能只把普通 DTW 放宽

无约束 DTW 会通过大量重复配对制造假相似。可变长度 DTW 必须同时满足：

- 时间顺序不可逆；
- 在归一化时间轴上限制偏离；
- 限制连续一对多、多对一的次数；
- 跳过低信息 K 线便宜，跳过关键锚点昂贵；
- 关键事件顺序不允许改变。

### 7.2 时间归一化 Sakoe-Chiba band

不同长度不能直接使用 abs(i-j) <= radius。应使用归一化位置：

~~~python
tq = i / max(query_length - 1, 1)
tc = j / max(candidate_length - 1, 1)
legal = abs(tq - tc) <= normalized_warp_band
~~~

这样 m=20 对 L=22 和 m=120 对 L=132 使用的是同一个时间伸缩语义。起始值建议 0.08～0.15，并通过标注样本调节。

### 7.3 带 salience 的弹性 DTW

对连续特征序列 Q 和 C，定义路径代价：

~~~text
D_elastic(Q, C) = min_path(
    Σ ground_distance(q_i, c_j)
  + Σ gap_cost(skipped_bar)
  + Σ slope_penalty(step)
) / path_length
~~~

允许的路径步先限制为：

~~~text
(1, 1)  普通一一对应
(1, 2)  查询一根对应候选两根
(2, 1)  查询两根对应候选一根
~~~

如果复用滚动行 DTW 实现，也可以使用 (1,0)、(0,1)，但必须限制连续水平/垂直步数为 max_repeat_step=2，并对每次跳过支付 gap cost。

单根 K 线的跳过代价：

~~~python
gap_cost(bar) = config.gap_penalty * (
    1.0 + config.salient_gap_multiplier * bar.salience
)
~~~

低波动、低成交量、无事件的 K 线可被便宜地插入或删除；涨跌停、长影线、显著拐点和高成交量柱的跳过代价显著更高。

### 7.4 先对齐阶段，再对齐 K 线

推荐两层对齐，而不是一次性让 DTW 解决所有问题：

~~~text
阶段序列对齐
    ↓
每一对阶段内部做带 band 的弹性 DTW
    ↓
汇总阶段距离 + 阶段边界距离
~~~

阶段序列可以用加权编辑距离或单调动态规划对齐：

- 相同阶段替换代价低；
- pullback ↔ washout 代价低于 pullback ↔ impulse_up；
- 插入/删除低置信度阶段的代价低；
- 插入/删除高置信度事件阶段的代价高；
- 阶段顺序反转直接产生高惩罚。

阶段对齐后，在每一段内部运行局部弹性 DTW。这样候选多一根 K 线时，额外 K 线会留在某个阶段内部，而不会导致整个窗口的时间轴任意漂移。

### 7.5 离散状态编辑距离

连续 DTW 负责数值形态，离散状态编辑距离负责“多一根/少一根”的语义：

~~~python
state_distance = weighted_levenshtein(
    query_states,
    candidate_states,
    substitution_cost=state_cost_matrix,
    insertion_cost=salience_aware_gap_cost,
    deletion_cost=salience_aware_gap_cost,
)
~~~

建议把它做成独立指标，而不是把状态直接塞进连续特征后交给 DTW。这样可以解释：候选是因为形态数值相近，还是因为阶段/状态顺序相近。

## 8. 综合评分

不要继续只用候选集合内部的 rank 作为最终质量定义。rank 可以用于稳定排序，但无法跨查询比较，也不能表达“这个候选本身是否足够好”。

第一版可以使用如下距离：

~~~text
D_total =
    0.30 * D_elastic_shape
  + 0.25 * D_state_edit
  + 0.20 * D_anchor
  + 0.10 * D_path_summary
  + 0.08 * D_event_order
  + 0.05 * P_length
  + 0.02 * D_volume_extra
~~~

各项含义：

| 项目 | 含义 |
| --- | --- |
| D_elastic_shape | 连续 OHLCV 特征的带约束弹性 DTW，按路径长度归一化 |
| D_state_edit | 离散方向、波动、K 线类型和量能状态的加权编辑距离 |
| D_anchor | 拐点类型、归一化位置、回撤深度、反弹幅度的锚点距离 |
| D_path_summary | 总收益、最大回撤、最大反弹、波动率等摘要差异 |
| D_event_order | 涨跌停、长影线、放量等关键事件的顺序和相对位置差异 |
| P_length | 候选总长度与查询长度的软惩罚 |
| D_volume_extra | 当量能被单独评估时的补充差异 |

长度惩罚不要使用硬阈值，可先用：

~~~python
P_length = abs(log(candidate_length / query_length))
~~~

并在“低信息 K 线插入/删除”的场景下降低惩罚；如果长度差导致关键阶段边界错位，则由 D_anchor 和 D_state_edit 提高总距离。

最终可以输出：

~~~python
score = calibrated_similarity(D_total)
~~~

在没有标注数据前，score 只称为“综合相似分”，不要称为概率。校准数据足够后，再使用 isotonic regression 或 logistic calibration 将距离映射到人工相关率。

## 9. 建议的数据结构和 API

### 9.1 SimilarityMatch 增加字段

~~~python
@dataclass
class SimilarityMatch:
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    score: float

    query_length: int
    window_length: int
    length_ratio: float
    match_mode: str                 # fixed / elastic

    elastic_shape_distance: float
    state_distance: float
    anchor_distance: float
    event_distance: float
    total_return_diff: float
    volatility_diff: float
    length_penalty: float
    warp_path_length: int
    skipped_query_bars: int
    skipped_candidate_bars: int

    # 保留现有字段，便于 UI 和旧脚本兼容
    mass_distance: float
    dtw_distance: float
    aligned_distance: float
    start_idx: int
    end_idx: int
~~~

### 9.2 search() 参数

建议新增但保持默认兼容：

~~~python
def search(
    self,
    query_symbol,
    start_date,
    end_date,
    *,
    timeframe="1d",
    top_k=None,
    history_only=True,
    recall_n=None,
    exclude_query=True,
    match_mode=None,          # None -> config.length_mode
    min_length=None,
    max_length=None,
    return_diagnostics=False,
):
    ...
~~~

match_mode="fixed" 时，结果应尽量与当前实现一致；match_mode="elastic" 时，返回 length_ratio、阶段距离和弹性对齐诊断。

### 9.3 推荐拆分的模块

~~~text
strategies/kline_similarity/
├── engine.py                 # API、过滤、流程编排
├── features.py               # OHLCV、ATR、状态、salience
├── phases.py                 # 锚点与阶段提取
├── variable_recall.py        # 多长度粗召回与 bounded heap
├── elastic_align.py          # 阶段对齐、弹性 DTW、状态编辑距离
├── scoring.py                # 距离组合、校准、解释字段
├── filters.py                # 历史窗口、自身重叠、数据质量
└── tests/
    ├── test_features.py
    ├── test_phases.py
    ├── test_variable_recall.py
    ├── test_elastic_align.py
    ├── test_scoring.py
    └── test_engine.py
~~~

不必一次性重构全部文件。可以先在 engine.py 内实现最小版本，确认结果和指标后再拆模块。

## 10. 最小可实施版本

第一轮不需要训练模型，按下面顺序即可验证收益。

### Step 1：保留 fixed 基线

增加 feature flag：

~~~python
SimilarityConfig(length_mode="fixed")
~~~

所有新结果都与 fixed 结果并行保存，确保能做回归比较。

### Step 2：增加多个 L 的粗召回

只实现 m-2, m-1, m, m+1, m+2，先验证“一根多、一根少”的核心诉求。候选窗口仍使用现有 close-shape 和 structure 距离，但分别按 L 计算并加入长度字段。

### Step 3：将最终 DTW 改成不同长度可用

先使用归一化时间 band 和有限重复步，确保：

- m 对 m 的结果接近旧 DTW；
- m 对 m+1 的中间低信息 K 线可以被吸收；
- 关键事件不能被轻易跳过。

### Step 4：加入状态编辑距离和 salience gap cost

这一步通常比继续调 close-shape 权重更能改善“神态/结构”匹配。

### Step 5：加入阶段/锚点距离

先只使用 2～4 个稳定锚点：起始方向、主拐点、回撤低点、结束方向。锚点过多会把检测误差直接放大到排序中。

### Step 6：再扩大长度范围

核心用例稳定后，再从 m±2 扩到 0.75m～1.35m，并通过 bounded heap 控制 Stage 1 候选数量。

## 11. 伪代码：弹性搜索主流程

~~~python
def elastic_search(query, universe, cfg):
    q_repr = extract_representation(query, cfg)
    lengths = build_length_grid(len(query), cfg)
    heap = BoundedCandidateHeap(cfg.recall_n)

    for symbol in universe:
        frame = load_and_validate(symbol)
        if len(frame) < min(lengths):
            continue

        for length in lengths:
            legal = legal_window_mask(
                frame, length,
                query=query,
                history_only=cfg.history_only,
                exclude_query=cfg.exclude_query,
            )

            coarse = recall_profiles(
                query=q_repr,
                frame=frame,
                length=length,
                paths=("shape", "structure", "state", "event"),
            )
            coarse[~legal] = INF

            for item in local_minima_by_path(
                coarse,
                local_top_n=cfg.local_top_n,
                exclusion_radius=round(length * cfg.exclusion_ratio),
            ):
                heap.push(RecallCandidate(
                    symbol=symbol,
                    start_idx=item.start_idx,
                    end_idx=item.start_idx + length - 1,
                    length=length,
                    coarse_distance=item.distance,
                    recall_paths=item.paths,
                ))

    metrics = []
    for candidate in heap.items():
        c_frame = load_window(candidate)
        c_repr = extract_representation(c_frame, cfg)

        phase_alignment = align_phases(
            q_repr.phases,
            c_repr.phases,
            gap_cost=salience_aware_gap_cost,
        )
        elastic = phase_constrained_dtw(
            q_repr.continuous,
            c_repr.continuous,
            phase_alignment=phase_alignment,
            normalized_band=cfg.normalized_warp_band,
            max_repeat_step=cfg.max_repeat_step,
        )
        state = weighted_state_edit_distance(
            q_repr.states,
            c_repr.states,
            salience=q_repr.salience + c_repr.salience,
        )

        metrics.append(score_candidate(
            candidate=candidate,
            elastic=elastic,
            state=state,
            anchor=anchor_distance(q_repr.anchors, c_repr.anchors),
            summary=summary_distance(q_repr, c_repr),
            event=event_order_distance(q_repr.events, c_repr.events),
            length=length_penalty(len(query), len(c_frame)),
        ))

    return global_interval_nms(sort_by_score(metrics), cfg.top_k)
~~~

## 12. NMS 和重复结果

可变长度会让同一段行情产生多个长度版本，因此 NMS 必须按 bar 区间处理，不能只比较起点日期。

对同一股票的候选 A、B：

~~~python
intersection = max(
    0,
    min(A.end_idx, B.end_idx) - max(A.start_idx, B.start_idx) + 1,
)
overlap = intersection / max(1, min(A.length, B.length))
~~~

当 overlap > 0.50 且阶段签名相同或高度相似时，只保留总距离更小者。对于周线和月线仍使用 bar index，不使用日历天数。

如果用户希望看到同一行情在不同“阶段边界”下的多个解释，可以把结果先聚类，再每个 cluster 保留一个主结果和一个备选结果，而不是直接把所有长度变体全部展示。

## 13. 数据泄漏和边界条件

### 13.1 历史搜索

history_only=True 时，候选窗口的最后一根必须早于查询起点：

~~~text
candidate.end_date < query.start_date
~~~

过滤必须发生在每个长度的 local minima 之前，不能先选再删。

### 13.2 锚点的确认延迟

离线研究可以使用“已确认的拐点”，但在线/预测模式必须记录锚点确认所需的后续 K 线，不能把未来确认信息泄漏到查询时点。建议给锚点增加：

~~~python
confirmed_at
confirmation_lag
~~~

并在严格模式下用 confirmed_at <= query_end 过滤。

### 13.3 停牌和合成 K 线

不要用补零或重复前收盘价填充停牌来制造连续形态。低信息停牌柱若确实存在，应显式标记为 synthetic，并让它们在状态编辑距离中产生较低或独立的代价，而不是当作真实走势。

### 13.4 查询太短

当 m < 8 时，拐点和阶段估计不稳定，默认退回 fixed 模式，或者只允许 m-1, m, m+1 三种长度。不要对极短查询使用大范围弹性。

## 14. 测试和评估

### 14.1 必须先写的单元测试

构造可控的合成行情，至少覆盖：

1. 查询长度为 30，候选长度为 29，只有一根低波动 K 线被删除；应仍排名靠前。
2. 查询长度为 30，候选长度为 31，中间插入低波动 K 线；应仍排名靠前。
3. 候选多出一根涨停或长影线关键柱；应受到明显惩罚，不能因为长度弹性而免费跳过。
4. 上攻—洗盘—反弹与上攻—崩跌—反弹长度相同；阶段和状态距离应区分两者。
5. 候选长度差距超过配置范围；不能进入 Stage 2。
6. 查询自身、未来窗口和重叠窗口在每个长度上都不会占用 local recall 名额。
7. 等长输入下，新弹性 DTW 与旧 constrained DTW 的排序高度一致。
8. (1,2)、(2,1) 连续步数超过上限时被拒绝。
9. 周线/月线 NMS 按 bar 区间去重。

### 14.2 人工标注集

至少准备三类查询：

~~~text
拉升后的洗盘
下跌后的反抽
横盘突破前后的蓄势
~~~

每个查询标注：

- 相似等级：强相关 / 可参考 / 不相关；
- 关键阶段边界；
- 关键锚点；
- 是否允许多一根或少一根 K 线；
- 关键事件是否必须存在。

比较 fixed 与 elastic：

~~~text
Recall@20 / Recall@100
nDCG@20
Top-K 人工相关率
关键阶段顺序准确率
关键事件保留率
长度差覆盖率
p50 / p95 延迟
峰值内存
~~~

训练、调参和测试按时间切分，不能让同一段行情的相邻窗口同时出现在标注训练和评估中。

### 14.3 消融实验

依次关闭一项功能，观察收益：

~~~text
fixed vs elastic length
无状态距离 vs 有状态编辑距离
无锚点 vs 有锚点
固定 gap cost vs salience gap cost
只 close-shape vs 多路径召回
无 length penalty vs 有 length penalty
~~~

如果某项只增加耗时、不提高 Recall 或人工相关率，就不要默认开启。

## 15. 性能实施顺序

~~~text
1. 先实现 m±2 的多长度召回
2. 只保存 candidate metadata
3. 用小数据暴力实现校验弹性距离
4. 实现归一化 band + 有限重复步
5. 加阶段和状态距离
6. 用 bounded heap 控制 recall_n
7. benchmark 找到真正瓶颈
8. 再考虑 FFT、Numba、并行和缓存
~~~

不要一开始就对全市场所有窗口运行完整可变长度 DTW。Stage 1 仍然必须是便宜的近似召回，Stage 2 只处理 bounded recall_n。

## 16. 建议的灰度发布方式

### Phase 0：只加诊断

记录每次查询：

~~~text
query_length
candidate_length 分布
各召回路径数量
非法窗口数量
进入 Stage 2 的数量
被 gap 跳过的 K 线数量
阶段对齐失败原因
~~~

### Phase 1：双跑，不改变 UI 主排序

fixed 和 elastic 同时搜索，UI 默认展示 fixed，后台保存 elastic 指标。确认没有未来窗口、关键事件漏匹配和明显性能回归后，再开放切换。

### Phase 2：只对长度差小的候选启用 elastic

先开启 m-2～m+2，并把 length_ratio 和 state_distance 展示在诊断面板中。

### Phase 3：扩大到比例长度

在标注集确认收益后开启 0.75m～1.35m，同时提高阶段边界和关键事件的权重，避免长窗口噪声进入 Top-K。

### Phase 4：再做性能优化

只有在语义指标稳定后，才做 FFT distance profile、Numba DTW、进程并行和更复杂的索引。优化前后必须比较 Top-K 交集、Recall 和人工相关率。

## 17. 验收标准

达到以下条件后，才考虑把 elastic 设为默认：

- 对“插入/删除一根低信息 K 线”的合成测试，elastic 能稳定保留相似候选；
- 对关键涨跌停、长影线、主拐点的插入/删除，结果排序符合人工预期；
- 等长查询的 Top-K 与 fixed 基线保持足够高的交集，且没有系统性降级；
- 人工标注集的 Recall@20、nDCG@20 或 Top-K 相关率高于 fixed 基线；
- 历史模式下没有未来窗口和锚点确认泄漏；
- p95 延迟和峰值内存都在可接受范围内；
- UI 和导出结果明确显示 window_length、length_ratio、match_mode 和主要距离分量。

## 18. 不建议的做法

- 只把窗口长度改成可变，却继续用纯 close-shape 召回；
- 使用 unrestricted DTW，让路径任意重复；
- 把候选统一线性插值到查询长度后直接当作最终 K 线；
- 对所有 K 线使用相同的插入/删除代价；
- 把涨停、跌停和主拐点当成普通低信息柱跳过；
- 继续把候选内 rank 当成跨查询可比较的绝对相似度；
- 在没有人工标注和消融实验前，把所有新权重直接替换线上默认值；
- 为解决速度问题，先并行化一个尚未验证正确的可变长度实现。

## 19. 最终建议

这次改造最重要的顺序是：

~~~text
多长度召回
    → 阶段/锚点表示
    → 归一化时间 band
    → salience-aware gap cost
    → 阶段内弹性 DTW
    → 状态编辑距离
    → 长度与关键事件惩罚
    → 标注校准和 NMS
~~~

其中最小有效改动是：

1. 先把候选长度从单一 m 扩展到 m-2～m+2；
2. 用时间归一化 band 支持不同长度 DTW；
3. 对低信息 K 线使用低插入/删除代价；
4. 对拐点、涨跌停、长影线和放量柱使用高跳过代价；
5. 增加状态/阶段距离，避免“收盘价看起来像、结构实际上不像”。

这样可以先解决“一根多一根少”的问题，再逐步扩展到更长的阶段伸缩，而不会把搜索退化为没有约束的形状拉伸。
