# K-Line Similarity 核心性能优化指南

> 目标：用最少改动，解决历史 K 线匹配中最关键的正确性、速度和内存问题。  
> 原则：**先修正确性，再优化单进程，最后才并行。**

---

## 1. 先修两个正确性问题

### 1.1 history/self 过滤必须发生在 Top-N 之前

如果先取 `local_top_n`，再过滤未来窗口或 query 自身重叠窗口，会出现：

- 未来窗口先占满 Top-N；
- 后续再被过滤；
- 真正合法的历史候选没有机会进入候选集。

正确顺序：

```text
计算 distance profile
    ↓
提前 mask 非法窗口为 +inf
    ↓
local minima / local Top-N
```

至少提前过滤：

- `history_only=True` 时 query 之后的窗口；
- query symbol 上与 query 区间重叠的窗口。

这是 correctness 修复，不只是性能优化。

### 1.2 NMS 必须按 bar 区间重叠计算

不要用：

```text
abs(start_date_A - start_date_B) <= radius
```

因为 `radius` 是 bar 数，而日期差是 calendar day，周线/月线会错位。

应基于：

```text
[start_idx, end_idx]
```

计算：

```text
intersection / min(len(A), len(B))
```

超过阈值再抑制低排名候选。

---

## 2. Stage 1 只保留 metadata，不复制 DataFrame

Stage 1 不应给每个候选复制整段 K 线 DataFrame。

只保存：

```python
@dataclass(slots=True)
class RecallCandidate:
    symbol_id: int
    start_idx: int
    end_idx: int
    distance: float
```

不要保存：

```text
DataFrame
Series
完整 OHLCV window
feature matrix
```

只有最终进入 `recall_n` 的候选，才读取 OHLCV 做 DTW。

同时维护一个大小为 `recall_n` 的全局 heap，不要把全市场所有局部候选都留在内存里。

这样候选内存从：

```text
O(全市场局部候选数量)
```

降为：

```text
O(recall_n)
```

---

## 3. 把当前“伪 MASS”换成真正的快速 distance profile

如果 `_mass_distances()` 仍然依赖：

```python
sliding_window_view(...)
mean(axis=1)
std(axis=1)
zscore(...)
```

虽然是 NumPy 向量化，本质仍接近：

```text
O(N * M)
```

这是 Stage 1 最值得优化的地方。

建议统一接口：

```python
distance_profile(
    series,
    query,
    method="auto",
)
```

策略：

```text
短序列 / 小窗口：
    direct compiled kernel

长序列：
    FFT / MASS
```

要求：

- optimized 结果与 brute-force reference 在浮点误差范围内一致；
- 明确处理零方差 / 近零方差窗口；
- 通过 benchmark 决定 direct 与 FFT 的切换阈值。

---

## 4. DTW 只做两件事：banded + JIT

Stage 2 已经只有少量候选，不需要复杂改造。

### 4.1 改为 rolling-row banded DTW

Sakoe-Chiba band 下不需要完整 `M x M` 矩阵。

目标：

```text
memory: O(M)
compute: O(M * radius)
```

### 4.2 JIT 编译核心循环

DTW 的 Python 双层循环适合用 Numba/JIT。

优先编译：

```text
feature distance
band traversal
DP update
```

先不要上复杂的 `LB_Keogh`、early abandon 或近似 DTW，除非 profiler 证明 DTW 仍是主要瓶颈。

---

## 5. 并行放到最后

不要先把当前实现直接开 `cpu_count() - 1` 个进程。

正确顺序：

```text
修 correctness
    ↓
Stage 1 metadata-only
    ↓
bounded recall heap
    ↓
真正快速 distance profile
    ↓
banded + JIT DTW
    ↓
profile
    ↓
最后再 multiprocessing
```

并行时：

- worker 不传大 DataFrame；
- worker 只返回 candidate metadata；
- benchmark `1 / 2 / 4 / 8 ...` workers；
- 不默认 `cpu_count() - 1` 最快；
- 避免 NumPy / FFT / BLAS 内部线程与多进程产生 oversubscription。

---

# 最小实施顺序

建议只拆成 3 个 PR。

## PR 1 — Correctness

```text
1. history/self mask 提前到 local Top-N 之前
2. NMS 改成 bar interval overlap
3. 加对应 regression tests
```

## PR 2 — Stage 1 核心优化

```text
1. RecallCandidate 改成 metadata-only
2. 删除 Stage 1 candidate DataFrame copy
3. 加 bounded global recall heap
4. 实现真正的 fast distance profile
```

这是预计收益最大的一组改动。

## PR 3 — DTW + 并行

```text
1. banded rolling-row DTW
2. Numba/JIT
3. profiler 验证
4. 再决定 multiprocessing worker 数量
```

---

# 最小验收标准

### Correctness

- [ ] future window 不会挤掉合法历史候选
- [ ] query 不会误匹配自身重叠区间
- [ ] 周线/月线 NMS 正确
- [ ] optimized Stage 1 与 brute-force 小数据结果一致

### Performance

至少记录：

```text
Stage 1 runtime
Stage 2 runtime
total runtime
peak RAM
```

目标：

```text
Stage 1 明显提速
Peak RAM 大幅下降
DTW 明显提速
最终 Top-K 结果语义不变
```

---

# 最终原则

如果只能做最少工作，就抓住这五件事：

```text
1. 提前过滤非法窗口
2. 修 NMS
3. Stage 1 只存 metadata + bounded heap
4. 真正的快速 distance profile
5. banded + JIT DTW，最后再并行
```

不要优先做缓存花活、复杂剪枝或大规模 multiprocessing。

**先减少不必要的工作，再并行剩下的工作。**
