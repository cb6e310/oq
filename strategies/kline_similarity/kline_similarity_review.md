# `oq` 仓库 K 线相似搜索：修改记录与后续建议

日期：2026-10-01
审阅基线：[`cb6e310/oq`](https://github.com/cb6e310/oq)，提交 `4ac6fd7`
范围：`strategies/kline_similarity/engine.py`、命令行入口、可视化入口及对应测试。

## 结论

当前搜索先对每只股票的收盘价序列做局部召回，再把全市场候选截到 `recall_n`，最后用 OHLCV 特征和约束 DTW 精排。结果偏少有两种含义：**返回条数少**，可能来自时间范围、展示上限或原有的过滤顺序问题；**看起来真正相似的少**，则可能是单一收盘价召回遗漏了 K 线结构相近的候选。前者已有一处明确的代码修复，后者需要真实行情和人工标注样本验证。

## 已完成的代码修改

| 修改 | 原来的行为 | 现在的行为 | 影响 |
| --- | --- | --- | --- |
| 提前过滤不合格窗口 | 每只股票先挑最多 `local_top_n=20` 个局部最优窗口，再剔除查询开始日之后及与查询区间重叠的窗口 | `history_only=True` 时先裁去查询开始日及以后的行情；允许扫描未来时，先将查询股票的重叠窗口距离设为无穷大，再做局部选优 | 不合格窗口不再占用每只股票的召回名额；可减少漏召回 |
| 同距离并列排名 | `_rank` 给数值相同的距离分配不同名次，顺序影响综合 `score` | 相同距离共享该并列组的最好名次 | 等价候选获得一致的排名分数；不直接增加返回条数 |

修改仍位于本次工作区的 Git 检出中，尚未提交或推送至 GitHub。文末附完整 unified diff，包含回归测试，可供审阅和应用。

### 验证

运行：

```bash
python -m unittest discover -s strategies/kline_similarity/tests -v
git diff --check
```

结果：7 项测试通过，`git diff --check` 通过。新增用例覆盖“更近但不允许返回的未来窗口占满局部名额”“查询自身的重叠窗口占满局部名额”以及相同距离的并列排名。仓库未附带默认的 `database/processed/stock_daily_qfq.parquet`，因此测试使用合成行情；真实全市场召回提升幅度尚未量化。

## 当前结果数量的实际限制

| 环节 | 当前默认值或行为 | 检查方式 |
| --- | --- | --- |
| 查询时间限制 | `history_only=True`：候选必须在查询区间开始前结束 | 对纯形态探索，可比较 `--include-future`；用于历史回测时保留时间限制 |
| 每只股票局部召回 | `local_top_n=20`，邻近窗口由 `max(5, int(窗口长度 × 0.25))` 的半径抑制 | 可通过 Python API 修改 `SimilarityConfig(local_top_n=...)`；命令行目前没有这个选项 |
| 全市场精排候选 | `recall_n=1000` | 命令行可用 `--recall-n` 调整；增大后 DTW 耗时也会上升 |
| 最终命令行输出 | `top_k=50` | `search.py` 可用 `--top-k` 调整 |
| 图表输出 | `visualize.py` 固定 Top 10 | 它展示的条数不能用于判断底层搜索总共找到多少候选 |

例如，先用命令行检查更多候选：

```bash
python -m strategies.kline_similarity.search 600519.SH 2024-01-01 2024-03-31 \
  --timeframe 1d --top-k 100 --recall-n 3000
```

`--include-future` 适合不要求时间因果顺序的形态探索；如果检索结果用于回测或预测评估，仍应只使用查询时点之前的数据。提高 `top_k` 仅增加输出上限，不能凭空提高候选的形态质量。

## 建议按以下顺序继续优化

### 1. 给召回链路加计数与原因统计（优先）

记录每次查询的有效股票数、各股票满足时间条件的窗口数、局部召回总数、全局截断前后数量、最终输出数量，以及每个候选来自哪条召回路径。对同一组查询分别记录 `history_only`、`local_top_n`、`recall_n` 与窗口长度。这样能分辨“没有可搜索的历史窗口”“候选在局部阶段丢失”和“显示上限较小”。这是诊断建议，尚未写入代码。

### 2. 增加第二条 K 线结构召回路径（需实验）

第一阶段目前仅对 **log close 的 z 标准化轨迹**计算距离；实体、上下影线、跳空和量能只参与已召回窗口的精排。因此，收盘线不够接近但蜡烛结构相近的窗口可能根本进不了 DTW。建议并行加入基于逐根收益率、实体和影线摘要的轻量召回，与原收盘价召回合并、去重，并在固定精排预算内给两条路径分别留名额。不要直接对全市场所有窗口跑 DTW。

用至少一批人工标注的查询及相似区间，对比修改前后 `recall@100`、Top 20 的人工相关率、覆盖的股票与时段、每次查询耗时。多路径可能提高覆盖，也可能带来噪声；在这些指标改善前不应默认替换现有排序。

### 3. 将 `score` 解释为相对排名，必要时另设绝对质量标尺

当前 `score = 1 -` 五项距离的加权**候选内名次**。同一个窗口的分数会随 `recall_n`、搜索范围及其他候选变化；它不是“有 90% 的概率相似”，也不适合跨不同查询直接比较。此次修复了距离并列时的名次不一致。展示层可把它称为“综合排序分”，同时保留 `mass_distance`、`dtw_distance` 等原始指标。如果要设置“只展示真正相似”的阈值，应先在标注样本上校准距离与人工判断的关系。

### 4. 搜索变慢时，再优化第一阶段实现

`_mass_distances` 虽称 MASS 等价召回，实际使用 `sliding_window_view` 加逐窗口标准化与逐元素差值，计算及临时数组规模随“行情长度 × 查询窗口长度”增加。长查询或更大的召回预算下，可改为滚动均值/方差加 FFT 卷积的点积实现，或先分块扫描并维持局部 Top N；改动前后需对短序列和常数序列比较距离数值与排序。

### 5. 周线和月线的去重半径统一按 K 线根数计算

局部候选抑制使用数组索引半径，最终 `_global_nms` 却把同一半径当作**日历天数**比较起点。周线、月线中这两个尺度不同，最终去重可能无法抑制相邻且高度重叠的窗口。这更可能造成结果重复，而不是条数偏少；建议在修复召回与完成标注评估后处理。

## 完整代码补丁

以下补丁相对于基线提交 `4ac6fd7`，只包含引擎和测试文件的修改。

```diff
diff --git a/strategies/kline_similarity/engine.py b/strategies/kline_similarity/engine.py
index 00dfcc0..b878077 100644
--- a/strategies/kline_similarity/engine.py
+++ b/strategies/kline_similarity/engine.py
@@ -198,19 +198,20 @@ class KlineSimilarityEngine:
         limit = int(recall_n or self.config.recall_n)
         for symbol in self.provider.get_symbols():
             frame = _valid_bars(_get_bars(self.provider, symbol, tf))
+            # Apply eligibility before the per-symbol local top-N.  Otherwise
+            # recent or overlapping windows can consume every recall slot.
+            if history_only:
+                frame = frame.loc[frame.date < start].reset_index(drop=True)
             if len(frame) < m:
                 continue
             distances = _mass_distances(q_close, np.log(frame.close.to_numpy(float)))
+            if not history_only and exclude_query and str(symbol) == str(query_symbol):
+                for idx in range(len(distances)):
+                    if _overlap_ratio(frame.date.iloc[idx], frame.date.iloc[idx + m - 1], start, end) > 0.2:
+                        distances[idx] = np.inf
             radius = max(5, int(m * self.config.exclusion_ratio))
             selected = _local_minima(distances, self.config.local_top_n, radius)
             for idx, mass_distance in selected:
-                c_start, c_end = frame.date.iloc[idx], frame.date.iloc[idx + m - 1]
-                if history_only and c_end >= start:
-                    continue
-                if exclude_query and str(symbol) == str(query_symbol):
-                    overlap = _overlap_ratio(c_start, c_end, start, end)
-                    if overlap > 0.2:
-                        continue
                 candidates.append({"symbol": str(symbol), "frame": frame.iloc[idx:idx + m].reset_index(drop=True),
                                    "mass_distance": float(mass_distance)})
         candidates.sort(key=lambda x: x["mass_distance"])
@@ -329,7 +330,10 @@ def _rank(values: Iterable[float]) -> np.ndarray:
         return np.zeros(len(values))
     order = np.argsort(values, kind="mergesort")
     ranks = np.empty(len(values), dtype=float)
-    ranks[order] = np.arange(len(values), dtype=float) / (len(values) - 1)
+    sorted_values = values[order]
+    starts = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
+    first_rank = np.maximum.accumulate(np.where(starts, np.arange(len(values)), 0))
+    ranks[order] = first_rank / (len(values) - 1)
     return ranks


diff --git a/strategies/kline_similarity/tests/test_engine.py b/strategies/kline_similarity/tests/test_engine.py
index 19e7354..251a5d2 100644
--- a/strategies/kline_similarity/tests/test_engine.py
+++ b/strategies/kline_similarity/tests/test_engine.py
@@ -1,11 +1,13 @@
 import unittest
 import tempfile
 from pathlib import Path
+from unittest.mock import patch

 import numpy as np
 import pandas as pd

 from strategies.kline_similarity import KlineSimilarityEngine, SimilarityConfig, aggregate_bars, build_visual_results
+from strategies.kline_similarity.engine import _rank


 def _bars(phase=0.0, n=180):
@@ -55,6 +57,45 @@ class EngineTests(unittest.TestCase):
         result = engine.search("A", "2020-04-01", "2020-04-30", history_only=True)
         self.assertTrue(all(x.end_date < pd.Timestamp("2020-04-01") for x in result))

+    def test_ineligible_windows_do_not_consume_local_recall_slots(self):
+        provider = Provider()
+        provider.data = {"A": _bars(n=30), "B": _bars(n=30)}
+        provider.get_symbols = lambda: ["B"]
+        engine = KlineSimilarityEngine(provider, SimilarityConfig(local_top_n=1, recall_n=1, top_k=1))
+
+        def distances(query, series):
+            result = np.full(len(series) - len(query) + 1, 2.0)
+            result[0] = 0.1  # eligible historical match
+            if len(series) > 20:
+                result[20] = 0.0  # better match, but after the query
+            return result
+
+        with patch("strategies.kline_similarity.engine._mass_distances", side_effect=distances):
+            result = engine.search("A", "2020-01-15", "2020-01-17")
+        self.assertEqual(len(result), 1)
+        self.assertEqual(result[0].symbol, "B")
+        self.assertEqual(result[0].start_date, pd.Timestamp("2020-01-01"))
+
+    def test_overlapping_query_window_does_not_consume_local_slot(self):
+        provider = Provider()
+        provider.data = {"A": _bars(n=30)}
+        engine = KlineSimilarityEngine(provider, SimilarityConfig(local_top_n=1, recall_n=1, top_k=1))
+
+        def distances(query, series):
+            result = np.full(len(series) - len(query) + 1, 2.0)
+            result[0] = 0.1
+            result[14] = 0.0  # the query itself is the closest match
+            return result
+
+        with patch("strategies.kline_similarity.engine._mass_distances", side_effect=distances):
+            result = engine.search("A", "2020-01-15", "2020-01-17", history_only=False)
+        self.assertEqual(len(result), 1)
+        self.assertEqual(result[0].start_date, pd.Timestamp("2020-01-01"))
+
+    def test_equal_distances_receive_equal_rank(self):
+        np.testing.assert_array_equal(_rank([0.1, 0.1, 0.5]), [0.0, 0.0, 1.0])
+        np.testing.assert_array_equal(_rank([0.1, 0.1]), [0.0, 0.0])
+
     def test_visual_results_write_top10_style_svg_and_html(self):
         engine = KlineSimilarityEngine(
             Provider(), SimilarityConfig(local_top_n=3, recall_n=20, top_k=3)
```
