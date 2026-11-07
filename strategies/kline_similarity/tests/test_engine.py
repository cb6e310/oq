import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from strategies.kline_similarity import KlineSimilarityEngine, SimilarityConfig, aggregate_bars, build_visual_results
from strategies.kline_similarity.engine import _event_features, _rank, _structure_distances
from strategies.kline_similarity.engine import _special_event_match, _prioritize_special_events
from strategies.kline_similarity.engine import _bounded_candidates, _candidate_lengths, _constrained_dtw
from strategies.kline_similarity.engine import _elastic_dtw, _global_nms, SimilarityMatch
from strategies.kline_similarity import engine as engine_module
from strategies.kline_similarity.engine import ParquetDataProvider
from strategies.kline_similarity.engine import (
    _coarse_event_distance, _coarse_shape_distance, _coarse_state_distance,
    _coarse_structure_distance, _elastic_coarse_distances, _state_features,
    _latest_shape_event_distances, _prepare_elastic_query,
)


def _bars(phase=0.0, n=180):
    date = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 10 + np.arange(n) * 0.03 + np.sin(np.arange(n) / 6 + phase)
    return pd.DataFrame({
        "date": date, "open": close - 0.05, "high": close + 0.15,
        "low": close - 0.15, "close": close, "vol": 100 + np.arange(n),
    })


class Provider:
    def __init__(self):
        self.data = {"A": _bars(), "B": _bars(0.01), "C": _bars(2.0)}

    def get_symbols(self):
        return list(self.data)

    def get_symbol_data(self, symbol):
        return self.data[symbol].copy()


class EngineTests(unittest.TestCase):
    def test_weekly_context_truncates_partial_week_and_limits_prehistory(self):
        frame = _bars(n=200)
        frame["date"] = pd.bdate_range("2020-01-06", periods=len(frame))
        start, end = frame.date.iloc[100], frame.date.iloc[117]  # Monday to Wednesday
        prepare = engine_module._prepare_weekly_context
        parts = engine_module._weekly_context_parts
        before, inside = parts(prepare(frame), start, end, 8)
        self.assertEqual(len(before), 8)
        self.assertEqual(len(inside), 4)
        self.assertEqual(inside[-1, 3], frame.close.iloc[117])
        self.assertEqual(inside[-1, 1], frame.high.iloc[115:118].max())
        changed = frame.copy()
        changed.loc[118:, ["open", "high", "low", "close", "vol"]] *= 20
        changed_parts = parts(prepare(changed), start, end, 8)
        np.testing.assert_array_equal(before, changed_parts[0])
        np.testing.assert_array_equal(inside, changed_parts[1])

    def test_weekly_distance_preserves_scale_and_penalizes_missing_prehistory(self):
        frame = _bars(n=200)
        frame["date"] = pd.bdate_range("2020-01-06", periods=len(frame))
        parts = engine_module._weekly_context_parts(
            engine_module._prepare_weekly_context(frame), frame.date.iloc[100], frame.date.iloc[117], 8)
        scaled = tuple(part * np.array([5, 5, 5, 5, 30]) for part in parts)
        distance, coverage, common = engine_module._weekly_context_distance(parts, scaled, True)
        self.assertAlmostEqual(distance, 0.0, places=12)
        self.assertEqual((coverage, common), (1.0, 8))
        _, missing_coverage, common = engine_module._weekly_context_distance(
            parts, (parts[0][:0], parts[1]), True)
        self.assertLess(missing_coverage, 1.0)
        self.assertEqual(common, 0)

    def test_weekly_layer_changes_daily_ranking_in_both_modes_and_scopes(self):
        for mode in ("fixed", "elastic"):
            for latest in (False, True):
                with self.subTest(mode=mode, latest=latest):
                    dates = pd.bdate_range("2020-01-06", periods=180)
                    matching = _bars(n=180)
                    matching["date"] = dates
                    close = np.r_[np.linspace(6, 10, 160), 10 + 0.1 * np.sin(np.arange(20) / 3)]
                    for col, delta in (("open", -0.02), ("high", 0.05), ("low", -0.05), ("close", 0)):
                        matching[col] = close + delta
                    mismatch = matching.copy()
                    # Identical daily interval, opposite trend in the prior eight weeks.
                    for col in ("open", "high", "low", "close"):
                        mismatch.loc[:159, col] = (20 - close[:160]) + (matching[col].to_numpy()[:160] - close[:160])
                    query = matching.copy()
                    if not latest:
                        query["date"] = dates + pd.Timedelta(weeks=28)
                    data = {"Q": query, "B": matching, "A": mismatch}
                    class WeeklyProvider:
                        def get_symbols(self): return ["B", "A"]
                        def get_symbol_data(self, symbol): return data[symbol].copy()
                    engine = KlineSimilarityEngine(WeeklyProvider(), SimilarityConfig(
                        recall_n=100, local_top_n=20, top_k=5, elastic_coarse_step=1,
                        min_window_length=20, max_window_length=20))
                    results = engine.search("Q", query.date.iloc[160], query.date.iloc[179],
                                            latest_only=latest, match_mode=mode)
                    self.assertTrue(results)
                    self.assertEqual(results[0].symbol, "B")
                    self.assertTrue(all(x.weekly_context_active for x in results))
                    self.assertEqual(engine.last_search_stats["weekly_context_after"], 0)
                    self.assertEqual(engine.last_search_stats["weekly_query_pre_bars"], 8)
                    if latest:
                        by_symbol = {x.symbol: x for x in results}
                        self.assertGreater(by_symbol["B"].weekly_context_score, by_symbol["A"].weekly_context_score)
                        self.assertGreater(by_symbol["B"].score, by_symbol["A"].score)
                        self.assertAlmostEqual(by_symbol["B"].daily_score, by_symbol["A"].daily_score)

    def test_weekly_layer_disabled_or_non_daily_preserves_daily_scores(self):
        for timeframe, weight in (("1d", 0.0), ("1w", 0.25), ("1m", 0.25)):
            with self.subTest(timeframe=timeframe, weight=weight):
                engine = KlineSimilarityEngine(Provider(), SimilarityConfig(weekly_context_weight=weight))
                item = SimilarityMatch("B", pd.Timestamp("2020-04-01"), pd.Timestamp("2020-04-10"),
                                       0.7, 0, 0, 0, 0, 0)
                stats = {}
                with patch.object(engine_module, "_prepare_weekly_context", side_effect=AssertionError("unexpected weekly work")):
                    engine_module._apply_weekly_context(engine, [item], "A", _bars(), timeframe, stats)
                self.assertEqual(item.score, 0.7)
                self.assertEqual(item.daily_score, 0.7)
                self.assertFalse(item.weekly_context_active)

    def test_weekly_layer_uses_each_interval_end_even_with_later_daily_rows(self):
        provider = Provider()
        for symbol in provider.data:
            provider.data[symbol]["date"] = pd.bdate_range("2020-01-06", periods=180)
        query = provider.data["A"].iloc[100:118].copy()
        start, end = query.date.iloc[0], query.date.iloc[-1]
        def apply():
            engine = KlineSimilarityEngine(provider)
            item = SimilarityMatch("B", start, end, 0.7, 0, 0, 0, 0, 0)
            stats = {}
            engine_module._apply_weekly_context(engine, [item], "A", query, "1d", stats)
            return item.to_dict()
        before = apply()
        for symbol in ("A", "B"):
            provider.data[symbol].loc[118:, ["open", "high", "low", "close", "vol"]] *= 30
        after = apply()
        self.assertEqual(before, after)
        self.assertTrue(after["weekly_context_active"])

    def test_weekly_layer_skips_query_with_insufficient_weekly_history(self):
        provider = Provider()
        query = provider.data["A"].iloc[:3].copy()
        item = SimilarityMatch("B", query.date.iloc[0], query.date.iloc[-1], 0.7, 0, 0, 0, 0, 0)
        stats = {}
        engine_module._apply_weekly_context(KlineSimilarityEngine(provider), [item], "A", query, "1d", stats)
        self.assertEqual(item.score, 0.7)
        self.assertFalse(item.weekly_context_active)
        self.assertEqual(stats["weekly_context_active"], 0)

    def test_all_board_types_activate_coverage_and_priority(self):
        common = dict(symbol="600001.SH", start_date=pd.Timestamp("2020-01-01"),
                      end_date=pd.Timestamp("2020-01-14"), mass_distance=0.0,
                      dtw_distance=0.0, aligned_distance=0.0,
                      total_return_diff=0.0, volatility_diff=0.0)
        for channel in range(6):
            with self.subTest(channel=channel):
                query = np.zeros((14, 6)); query[[2, 6, 10], channel] = 1
                partial = query.copy(); partial[6, channel] = 0
                self.assertEqual(_special_event_match(query, query), (1.0, 0))
                coverage, missing = _special_event_match(query, partial)
                self.assertAlmostEqual(coverage, 2 / 3)
                self.assertEqual(missing, 1)
                full = SimilarityMatch(score=0.2, special_event_match=1.0,
                                       missing_special_events=0, **common)
                incomplete = SimilarityMatch(score=0.99, special_event_match=coverage,
                                             missing_special_events=missing, **common)
                _prioritize_special_events([full, incomplete], query)
                self.assertGreater(full.score, incomplete.score)
                self.assertEqual(full.base_score, 0.2)
                self.assertAlmostEqual(full.score, 0.6)
                displaced = SimilarityMatch(score=0.99, special_event_match=1.0,
                                            missing_special_events=0, event_distance=0.2, **common)
                _prioritize_special_events([displaced], query)
                self.assertLess(displaced.score, 0.5)
                self.assertGreaterEqual(displaced.score, 0.25)
                self.assertGreater(displaced.score, incomplete.score)

    def test_unmatched_event_candidates_keep_shape_ordering(self):
        common = dict(symbol="600001.SH", start_date=pd.Timestamp("2020-01-01"),
                      end_date=pd.Timestamp("2020-01-14"), mass_distance=0.0,
                      dtw_distance=0.0, aligned_distance=0.0, total_return_diff=0.0,
                      volatility_diff=0.0, special_event_match=0.0,
                      missing_special_events=1, event_distance=1.0)
        query = np.zeros((14, 6)); query[4, 0] = 1
        weaker = SimilarityMatch(score=0.3, **common)
        stronger = SimilarityMatch(score=0.9, **common)
        _prioritize_special_events([weaker, stronger], query)
        self.assertGreater(stronger.score, weaker.score)
        self.assertLess(stronger.score, 0.25)
        self.assertEqual(stronger.to_dict()["event_required"], stronger.special_event_required)
        self.assertEqual(stronger.to_dict()["event_match"], stronger.special_event_match)
        self.assertEqual(stronger.to_dict()["missing_events"], stronger.missing_special_events)

    def test_event_coverage_windows_match_scalar_for_ordinary_and_special_boards(self):
        rng = np.random.default_rng(79)
        query = rng.integers(0, 2, size=(14, 6)).astype(float)
        series = rng.integers(0, 2, size=(50, 6)).astype(float)
        starts = np.array([0, 1, 8, 19, 36])
        coverage, missing = engine_module._special_event_coverage_windows(query, series, starts, 14)
        for i, start in enumerate(starts):
            expected_coverage, expected_missing = _special_event_match(query, series[start:start + 14])
            self.assertAlmostEqual(coverage[i], expected_coverage)
            self.assertEqual(missing[i], expected_missing)

    def test_event_sequence_distance_counts_order_extras_and_elastic_position(self):
        query = np.zeros((11, 6)); query[[2, 6], 0] = 1; query[8, 1] = 1
        stretched = np.zeros((21, 6)); stretched[[4, 12], 0] = 1; stretched[16, 1] = 1
        self.assertEqual(engine_module._event_sequence_distance(query, stretched), (0.0, 0.0, 0.0))
        reordered = query.copy(); reordered[[2, 6], 0] = 0; reordered[[7, 9], 0] = 1
        distance, count, position = engine_module._event_sequence_distance(query, reordered)
        self.assertEqual(count, 0.0)
        self.assertGreater(position, 0.0)
        self.assertGreater(distance, 0.0)
        self.assertEqual(engine_module._event_sequence_distance(query, np.zeros_like(query)), (1.0, 1.0, 1.0))
        extras = query.copy(); extras[4, 1] = 1
        self.assertGreater(engine_module._event_sequence_distance(query, extras)[0], 0.0)
        self.assertEqual(engine_module._event_sequence_distance(np.zeros((5, 6)), np.zeros((8, 6))),
                         (0.0, 0.0, 0.0))

    def test_no_event_priority_leaves_base_score_unchanged_and_switch_disables_banding(self):
        common = dict(symbol="600001.SH", start_date=pd.Timestamp("2020-01-01"),
                      end_date=pd.Timestamp("2020-01-14"), mass_distance=0.0,
                      dtw_distance=0.0, aligned_distance=0.0, total_return_diff=0.0,
                      volatility_diff=0.0, special_event_match=1.0, missing_special_events=0)
        query = np.zeros((14, 6))
        item = SimilarityMatch(score=0.6, **common)
        _prioritize_special_events([item], query)
        self.assertEqual(item.score, 0.6)
        query[3, 0] = 1
        _prioritize_special_events([item], query, enabled=False)
        self.assertEqual(item.score, 0.6)
        self.assertEqual(item.base_score, 0.6)

    def test_state_salience_uses_board_aware_events_and_first_bar_previous_close(self):
        frame = _bars(n=8)
        frame.loc[:, ["open", "high", "low", "close"]] = 12.0
        events = _event_features(frame, "300001.SZ", previous_close=10.0)
        self.assertEqual(events[0, 0], 1.0)
        _, salience = _state_features(frame, events)
        self.assertEqual(salience[0], 1.0)
        self.assertLess(salience[1], 1.0)
        frame.loc[3:, ["open", "high", "low", "close"]] = 14.4
        events = _event_features(frame, "300001.SZ", previous_close=10.0)
        self.assertEqual(_state_features(frame, events)[1][3], 1.0)

    def test_normal_limit_boards_receive_priority_in_both_modes_and_scopes(self):
        for sign in (1, -1):
            data = {}
            for symbol, event_start, limit_pct in (("600001.SH", 70, 0.1),
                                                   ("600002.SH", 20, 0.1),
                                                   ("300001.SZ", 20, 0.0)):
                frame = _bars(n=100)
                frame.loc[:, ["open", "high", "low", "close"]] = 10.0
                price = 10.0
                event_indexes = {event_start + offset for offset in (2, 6, 10)}
                for idx in range(100):
                    previous = price
                    if idx in event_indexes:
                        price = round(price * (1 + sign * limit_pct), 2)
                    frame.loc[idx, ["open", "high", "low", "close"]] = [previous, max(previous, price),
                                                                                min(previous, price), price]
                data[symbol] = frame
            class BoardProvider:
                def get_symbols(self): return list(data)
                def get_symbol_data(self, symbol): return data[symbol].copy()
            for mode in ("fixed", "elastic"):
                for latest in (False, True):
                    with self.subTest(sign=sign, mode=mode, latest=latest):
                        candidate_data = data
                        if latest:
                            candidate_data = {"600001.SH": data["600001.SH"]}
                            for symbol, pct in (("600002.SH", 0.1), ("300001.SZ", 0.0)):
                                frame = _bars(n=100)
                                frame.loc[:, ["open", "high", "low", "close"]] = 10.0
                                price = 10.0
                                for idx in range(100):
                                    previous = price
                                    if idx in {88, 92, 96}:
                                        price = round(price * (1 + sign * pct), 2)
                                    frame.loc[idx, ["open", "high", "low", "close"]] = [
                                        previous, max(previous, price), min(previous, price), price,
                                    ]
                                candidate_data[symbol] = frame
                        provider = BoardProvider()
                        provider.get_symbol_data = lambda symbol, d=candidate_data: d[symbol].copy()
                        engine = KlineSimilarityEngine(provider, SimilarityConfig(
                            local_top_n=10, recall_n=100, top_k=10, elastic_coarse_step=1,
                            min_window_length=14, max_window_length=14,
                        ))
                        result = engine.search("600001.SH", data["600001.SH"].date.iloc[70],
                                               data["600001.SH"].date.iloc[83], match_mode=mode,
                                               latest_only=latest, exclude_query=True)
                        self.assertTrue(result)
                        self.assertEqual(result[0].symbol, "600002.SH")
                        self.assertEqual(result[0].missing_special_events, 0)
                        self.assertTrue(result[0].special_event_required)
                        self.assertGreaterEqual(result[0].score, 0.5)
                        self.assertTrue(all(x.score < 0.5 for x in result if x.symbol == "300001.SZ"))
                        self.assertEqual(engine.last_search_stats["event_policy"], "all_board_types_v2")
                        self.assertEqual(engine.last_search_stats["event_priority_active"], 1)

    def test_compiled_fixed_dtw_matches_python_fallback(self):
        if engine_module._compiled_fixed_dtw is None:
            self.skipTest("Numba is unavailable")
        rng = np.random.default_rng(57)
        compiled = engine_module._compiled_fixed_dtw
        try:
            engine_module._compiled_fixed_dtw = None
            for length, radius in ((9, 1), (30, 3), (50, 5)):
                a, b = rng.normal(size=(length, 13)), rng.normal(size=(length, 13))
                expected = _constrained_dtw(a, b, radius)
                self.assertAlmostEqual(compiled(a, b, radius), expected, places=12)
        finally:
            engine_module._compiled_fixed_dtw = compiled

    def test_parquet_fast_path_matches_aggregate_and_returns_latest_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = pd.concat([_bars(n=20).assign(ts_code="A"),
                                _bars(0.2, n=20).assign(ts_code="B")], ignore_index=True)
            path = Path(tmp) / "bars.parquet"
            source.to_parquet(path)
            provider = ParquetDataProvider(path)
            self.assertEqual(provider.get_latest_date("1d"), source.date.max())
            self.assertEqual(provider.get_latest_date("1w"),
                             source.date.max().to_period("W-FRI").end_time.normalize())
            pd.testing.assert_frame_equal(
                provider.get_search_bars("A", "1d")[list(aggregate_bars(_bars(n=20), "1d"))].reset_index(drop=True),
                aggregate_bars(_bars(n=20), "1d"), check_dtype=False,
            )

    def test_parquet_fast_path_falls_back_for_invalid_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = _bars(n=20).assign(ts_code="A")
            source.loc[5, "close"] = -1.0
            path = Path(tmp) / "bars.parquet"
            source.to_parquet(path)
            provider = ParquetDataProvider(path)
            self.assertFalse(provider._daily_search_ready)
            self.assertEqual(len(provider.get_search_bars("A", "1d")), 19)
            self.assertEqual(provider.get_latest_date("1d"), source.date.max())

    def test_compiled_elastic_distances_match_python_fallback(self):
        if engine_module._compiled_elastic_dtw is None:
            self.skipTest("Numba is unavailable")
        rng = np.random.default_rng(47)
        compiled_dtw = engine_module._compiled_elastic_dtw
        compiled_state = engine_module._compiled_state_edit_distance
        try:
            engine_module._compiled_elastic_dtw = None
            engine_module._compiled_state_edit_distance = None
            for n, m in ((9, 9), (11, 13), (17, 14)):
                a, b = rng.normal(size=(n, 13)), rng.normal(size=(m, 13))
                sa, sb = rng.random(n), rng.random(m)
                states_a = rng.integers(0, 3, size=(n, 5), dtype=np.int8)
                states_b = rng.integers(0, 3, size=(m, 5), dtype=np.int8)
                expected_dtw = _elastic_dtw(a, b, sa, sb, 0.25, 2, 0.35, 2.5)
                expected_state = engine_module._state_edit_distance(states_a, states_b, sa, sb)
                actual_dtw = compiled_dtw(a, b, sa, sb, 0.25, 2, 0.35, 2.5)
                actual_state = compiled_state(states_a, states_b, sa, sb)
                self.assertAlmostEqual(actual_dtw[0], expected_dtw[0], places=12)
                self.assertEqual(actual_dtw[1:], expected_dtw[1:])
                self.assertAlmostEqual(actual_state, expected_state, places=12)
        finally:
            engine_module._compiled_elastic_dtw = compiled_dtw
            engine_module._compiled_state_edit_distance = compiled_state

    def test_latest_coarse_distances_match_existing_scalar_helpers(self):
        frame = _bars(n=80)
        frame.loc[75, ["open", "high", "low", "close"]] = [11.0, 11.0, 11.0, 11.0]
        query = frame.iloc[25:38].reset_index(drop=True)
        states, _ = _state_features(query)
        query_events = _event_features(query)
        frame_events = _event_features(frame)
        lengths = [9, 13, 17]
        prepared = {length: _prepare_elastic_query(query, states, query_events, length, True)
                    for length in lengths}
        actual = _latest_shape_event_distances(frame, frame_events, lengths, prepared)
        for length in lengths:
            window = frame.iloc[-length:].reset_index(drop=True)
            events = _event_features(window, previous_close=float(frame.close.iloc[-length - 1]))
            self.assertAlmostEqual(actual[length][0], _coarse_shape_distance(query, window, length)[0])
            self.assertAlmostEqual(actual[length][1], _coarse_event_distance(query_events, events, length)[0])

    def test_batched_elastic_recall_matches_scalar_distances(self):
        frame = _bars(n=70)
        query = frame.iloc[20:31].reset_index(drop=True)
        query_states, _ = _state_features(query)
        frame_states, _ = _state_features(frame)
        query_events, frame_events = _event_features(query), _event_features(frame)
        for length in (8, 11, 15):
            expected = (
                _coarse_shape_distance(query, frame, length),
                _coarse_structure_distance(query, frame, length),
                _coarse_state_distance(query_states, frame_states, length),
                _coarse_event_distance(query_events, frame_events, length),
            )
            actual = _elastic_coarse_distances(query, frame, query_states, frame_states,
                                               query_events, frame_events, length, True)
            for fast, slow in zip(actual, expected):
                np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-10)

    def test_numpy_coarse_prefix_counts_match_scalar_across_batch_boundary(self):
        rng = np.random.default_rng(20261024)
        frame = _bars(n=2150)
        query = frame.iloc[30:47].reset_index(drop=True)
        query_states = rng.integers(0, 3, size=(len(query), 5), dtype=np.int8)
        frame_states = rng.integers(0, 3, size=(len(frame), 5), dtype=np.int8)
        query_events = rng.integers(0, 2, size=(len(query), 6)).astype(float)
        frame_events = rng.integers(0, 2, size=(len(frame), 6)).astype(float)
        # Equal counts must choose the lowest state, including empty event buckets.
        frame_states[:2] = np.array([[2] * 5, [0] * 5])
        frame_events[:20] = 0.0
        for length in (13, 23):
            with patch.object(engine_module, "_compiled_coarse_distances", None):
                actual = _elastic_coarse_distances(
                    query, frame, query_states, frame_states, query_events,
                    frame_events, length, False,
                )
            for start in (0, 1, 2047, 2048, len(frame) - length):
                window = frame.iloc[start:start + length].reset_index(drop=True)
                expected = (
                    _coarse_shape_distance(query, window, length)[0],
                    _coarse_structure_distance(query, window, length, False)[0],
                    _coarse_state_distance(query_states, frame_states[start:start + length], length)[0],
                    _coarse_event_distance(query_events, frame_events[start:start + length], length)[0],
                )
                np.testing.assert_allclose([values[start] for values in actual], expected,
                                           rtol=1e-10, atol=1e-10)

    def test_aggregate_week_and_month(self):
        data = _bars(n=70)
        weekly = aggregate_bars(data, "1w")
        monthly = aggregate_bars(data, "1m")
        self.assertGreaterEqual(len(weekly), 9)
        self.assertGreaterEqual(len(monthly), 3)
        self.assertTrue((weekly.high >= weekly.low).all())

    def test_search_returns_ranked_matches_and_fields(self):
        engine = KlineSimilarityEngine(
            Provider(), SimilarityConfig(local_top_n=3, recall_n=20, top_k=5)
        )
        result = engine.search("A", "2020-04-01", "2020-04-30", history_only=False)
        self.assertTrue(result)
        self.assertNotEqual(result[0].symbol, "A")
        self.assertGreaterEqual(result[0].score, result[-1].score)
        self.assertEqual(result[0].window_length, 30)
        self.assertIn("dtw_distance", result[0].to_dict())

    def test_history_only_excludes_windows_after_query_start(self):
        engine = KlineSimilarityEngine(
            Provider(), SimilarityConfig(local_top_n=3, recall_n=20, top_k=5)
        )
        result = engine.search("A", "2020-04-01", "2020-04-30", history_only=True)
        self.assertTrue(all(x.end_date < pd.Timestamp("2020-04-01") for x in result))

    def test_visual_results_write_top10_style_svg_and_html(self):
        engine = KlineSimilarityEngine(
            Provider(), SimilarityConfig(local_top_n=3, recall_n=20, top_k=3)
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = build_visual_results(engine, "A", "2020-04-01", "2020-04-30",
                                          timeframe="1d", output_dir=tmp, top_k=3,
                                          history_only=False)
            self.assertEqual(len(result), 3)
            self.assertTrue(all(Path(x["image_path"]).exists() for x in result))
            self.assertTrue(all(set(x["image_paths"]) == {"daily", "weekly", "monthly"} for x in result))
            self.assertTrue(all(Path(x["image_paths"][p]).exists() for x in result for p in ("daily", "weekly", "monthly")))
            svg = Path(result[0]["image_paths"]["daily"]).read_text(encoding="utf-8")
            self.assertIn('xmlns="http://www.w3.org/2000/svg"', svg)
            self.assertIn("<style>", svg)
            self.assertTrue(all("interval" in x and x["timeframe"] == "1d" for x in result))
            html = Path(result[0]["html_path"]).read_text(encoding="utf-8")
            self.assertIn("前后各90根", html)
            self.assertIn("前后各22根", html)
            self.assertIn("weekly", html)
            self.assertIn("<svg", html)
            self.assertIn("图表中的后续行情仅供复盘展示", html)
            self.assertTrue(all(x["weekly_context_active"] for x in result))
            self.assertIn("周线背景分", svg)

    def test_elastic_visual_results_expose_length_metadata_and_diagnostics(self):
        engine = KlineSimilarityEngine(
            Provider(), SimilarityConfig(length_mode="elastic", local_top_n=2, recall_n=10, top_k=2,
                                         min_window_length=8, max_window_length=40)
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = build_visual_results(
                engine, "A", "2020-04-01", "2020-04-30", timeframe="1d",
                output_dir=tmp, top_k=2, history_only=False, match_mode="elastic",
                return_diagnostics=True,
            )
            self.assertTrue(result)
            self.assertTrue(all(item["match_mode"] == "elastic" for item in result))
            self.assertTrue(all(item["window_length"] > 0 for item in result))
            self.assertTrue(all("search_diagnostics" in item for item in result))
            self.assertIn("length_grid", result[0]["search_diagnostics"])
            html = Path(result[0]["html_path"]).read_text(encoding="utf-8")
            self.assertIn("匹配模式：elastic", html)
            self.assertIn("模式 elastic", html)
            svg = Path(result[0]["image_paths"]["daily"]).read_text(encoding="utf-8")
            self.assertIn("模式 elastic", svg)

    def test_board_events_are_explicit_features(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=4),
            "open": [10, 11, 12, 10.8], "high": [10, 11, 12, 11],
            "low": [10, 11, 12, 10], "close": [10, 11, 12, 10],
            "vol": [1, 1, 1, 1],
        })
        events = _event_features(frame, "600000.SH")
        self.assertEqual(events.shape, (4, 6))
        self.assertEqual(events[1, 0], 1.0)
        self.assertEqual(events[1, 2], 1.0)

        # A normal +9% candle must not be promoted to a 10% limit-up event.
        ordinary = frame.copy()
        ordinary[["open", "high", "low", "close"]] = ordinary[["open", "high", "low", "close"]].astype(float)
        ordinary.loc[1, ["open", "high", "low", "close"]] = [10.2, 10.9, 10.1, 10.9]
        ordinary_events = _event_features(ordinary, "600000.SH")
        self.assertEqual(ordinary_events[1, 0], 0.0)
        ordinary.loc[1, ["open", "high", "low", "close"]] = [10.2, 11.5, 10.1, 11.5]
        self.assertEqual(_event_features(ordinary, "600000.SH")[1, 0], 0.0)

    def test_t_board_and_one_word_up_are_distinct_special_events(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=4),
            "open": [10.0, 11.0, 10.5, 11.55],
            "high": [10.0, 11.0, 10.5, 11.55],
            "low": [10.0, 10.5, 10.5, 11.55],
            "close": [10.0, 11.0, 10.5, 11.55],
            "vol": [100, 120, 110, 130],
        })
        events = _event_features(frame, "600000.SH")
        self.assertEqual(events[1, 0], 1.0)
        self.assertEqual(events[1, 4], 1.0)
        self.assertEqual(events[1, 5], 1.0)
        self.assertEqual(events[1, 2], 0.0)
        self.assertEqual(events[3, 2], 1.0)
        self.assertEqual(events[3, 4], 1.0)
        self.assertEqual(events[3, 5], 0.0)

        query = events[1:2]
        exact_match, exact_missing = _special_event_match(query, events[1:2])
        ordinary_match, ordinary_missing = _special_event_match(query, events[3:4])
        self.assertEqual((exact_match, exact_missing), (1.0, 0))
        self.assertLess(ordinary_match, exact_match)
        self.assertGreater(ordinary_missing, 0)

        common = dict(symbol="600000.SH", start_date=frame.date.iloc[0],
                      end_date=frame.date.iloc[1], mass_distance=0.0,
                      dtw_distance=0.0, aligned_distance=0.0,
                      total_return_diff=0.0, volatility_diff=0.0)
        t_board = SimilarityMatch(score=0.6, special_event_match=exact_match,
                                  missing_special_events=exact_missing, **common)
        ordinary = SimilarityMatch(score=0.9, special_event_match=ordinary_match,
                                   missing_special_events=ordinary_missing, **common)
        _prioritize_special_events([t_board, ordinary], query)
        self.assertGreater(t_board.score, ordinary.score)

    def test_elastic_search_prefers_t_board_candidate(self):
        class BoardProvider:
            def __init__(self):
                self.data = {}
                for symbol in ("600001.SH", "600002.SH", "600003.SH"):
                    frame = _bars(n=75)
                    frame.loc[:, ["open", "high", "low", "close"]] = 10.0
                    self.data[symbol] = frame
                for symbol, index, low in (("600001.SH", 55, 10.5),
                                           ("600002.SH", 20, 10.5),
                                           ("600003.SH", 20, 10.8)):
                    self.data[symbol].loc[index, ["open", "high", "low", "close"]] = [11.0, 11.0, low, 11.0]

            def get_symbols(self):
                return list(self.data)

            def get_symbol_data(self, symbol):
                return self.data[symbol].copy()

        provider = BoardProvider()
        engine = KlineSimilarityEngine(provider, SimilarityConfig(
            recall_n=30, local_top_n=5, top_k=3, elastic_coarse_step=1,
            min_window_length=8, max_window_length=20,
        ))
        query_start = provider.data["600001.SH"].date.iloc[53]
        query_end = provider.data["600001.SH"].date.iloc[60]
        results = engine.search("600001.SH", query_start, query_end,
                                match_mode="elastic", history_only=True)
        self.assertTrue(results)
        self.assertEqual(results[0].symbol, "600002.SH")
        self.assertEqual(results[0].missing_special_events, 0)

    def test_event_queries_are_still_sorted_by_displayed_score(self):
        # A query containing a one-word limit-up bar used to activate an
        # event-distance-first sort, which could make displayed scores rise
        # between adjacent ranks.
        provider = Provider()
        provider.data["A"].loc[60, ["open", "high", "low", "close"]] = 10.0
        provider.data["A"].loc[59, "close"] = 9.0
        provider.data["A"].loc[62, ["open", "high", "low", "close"]] = 11.1
        provider.data["B"].loc[60, ["open", "high", "low", "close"]] = 10.0
        provider.data["B"].loc[59, "close"] = 9.0
        provider.data["B"].loc[62, ["open", "high", "low", "close"]] = 11.1
        engine = KlineSimilarityEngine(
            provider, SimilarityConfig(local_top_n=3, recall_n=20, top_k=5)
        )
        result = engine.search("A", "2020-03-01", "2020-03-10", history_only=False)
        self.assertTrue(result)
        scores = [item.score for item in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_equal_distances_share_rank_and_structure_recall_is_available(self):
        np.testing.assert_array_equal(_rank([0.1, 0.1, 0.5]), [0.0, 0.0, 1.0])
        query = _bars(0.0, 10)
        series = _bars(0.2, 20)
        distances = _structure_distances(query, series)
        self.assertEqual(len(distances), len(series) - len(query) + 1)
        self.assertTrue(np.isfinite(distances).all())

    def test_global_recall_reserves_budget_for_structure_path(self):
        items = [{"mass_distance": float(i) / 100, "event_distance": 0.0,
                  "recall_paths": "mass"} for i in range(20)]
        items += [{"mass_distance": 0.5 + float(i) / 100, "event_distance": 0.0,
                   "recall_paths": "structure"} for i in range(8)]
        selected = _bounded_candidates(items, 9, 0.75)
        self.assertEqual(len(selected), 9)
        self.assertGreaterEqual(sum("structure" in x["recall_paths"] for x in selected), 3)

    def test_global_recall_reserves_matching_special_board_candidates(self):
        ordinary = [{"mass_distance": i / 1000, "event_distance": 0.0,
                     "special_event_match": 0.0, "missing_special_events": 1,
                     "recall_paths": "mass"} for i in range(20)]
        matching = [{"mass_distance": 0.8 + i / 1000, "event_distance": 0.0,
                     "special_event_match": 1.0, "missing_special_events": 0,
                     "recall_paths": "event"} for i in range(4)]
        selected = _bounded_candidates(ordinary + matching, 8, 0.75, special_required=True)
        self.assertEqual(len(selected), 8)
        self.assertGreaterEqual(sum(x["missing_special_events"] == 0 for x in selected), 4)

    def test_search_records_recall_statistics(self):
        engine = KlineSimilarityEngine(
            Provider(), SimilarityConfig(local_top_n=3, recall_n=10, top_k=3)
        )
        result = engine.search("A", "2020-04-01", "2020-04-30", history_only=False)
        self.assertTrue(result)
        stats = engine.last_search_stats
        self.assertGreater(stats["symbols_eligible"], 0)
        self.assertGreater(stats["windows_total"], 0)
        self.assertGreaterEqual(stats["candidates_before_global_limit"], stats["candidates_after_global_limit"])
        self.assertEqual(stats["final_results"], len(result))

    def test_banded_dtw_matches_reference_shape_and_interval_nms(self):
        a = np.arange(30, dtype=float)[:, None]
        b = (np.arange(30, dtype=float) + 0.5)[:, None]
        self.assertAlmostEqual(_constrained_dtw(a, b, 2), 0.5 * np.sqrt(30 / 60), places=6)
        base = dict(score=1.0, mass_distance=0.0, dtw_distance=0.0,
                    aligned_distance=0.0, total_return_diff=0.0,
                    volatility_diff=0.0, event_distance=0.0, window_length=10)
        items = [SimilarityMatch("A", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"), start_idx=0, end_idx=9, **base),
                 SimilarityMatch("A", pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-14"), start_idx=4, end_idx=13, **base)]
        self.assertEqual(len(_global_nms(items, 2, overlap_threshold=0.25)), 1)

    def test_elastic_length_grid_and_dtw_support_one_bar_insert_delete(self):
        config = SimilarityConfig(length_mode="elastic", min_window_length=3, max_window_length=40)
        lengths = _candidate_lengths(10, config)
        self.assertIn(10, lengths)
        self.assertIn(8, lengths)
        self.assertIn(12, lengths)
        base = np.arange(10, dtype=float)[:, None]
        inserted = np.arange(11, dtype=float)[:, None]
        distance, path_length, skipped_query, skipped_candidate = _elastic_dtw(
            base, inserted, np.zeros(10), np.zeros(11), 0.25, 2, 0.1, 2.5
        )
        self.assertTrue(np.isfinite(distance))
        self.assertEqual(path_length, 11)
        self.assertEqual(skipped_query, 0)
        self.assertEqual(skipped_candidate, 1)

    def test_elastic_search_returns_length_and_diagnostic_fields(self):
        engine = KlineSimilarityEngine(
            Provider(), SimilarityConfig(length_mode="elastic", local_top_n=2, recall_n=12, top_k=4,
                                         min_window_length=8, max_window_length=40)
        )
        result = engine.search("A", "2020-04-01", "2020-04-30", history_only=False)
        self.assertTrue(result)
        self.assertTrue(all(item.match_mode == "elastic" for item in result))
        self.assertTrue(all(item.query_length == 30 for item in result))
        self.assertTrue(all(item.window_length >= 8 for item in result))
        self.assertTrue(all(item.length_ratio > 0 for item in result))
        self.assertIn("length_grid", engine.last_search_stats)
        self.assertIn("skipped_candidate_bars_total", engine.last_search_stats)

    def test_elastic_five_bar_query_with_eight_bar_candidate(self):
        engine = KlineSimilarityEngine(
            Provider(), SimilarityConfig(length_mode="elastic", local_top_n=2,
                                         recall_n=8, top_k=2, min_window_length=8,
                                         max_window_length=40)
        )
        result = engine.search("A", "2020-04-01", "2020-04-05",
                               history_only=False, match_mode="elastic")
        self.assertTrue(result)
        self.assertTrue(all(item.query_length == 5 for item in result))
        self.assertTrue(all(item.window_length == 8 for item in result))

    def test_search_can_return_diagnostics_without_breaking_result_shape(self):
        engine = KlineSimilarityEngine(
            Provider(), SimilarityConfig(local_top_n=2, recall_n=8, top_k=2)
        )
        payload = engine.search("A", "2020-04-01", "2020-04-30", history_only=False, return_diagnostics=True)
        self.assertIn("results", payload)
        self.assertIn("diagnostics", payload)
        self.assertEqual(len(payload["results"]), 2)

    def test_latest_only_uses_each_stocks_terminal_window(self):
        provider = Provider()
        provider.data["C"] = provider.data["C"].iloc[:-20].copy()
        engine = KlineSimilarityEngine(provider, SimilarityConfig(local_top_n=3, recall_n=20, top_k=10))
        result = engine.search("A", "2020-04-01", "2020-04-10", latest_only=True,
                               history_only=True, top_k=10)
        self.assertEqual([match.symbol for match in result], ["B"])
        self.assertEqual(result[0].start_date, provider.data["B"].date.iloc[-10])
        self.assertEqual(result[0].end_date, provider.data["B"].date.iloc[-1])
        self.assertEqual(engine.last_search_stats["candidate_scope"], "latest")

    def test_latest_elastic_windows_all_end_at_latest_bar(self):
        engine = KlineSimilarityEngine(Provider(), SimilarityConfig(
            recall_n=20, top_k=10, min_window_length=8, max_window_length=20,
        ))
        result = engine.search("A", "2020-04-01", "2020-04-10",
                               match_mode="elastic", latest_only=True, top_k=10)
        self.assertTrue(result)
        self.assertTrue(all(match.end_date == pd.Timestamp("2020-06-28") for match in result))
        self.assertTrue(all(match.symbol != "A" for match in result))

    def test_one_word_down_receives_special_event_priority(self):
        query = np.zeros((3, 6))
        query[1, 3] = 1.0
        matching = query.copy()
        ordinary = np.zeros_like(query)
        self.assertEqual(_special_event_match(query, matching), (1.0, 0))
        self.assertEqual(_special_event_match(query, ordinary), (0.0, 1))


if __name__ == "__main__":
    unittest.main()
