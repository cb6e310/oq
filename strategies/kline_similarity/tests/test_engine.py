import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.kline_similarity import KlineSimilarityEngine, SimilarityConfig, aggregate_bars, build_visual_results
from strategies.kline_similarity.engine import _event_features, _rank, _structure_distances
from strategies.kline_similarity.engine import _special_event_match, _prioritize_special_events
from strategies.kline_similarity.engine import _bounded_candidates, _candidate_lengths, _constrained_dtw
from strategies.kline_similarity.engine import _elastic_dtw, _global_nms, SimilarityMatch
from strategies.kline_similarity.engine import (
    _coarse_event_distance, _coarse_shape_distance, _coarse_state_distance,
    _coarse_structure_distance, _elastic_coarse_distances, _state_features,
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
