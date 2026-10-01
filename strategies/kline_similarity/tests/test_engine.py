import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.kline_similarity import KlineSimilarityEngine, SimilarityConfig, aggregate_bars, build_visual_results
from strategies.kline_similarity.engine import _event_features
from strategies.kline_similarity.engine import _constrained_dtw, _global_nms, SimilarityMatch


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

    def test_board_events_are_explicit_features(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=4),
            "open": [10, 11, 12, 10.8], "high": [10, 11, 12, 11],
            "low": [10, 11, 12, 10], "close": [10, 11, 12, 10],
            "vol": [1, 1, 1, 1],
        })
        events = _event_features(frame, "600000.SH")
        self.assertEqual(events.shape, (4, 4))
        self.assertEqual(events[1, 0], 1.0)
        self.assertEqual(events[1, 2], 1.0)

        # A normal +9% candle must not be promoted to a 10% limit-up event.
        ordinary = frame.copy()
        ordinary[["open", "high", "low", "close"]] = ordinary[["open", "high", "low", "close"]].astype(float)
        ordinary.loc[1, ["open", "high", "low", "close"]] = [10.2, 10.9, 10.1, 10.9]
        ordinary_events = _event_features(ordinary, "600000.SH")
        self.assertEqual(ordinary_events[1, 0], 0.0)

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


if __name__ == "__main__":
    unittest.main()
