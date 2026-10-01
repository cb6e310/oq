import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.kline_similarity import KlineSimilarityEngine, SimilarityConfig, aggregate_bars, build_visual_results


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
            self.assertTrue(all("interval" in x and x["timeframe"] == "1d" for x in result))
            html = Path(result[0]["html_path"]).read_text(encoding="utf-8")
            self.assertIn("前后各44根", html)
            self.assertIn("<svg", html)


if __name__ == "__main__":
    unittest.main()
