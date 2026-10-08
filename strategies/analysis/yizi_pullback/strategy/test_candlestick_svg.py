import re
import sys
import unittest
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from candlestick_svg import BOARD_COLORS, aggregate_bars, render_chart, wrap_subtitle


class LimitColorTests(unittest.TestCase):
    def test_detects_main_board_limits_and_one_word_bars(self):
        dates = pd.date_range("2024-01-02", periods=5)
        daily = pd.DataFrame({
            "ts_code": ["000001.SZ"] * 5, "date": dates,
            "open": [10, 10.5, 9.9, 10, 9.9],
            "high": [10.2, 11, 9.9, 10, 9.9],
            "low": [9.8, 10.4, 9.9, 10, 9.9],
            "close": [10, 11, 9.9, 10, 9.9],
            "adj_factor": [1] * 5, "vol": [100] * 5,
        })
        bars = aggregate_bars(daily)
        self.assertEqual(bars.board_type.tolist(), ["", "limit_up", "one_word_down", "", ""])
        svg = render_chart(bars, "日线")
        self.assertIn(f'fill="{BOARD_COLORS["limit_up"]}"', svg)
        self.assertIn(f'fill="{BOARD_COLORS["one_word_down"]}"', svg)

    def test_chinext_reform_and_ex_right_reference(self):
        daily = pd.DataFrame({
            "ts_code": ["300001.SZ"] * 4,
            "date": pd.to_datetime(["2020-08-20", "2020-08-21", "2020-08-24", "2020-08-25"]),
            "open": [10, 11, 4.4, 5], "high": [10, 11, 4.4, 5.28],
            "low": [10, 11, 4.4, 5], "close": [10, 11, 4.4, 5.28],
            "adj_factor": [1, 1, 2, 2], "vol": [100] * 4,
        })
        self.assertEqual(aggregate_bars(daily).board_type.tolist(), ["", "one_word_up", "one_word_down", "limit_up"])

    def test_explicit_labels_survive_and_index_is_not_classified(self):
        dates = pd.date_range("2024-01-02", periods=2)
        daily = pd.DataFrame({"date": dates, "open": [10, 11], "high": [10, 11],
                              "low": [10, 11], "close": [10, 11], "vol": [1, 1],
                              "board_type": ["", "limit_up"]})
        self.assertEqual(aggregate_bars(daily).board_type.tolist(), ["", "limit_up"])
        self.assertNotIn("board_type", aggregate_bars(daily.drop(columns="board_type")))


class TradeMarkerLayoutTests(unittest.TestCase):
    def test_subtitle_wraps_without_truncating_content(self):
        subtitle = " · ".join(["买入日期 2026-01-02 后复权收盘 20.4431"] * 5)
        lines = wrap_subtitle(subtitle, 696)
        self.assertGreater(len(lines), 1)
        self.assertEqual("".join(lines).replace(" · ", ""), subtitle.replace(" · ", ""))

    def test_price_badges_stay_above_chart_and_point_to_prices(self):
        dates = pd.date_range("2026-01-01", periods=4)
        bars = pd.DataFrame({
            "date": dates, "period_key": dates.strftime("%Y-%m-%d"),
            "open": [10, 9, 10, 11], "high": [11, 10, 11, 12],
            "low": [9, 8, 9, 10], "close": [10, 9, 10, 11],
            "vol": [100, 120, 140, 160],
        })
        markers = [
            {"date": dates[1], "price": 9, "side": "buy"},
            {"date": dates[2], "price": 10, "side": "sell"},
        ]
        svg = render_chart(bars, "日线", " · ".join(["D1-D3 买入收盘 9.00"] * 6), price_markers=markers)
        root = ElementTree.fromstring(svg)
        grid_top = min(float(line.attrib["y1"]) for line in root.findall("line") if line.attrib.get("class") == "grid")
        for side in ("buy", "sell"):
            marker = next(group for group in root.findall("g") if group.attrib.get("class") == f"trade-marker {side}-marker")
            self.assertLess(float(marker.find("rect").attrib["y"]) + 18, grid_top)
            self.assertEqual(len(marker.findall("text")), 1)
            self.assertNotIn("<circle", ElementTree.tostring(marker, encoding="unicode"))
        self.assertEqual(len(re.findall(r'stroke-dasharray="3 3"', svg)), 2)


if __name__ == "__main__":
    unittest.main()
