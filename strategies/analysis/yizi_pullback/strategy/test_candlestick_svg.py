import re
import sys
import unittest
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from candlestick_svg import render_chart, wrap_subtitle


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
