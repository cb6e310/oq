import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_violent_pullback_charts import limit_down_dates, pick_events


def stock(code, closes, factors=None, start="2020-08-21"):
    return pd.DataFrame({
        "ts_code": code,
        "date": pd.bdate_range(start, periods=len(closes)),
        "close": closes,
        "adj_factor": factors or [1] * len(closes),
    })


class ViolentPullbackSelectionTests(unittest.TestCase):
    def test_limit_down_after_d0_is_sorted_first_without_dropping_events(self):
        stocks = {
            "000001.SZ": stock("000001.SZ", [10, 9, 8.1, 8, 7.2]),
            "000002.SZ": stock("000002.SZ", [10, 9.5, 9, 9, 9]),
        }
        events = pd.DataFrame({
            "ts_code": ["000002.SZ", "000001.SZ", "000001.SZ"],
            "yizi_date": [stocks["000002.SZ"].date.iloc[0], stocks["000001.SZ"].date.iloc[0], stocks["000001.SZ"].date.iloc[1]],
            "signal_date": pd.to_datetime(["2020-08-25", "2020-08-24", "2020-08-26"]),
            "drop_pct": [-.08, -.09, -.07],
        })
        chosen = pick_events(events, stocks)
        self.assertEqual(chosen.ts_code.tolist(), ["000001.SZ", "000002.SZ"])
        self.assertEqual(chosen.has_limit_down.tolist(), [True, False])
        self.assertEqual(chosen["rank"].tolist(), [1, 2])
        self.assertEqual(chosen.limit_down_dates.iloc[0], [stocks["000001.SZ"].date.iloc[1], stocks["000001.SZ"].date.iloc[2]])

    def test_chinext_reform_and_adjusted_reference(self):
        before = stock("300001.SZ", [10, 9, 9], start="2020-08-20")
        after = stock("300001.SZ", [10, 8, 8], start="2020-08-24")
        adjusted = stock("000001.SZ", [10, 4.5, 4.05], factors=[1, 2, 2])
        self.assertEqual(limit_down_dates(before, before.date.iloc[0]), [before.date.iloc[1]])
        self.assertEqual(limit_down_dates(after, after.date.iloc[0]), [after.date.iloc[1]])
        self.assertEqual(limit_down_dates(adjusted, adjusted.date.iloc[0]), [adjusted.date.iloc[1], adjusted.date.iloc[2]])


if __name__ == "__main__":
    unittest.main()
