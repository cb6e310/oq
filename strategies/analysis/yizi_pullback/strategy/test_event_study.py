import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_study import d3_low_bar_close_to_t25_best_close


def bars(closes, lows, codes=None):
    if codes is None:
        codes = ["A"] * len(closes)
    return pd.DataFrame({
        "ts_code": pd.Categorical(codes),
        "date": pd.date_range("2026-01-01", periods=len(closes)),
        "hfq_close": closes,
        "hfq_low": lows,
    })


class PathReturnTests(unittest.TestCase):
    def test_entry_is_close_of_lowest_low_bar_not_low_price(self):
        closes = [10, 9, 8, 7, 6, 5, 12, 11, 10]
        lows = [10, 8, 4, 6, 5, 4, 11, 10, 9]
        result = d3_low_bar_close_to_t25_best_close(bars(closes, lows), [0], [1], horizon=4).iloc[0]
        self.assertEqual(result.return_start_pos, 2)
        self.assertEqual(result.return_start_close, 8)
        self.assertEqual(result.return_end_pos, 6)
        self.assertEqual(result.return_end_close, 12)
        self.assertAlmostEqual(result.ret_close_to_best_close_25, 12 / 8 - 1)

    def test_t_plus_one_prevents_same_day_sale(self):
        result = d3_low_bar_close_to_t25_best_close(
            bars([10, 15, 9, 8, 7, 6, 5], [10, 5, 8, 7, 6, 5, 4]), [0], [1], horizon=3
        ).iloc[0]
        self.assertEqual(result.return_start_pos, 1)
        self.assertEqual(result.return_end_pos, 2)
        self.assertAlmostEqual(result.ret_close_to_best_close_25, 9 / 15 - 1)

    def test_d3_entry_has_full_horizon(self):
        result = d3_low_bar_close_to_t25_best_close(
            bars([10, 9, 8, 7, 8, 9, 11, 12], [10, 8, 7, 6, 7, 8, 10, 11]), [0], [2], horizon=4
        ).iloc[0]
        self.assertEqual(result.return_start_pos, 3)
        self.assertEqual(result.return_horizon_date, pd.Timestamp("2026-01-08"))
        self.assertEqual(result.return_end_pos, 7)

    def test_missing_entry_or_exit_bars_do_not_cross_stock_boundary(self):
        df = bars([10, 9, 8, 7, 10, 9, 8, 7], [10, 9, 8, 7, 10, 9, 8, 7], ["A"] * 4 + ["B"] * 4)
        result = d3_low_bar_close_to_t25_best_close(df, [1, 4], [2, 5], horizon=3)
        self.assertTrue(result.ret_close_to_best_close_25.isna().all())
        self.assertTrue(result.return_horizon_date.isna().all())

    def test_002660_uses_d1_d3_close_not_later_downtrend(self):
        closes = [8, 8.13, 7.72, 7.83, 7.71, 7.60, 7.58, 7.43, 7.21, 7.28]
        lows = [8, 7.88, 7.50, 7.52, 7.66, 7.59, 7.47, 7.43, 7.20, 7.17]
        result = d3_low_bar_close_to_t25_best_close(bars(closes, lows), [0], [2], horizon=5).iloc[0]
        self.assertEqual(result.return_start_pos, 2)
        self.assertEqual(result.return_start_close, 7.72)
        self.assertEqual(result.return_end_pos, 3)
        self.assertAlmostEqual(result.ret_close_to_best_close_25, 7.83 / 7.72 - 1)

    def test_d2_signal_rejects_d1_buy_even_if_d1_has_lowest_low(self):
        df = bars([10, 9, 8, 7, 9, 10, 11, 12], [10, 5, 7, 6, 8, 9, 10, 11])
        result = d3_low_bar_close_to_t25_best_close(df, [0], [2], horizon=3).iloc[0]
        self.assertTrue(np.isnan(result.ret_close_to_best_close_25))


if __name__ == "__main__":
    unittest.main()
