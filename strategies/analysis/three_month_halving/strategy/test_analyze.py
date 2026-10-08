import unittest

import numpy as np

from analyze import study_stock


class StudyStockTests(unittest.TestCase):
    def test_first_crossing_and_exact_horizon(self):
        dates = np.arange("2020-01-01", "2020-01-13", dtype="datetime64[D]")
        prices = np.array([100, 100, 90, 80, 60, 49, 45, 48, 51, 55, 60, 65], dtype=float)
        highs = np.array([101, 101, 91, 81, 61, 50, 46, 54, 52, 56, 61, 66], dtype=float)
        events = study_stock(dates, prices, highs, window=4, horizon=3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["signal_date"], dates[5])
        self.assertEqual(events[0]["t25_date"], dates[8])
        self.assertEqual(events[0]["best_date_t25"], dates[7])
        self.assertAlmostEqual(events[0]["best_return_t25"], 54 / 49 - 1)
        self.assertEqual(events[0]["t25_close_adj"], 51)
        self.assertAlmostEqual(events[0]["terminal_return_t25"], 51 / 49 - 1)

    def test_strict_threshold_and_incomplete_future(self):
        dates = np.arange("2020-01-01", "2020-01-08", dtype="datetime64[D]")
        prices = np.array([100, 100, 100, 50, 49, 48, 47], dtype=float)
        events = study_stock(dates, prices, window=3, horizon=3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["signal_date"], dates[4])
        self.assertEqual(events[0]["observation_status"], "no_complete_t25_window")
        self.assertTrue(np.isnan(events[0]["terminal_return_t25"]))

    def test_first_eligible_bar_can_trigger(self):
        dates = np.arange("2020-01-01", "2020-01-07", dtype="datetime64[D]")
        prices = np.array([100, 90, 80, 49, 48, 47], dtype=float)
        events = study_stock(dates, prices, window=3, horizon=2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["signal_date"], dates[3])

    def test_pre_decline_peak_two_year_high_requires_full_prior_history(self):
        dates = np.arange("2020-01-01", "2020-01-12", dtype="datetime64[D]")
        prices = np.array([100, 90, 80, 70, 110, 80, 60, 54, 50, 49, 48], dtype=float)
        events = study_stock(dates, prices, window=4, horizon=1, two_year_bars=4)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["peak_date"], dates[4])
        self.assertTrue(events[0]["peak_has_2y_history"])
        self.assertTrue(events[0]["peak_is_2y_new_high"])
        self.assertEqual(events[0]["peak_prior_2y_max_close_adj"], 100)

        early = np.array([110, 90, 80, 70, 100, 80, 60, 49, 48], dtype=float)
        events = study_stock(dates[:len(early)], early, window=4, horizon=1, two_year_bars=5)
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["peak_has_2y_history"])
        self.assertFalse(events[0]["peak_is_2y_new_high"])


if __name__ == "__main__":
    unittest.main()
