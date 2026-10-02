import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pullback_charts import pick_events


class DropBucketSelectionTests(unittest.TestCase):
    def test_all_violent_events_and_reproducible_medium_sample(self):
        events = pd.DataFrame({
            "ts_code": [f"{index:06d}.SZ" for index in range(24)],
            "signal_date": pd.date_range("2026-01-01", periods=24),
            "drop_pct": [-0.08, -0.071, -0.07] + [-0.05] * 20 + [-0.03],
            "return_start_date": [None, "2026-01-05"] + ["2026-01-05"] * 22,
        })
        chosen, violent_count, medium_total = pick_events(events)
        repeated, _, _ = pick_events(events)
        self.assertEqual((violent_count, medium_total, len(chosen)), (2, 21, 16))
        self.assertEqual(chosen.iloc[:2].ts_code.tolist(), events.iloc[:2].ts_code.tolist())
        self.assertTrue(pd.isna(chosen.iloc[0].return_start_date))
        self.assertEqual(chosen.iloc[2:].ts_code.tolist(), repeated.iloc[2:].ts_code.tolist())
        self.assertEqual(chosen.ts_code.nunique(), 16)


if __name__ == "__main__":
    unittest.main()
