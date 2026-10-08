import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_bear_gain_charts import pick_events


class BearGainSelectionTests(unittest.TestCase):
    def test_strict_threshold_and_event_level_selection(self):
        events = pd.DataFrame({
            "ts_code": ["A", "B", "A", "C", "D"],
            "signal_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]),
            "t25_regime": ["熊市", "熊市", "熊市", "牛市", "跨牛熊"],
            "ret_close_to_best_close_25": [0.06, 0.05, 0.12, 0.4, 0.3],
        })
        chosen = pick_events(events)
        self.assertEqual(chosen.ts_code.tolist(), ["A", "A"])
        self.assertEqual(chosen["rank"].tolist(), [1, 2])
        self.assertEqual(chosen.signal_date.dt.day.tolist(), [3, 1])


if __name__ == "__main__":
    unittest.main()
