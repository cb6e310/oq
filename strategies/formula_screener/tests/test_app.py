import unittest

import pandas as pd

from strategies.formula_screener.app import summarize_returns


class FormulaScreenerAppTests(unittest.TestCase):
    def test_return_summary_averages_available_signal_returns(self):
        result = pd.DataFrame({
            "return_5d": [0.1, -0.02, None],
            "return_10d": [0.2, None, None],
        })

        summary = summarize_returns(result, horizons=(5, 10, 20))

        self.assertAlmostEqual(summary[0]["average"], 0.04)
        self.assertEqual(summary[0]["samples"], 2)
        self.assertEqual(summary[1]["average"], 0.2)
        self.assertEqual(summary[1]["samples"], 1)
        self.assertIsNone(summary[2]["average"])
        self.assertEqual(summary[2]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
