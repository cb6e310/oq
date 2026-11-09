import unittest
from unittest.mock import patch

import pandas as pd

from strategies.formula_screener.engine import add_forward_returns, collect_events, stock_requests


def sample_market():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "ts_code": ["600000.SH"] * 4,
        "open": [10.0, 10.1, 10.2, 10.3],
        "high": [10.2, 10.3, 10.4, 10.5],
        "low": [9.9, 10.0, 10.1, 10.2],
        "close": [10.0, 10.2, 10.3, 10.5],
        "vol": [100.0] * 4,
        "amount": [1000.0] * 4,
    })


class FormulaScreenerEngineTests(unittest.TestCase):
    def test_formula_request_keeps_pre_start_warmup_bars(self):
        request = next(stock_requests(sample_market(), "XG: CLOSE > MA(CLOSE, 2);", "2024-01-04", None))
        self.assertEqual([bar["date"] for bar in request["bars"]], [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"
        ])


    def test_forward_returns_use_future_symbol_bars_from_signal_close(self):
        events = pd.DataFrame([{"ts_code": "600000.SH", "date": "2024-01-03", "value": 1.0}])
        result = add_forward_returns(events, sample_market(), horizons=(1, 2, 3))
        row = result.iloc[0]
        self.assertEqual(row.signal_close, 10.2)
        self.assertAlmostEqual(row.return_1d, 10.3 / 10.2 - 1)
        self.assertAlmostEqual(row.return_2d, 10.5 / 10.2 - 1)
        self.assertTrue(pd.isna(row.return_3d))

    def test_forward_returns_keep_results_for_multiple_signals_per_symbol(self):
        events = pd.DataFrame([
            {"ts_code": "600000.SH", "date": "2024-01-02", "value": 1.0},
            {"ts_code": "600000.SH", "date": "2024-01-03", "value": 2.0},
            {"ts_code": "600000.SH", "date": "2024-01-04", "value": 3.0},
        ])
        result = add_forward_returns(events, sample_market(), horizons=(1,))
        self.assertEqual(result.signal_close.tolist(), [10.0, 10.2, 10.3])
        self.assertAlmostEqual(result.return_1d.iloc[0], 10.2 / 10.0 - 1)
        self.assertAlmostEqual(result.return_1d.iloc[1], 10.3 / 10.2 - 1)
        self.assertAlmostEqual(result.return_1d.iloc[2], 10.5 / 10.3 - 1)

    def test_forward_returns_sort_unsorted_market_bars(self):
        events = pd.DataFrame([{"ts_code": "600000.SH", "date": "2024-01-03", "value": 1.0}])
        result = add_forward_returns(events, sample_market().iloc[::-1], horizons=(1,))
        self.assertEqual(result.signal_close.iloc[0], 10.2)
        self.assertAlmostEqual(result.return_1d.iloc[0], 10.3 / 10.2 - 1)

    def test_screening_reports_progress_until_final_stock(self):
        class Worker:
            def evaluate(self, request):
                return {"symbol": request["symbol"], "signals": []}

            def close(self):
                pass

        market = pd.concat([sample_market().assign(ts_code=f"60000{i}.SH") for i in range(3)])
        updates = []
        with patch("strategies.formula_screener.engine.FormulaWorker", return_value=Worker()):
            collect_events(market, "XG: CLOSE > OPEN;", progress=lambda done, total: updates.append((done, total)))
        self.assertEqual(updates[-1], (3, 3))
        self.assertEqual(len(updates), 3)


if __name__ == "__main__":
    unittest.main()
