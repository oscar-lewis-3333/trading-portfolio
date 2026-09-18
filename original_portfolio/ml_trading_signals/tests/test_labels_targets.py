import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labels import excess_return_target, simple_labels, triple_barrier_labels

#file dedicated to the testing that the labels and targets acts as they should

def barrier_frame():
    dates = pd.bdate_range("2024-01-01", periods=6)
    return pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "High": [101.0, 111.0, 100.0, 110.0, 101.0, 101.0],
            "Low": [99.0, 99.0, 98.0, 90.0, 99.0, 99.0],
            "Close": [100.0, 110.0, 99.0, 100.0, 100.0, 100.0],
        }, index=dates)


class LabelAndTargetTests(unittest.TestCase):
    def test_simple_labels_enter_next_open(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.DataFrame({
                "Open": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
                "Close": [101.0, 111.0, 121.0, 131.0, 141.0, 151.0],
            }, index=dates)

        labels, returns = simple_labels(prices, horizon=2)

        self.assertAlmostEqual(returns.iloc[0], 121.0 / 110.0 - 1.0)
        self.assertEqual(labels.iloc[0], 1.0)
        self.assertTrue(returns.iloc[-2:].isna().all())
        self.assertTrue(labels.iloc[-2:].isna().all())

    def test_triple_barrier_handles_ambiguous_day_conservatively(self):
        prices = barrier_frame()
        signal_vol = prices["Close"].pct_change().rolling(2).std().iloc[2]
        entry_price = prices["Open"].iloc[3]
        upper = entry_price * (1.0 + 2.0 * signal_vol)
        lower = entry_price * (1.0 - signal_vol)

        prices.iloc[3, prices.columns.get_loc("High")] = upper + 1.0
        prices.iloc[3, prices.columns.get_loc("Low")] = lower - 1.0

        labels, holding_days, returns = triple_barrier_labels(prices, horizon=2, vol_window=2)
        self.assertEqual(labels.iloc[2], 0.0)
        self.assertEqual(holding_days.iloc[2], 1.0)
        self.assertAlmostEqual(returns.iloc[2], lower / entry_price - 1.0)

    def test_triple_barrier_exits_a_later_gap_at_the_open(self):
        prices = barrier_frame()
        signal_vol = prices["Close"].pct_change().rolling(2).std().iloc[2]
        entry_price = prices["Open"].iloc[3]
        lower = entry_price * (1.0 - signal_vol)
        gap_open = lower - 2.0

        prices.iloc[4, prices.columns.get_loc("Open")] = gap_open
        prices.iloc[4, prices.columns.get_loc("High")] = gap_open + 1.0
        prices.iloc[4, prices.columns.get_loc("Low")] = gap_open - 1.0

        labels, holding_days, returns = triple_barrier_labels(prices, horizon=2, vol_window=2)

        self.assertEqual(labels.iloc[2], 0.0)
        self.assertEqual(holding_days.iloc[2], 2.0)
        self.assertAlmostEqual(returns.iloc[2], gap_open / entry_price - 1.0)

    def test_excess_returns_sum_to_zero_within_each_date(self):
        dates = pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3)
        pooled = pd.DataFrame({
                "ticker": ["A", "B", "C"] * 2,
                "ranking_return": [0.1, 0.2, 0.3, -0.1, 0.0, 0.1],
            }, index=pd.Index(dates, name="date"))

        result = excess_return_target(pooled)
        daily_sums = result.groupby(level=0)["excess_return"].sum()

        np.testing.assert_allclose(daily_sums.to_numpy(), 0.0, atol=1e-12)
        np.testing.assert_allclose(result.iloc[:3]["excess_return"].to_numpy(), [-0.1, 0.0, 0.1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
