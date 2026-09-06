import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import build_momentum_equity_curve

#make sure equity curve built here, but used in risk_management works as intented. check on synthetic dataset.
def synthetic_equity_data(asset_returns, lookbacks):
    n_dates = len(next(iter(asset_returns.values())))
    dates = pd.bdate_range("2024-01-01", periods=n_dates, name="date")
    rows = []

    for date_i, date in enumerate(dates):
        entry_date = dates[date_i + 1] if date_i + 1 < n_dates else pd.NaT
        exit_date = dates[date_i + 2] if date_i + 2 < n_dates else pd.NaT

        for ticker, returns in asset_returns.items():
            rows.append(
                {"date": date,
                    "ticker": ticker,
                    "return_63d": lookbacks[ticker],
                    "mom_return_1d": returns[date_i],
                    "mom_entry_date": entry_date,
                    "mom_exit_date": exit_date
                })

    return pd.DataFrame(rows).set_index("date").sort_index()


class EquityCurveTests(unittest.TestCase):
    def test_holdings_drift_between_rebalances(self): #check that holdings are NOT fixed between rebalances each 21 days
        pooled = synthetic_equity_data(asset_returns={
                "A": [0.10, 0.10, 0.0, 0.0, np.nan, np.nan],
                "B": [0.00, 0.00, 0.0, 0.0, np.nan, np.nan],
            },lookbacks={"A": 1.0, "B": 0.0})

        curve = build_momentum_equity_curve(pooled, lookback_col="return_63d",top_frac=1.0, rebalance_days=10, min_universe=2)

        self.assertEqual(len(curve), 5)
        self.assertFalse(curve.index.has_duplicates)
        self.assertAlmostEqual(curve["equity"].iloc[0], 1.0)
        self.assertAlmostEqual(curve["equity"].iloc[1], 1.05)
        self.assertAlmostEqual(curve["equity"].iloc[2], 1.105)
        self.assertAlmostEqual(curve["portfolio_return"].iloc[2], 1.105 / 1.05 - 1.0)
        self.assertAlmostEqual(curve["turnover"].iloc[1], 1.0)
        self.assertEqual(curve["turnover_date"].iloc[1], curve.index[0])
        self.assertGreater(curve["weights"].iloc[2]["A"], 0.5)

    def test_unselected_missing_return_does_not_change_eligibility(self):
        pooled = synthetic_equity_data(asset_returns={
                "A": [0.01, 0.01, 0.01, 0.01, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            },
            lookbacks={"A": 1.0, "B": 0.0})
        curve = build_momentum_equity_curve(pooled, lookback_col="return_63d", top_frac=0.5, rebalance_days=10, min_universe=2)

        self.assertEqual(len(curve), 5)
        self.assertTrue(curve["n_holdings"].eq(1).all())
        self.assertAlmostEqual(curve["equity"].iloc[-1], 1.01**4)

    def test_missing_selected_return_is_rejected(self):
        pooled = synthetic_equity_data(
            asset_returns={
                "A": [0.01, np.nan, 0.01, 0.01, np.nan, np.nan],
                "B": [0.00, 0.00, 0.00, 0.00, np.nan, np.nan],
                }, lookbacks={"A": 1.0, "B": 0.0})

        with self.assertRaisesRegex(ValueError, "Missing or misaligned return"):
            build_momentum_equity_curve(pooled, lookback_col="return_63d", top_frac=0.5, rebalance_days=10, min_universe=2)


if __name__ == "__main__":
    unittest.main()
