import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from risk_limits import apply_volatility_overlay


#ensure newly added volatility overlay acts in the way we expect
class VolatilityOverlayTests(unittest.TestCase):
    def test_shock_changes_only_next_exposure(self):
        dates = pd.bdate_range("2024-01-01", periods=3, name="date")
        returns = pd.Series([0.0, 0.50, 0.10], index=dates)

        result = apply_volatility_overlay(returns, vol_window=2, low_vol_threshold=0.10, high_vol_threshold=0.20, min_exposure=0.20)

        shock_date = dates[1]
        following_date = dates[2]

        self.assertAlmostEqual(result.loc[shock_date, "exposure"], 1.0)
        self.assertAlmostEqual(result.loc[shock_date, "managed_return"], 0.50)
        self.assertAlmostEqual(result.loc[shock_date, "signal_exposure"], 0.20)

        self.assertAlmostEqual(result.loc[following_date, "exposure"], 0.20)
        self.assertAlmostEqual(result.loc[following_date, "managed_return"], 0.02)


if __name__ == "__main__":
    unittest.main()
