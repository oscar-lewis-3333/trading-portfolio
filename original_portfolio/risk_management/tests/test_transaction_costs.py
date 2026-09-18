import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from risk_limits import apply_transaction_costs, combine_turnover_overlay

#test to ensure transaction costs apply in the correct manner

class TransactionCostTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=3, name="date")
        self.returns = pd.Series([0.10, 0.10, 0.10], index=self.dates)

    def test_no_costs_match_gross_returns(self): #if no costs, then gross should be same as gross returns
        turnover = pd.Series([1.0, 0.0, 0.5], index=self.dates)
        result = apply_transaction_costs(self.returns, turnover, cost_bps=0) #no costs

        np.testing.assert_allclose(result["net_return"], result["gross_return"])
        np.testing.assert_allclose(result["net_equity"], result["gross_equity"])

    def test_cost_applies_only_when_turnover(self): #ensure costs only apply when there's turnover
        turnover = pd.Series([1.0, 0.0, 0.5], index=self.dates)
        result = apply_transaction_costs(self.returns, turnover, cost_bps=100)

        self.assertAlmostEqual(result["cost_fraction"].iloc[0], 0.01) #expected values due to the turnover series
        self.assertAlmostEqual(result["cost_fraction"].iloc[1], 0.0)
        self.assertAlmostEqual(result["cost_fraction"].iloc[2], 0.005)

        self.assertAlmostEqual(result["net_return"].iloc[0], (1 - 0.01) * (1 + 0.10) - 1) #putting above numbers into formula
        self.assertAlmostEqual(result["net_return"].iloc[1], 0.10)
        self.assertAlmostEqual(result["net_return"].iloc[2], (1 - 0.005) * (1 + 0.10) - 1)

    def test_constant_exposure_scales_base_turnover(self): #ensure constant exposure scales correctly
        base_turnover = pd.Series([1.0, 0.4, 0.0], index=self.dates)
        exposure = pd.Series([0.5, 0.5, 0.5], index=self.dates)

        result = combine_turnover_overlay(base_turnover, exposure) 

        np.testing.assert_allclose(result["total_turnover"],[0.5, 0.2, 0.0])

    def test_reduced_exposure_adds_overlay_turnover(self):  #ensure reduced exposures 
        base_turnover = pd.Series([0.0, 0.4, 0.0], index=self.dates)
        exposure = pd.Series([1.0, 0.5, 0.5], index=self.dates)

        result = combine_turnover_overlay(base_turnover, exposure)

        self.assertAlmostEqual(result["strategy_turnover"].iloc[1], 0.2)
        self.assertAlmostEqual(result["overlay_turnover"].iloc[1], 0.5)
        self.assertAlmostEqual(result["total_turnover"].iloc[1], 0.7)

    def test_reentry_from_cash_is_not_double_counted(self): #we ensure that combining the turnover, exposure doesnt double count
        base_turnover = pd.Series([0.0, 0.0, 1.0], index=self.dates)
        exposure = pd.Series([1.0, 0.0, 1.0], index=self.dates)
        result = combine_turnover_overlay(base_turnover, exposure)
        np.testing.assert_allclose(result["total_turnover"], [0.0, 1.0, 1.0])

if __name__ == "__main__":
    unittest.main()