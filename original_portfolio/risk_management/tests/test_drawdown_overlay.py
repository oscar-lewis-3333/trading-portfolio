import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from risk_limits import apply_drawdown_overlay

#test dedicated to ensuring our new drawdown test has required property of only changing exposure of the next day

class DrawdownOverlayTests(unittest.TestCase):
    def test_changes_only_next_exposure(self):
        dates = pd.bdate_range("2024-01-01", periods=3, name="date")
        returns = pd.Series([0.0, -0.20, 0.10], index=dates)

        result = apply_drawdown_overlay(returns, soft_limit=-0.10, hard_limit=-0.25)

        loss_date = dates[1]
        following_date = dates[2]

        #must suffer 20% loss with full exposure
        self.assertAlmostEqual(result.loc[loss_date, "exposure"], 1.0)
        self.assertAlmostEqual(result.loc[loss_date, "managed_return"], -0.20)

        #at 20% drawdown, we lay 2/3 of the way towards the hard limit, hence expect 1/3 exposure due to the linear nature of the circuit breaker
        self.assertAlmostEqual(result.loc[loss_date, "signal_exposure"], 1/3)

        #reduced exposure must apply to following day returns
        self.assertAlmostEqual(result.loc[following_date, "exposure"], 1/3)
        self.assertAlmostEqual(result.loc[following_date, "managed_return"], 0.10 / 3)


if __name__ == "__main__":
    unittest.main()