import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADING_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(TRADING_ROOT / "ml_trading_signals" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import build_momentum_equity_curve
from risk_limits import apply_position_stops, position_stop_exit


def synthetic_pooled(prices_a):
    dates = pd.bdate_range("2024-01-01", periods=len(prices_a), name="date")
    prices = {
        "A": np.asarray(prices_a, dtype=float),
        "B": np.full(len(dates), 50.0),
    }

    rows = []
    for date_i, date in enumerate(dates):
        entry_i = min(date_i + 1, len(dates) - 1)
        exit_i = min(date_i + 2, len(dates) - 1)

        for ticker in ("A", "B"):
            entry_open = prices[ticker][entry_i]
            exit_open = prices[ticker][exit_i]
            rows.append({
                "date": date,
                "ticker": ticker,
                "return_63d": 1.0 if ticker == "A" else 0.0,
                "stop_vol": 0.10,
                "mom_entry_date": dates[entry_i],
                "mom_exit_date": dates[exit_i],
                "mom_entry_open": entry_open,
                "mom_high_1d": entry_open * 1.01,
                "mom_low_1d": entry_open * 0.99,
                "mom_exit_open": exit_open,
                "mom_return_1d": exit_open / entry_open - 1,
            })

    return pd.DataFrame(rows).set_index("date").sort_index(), dates


class PositionStopExitTests(unittest.TestCase):
    def test_same_bar_ambiguity_uses_stop(self):
        result = position_stop_exit(
            entry_price=100.0,
            entry_vol=0.10,
            day_open=100.0,
            day_high=125.0,
            day_low=85.0,
            next_open=100.0,
            stop_mult=1.0,
            profit_mult=2.0,
        )

        self.assertEqual(result, (90.0, "stop", "intraday"))

    def test_next_open_gap_uses_open_price(self):
        result = position_stop_exit(
            entry_price=100.0,
            entry_vol=0.10,
            day_open=100.0,
            day_high=105.0,
            day_low=95.0,
            next_open=85.0,
            stop_mult=1.0,
            profit_mult=2.0,
        )

        self.assertEqual(result, (85.0, "stop", "next_open"))


class PositionStopPortfolioTests(unittest.TestCase):
    def test_stopped_cash_waits_for_scheduled_rebalance(self):
        pooled, dates = synthetic_pooled([100, 100, 95, 95, 100, 105])
        first_a = (pooled.index == dates[0]) & pooled["ticker"].eq("A")
        pooled.loc[first_a, "mom_low_1d"] = 89.0

        daily, events = apply_position_stops(
            pooled,
            top_frac=0.5,
            rebalance_days=3,
            profit_mult=100.0,
            stop_mult=1.0,
            min_universe=2,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events.loc[0, "outcome"], "stop")
        self.assertEqual(events.loc[0, "execution"], "intraday")
        self.assertAlmostEqual(events.loc[0, "exit_price"], 90.0)

        self.assertAlmostEqual(daily.loc[dates[2], "equity"], 0.90)
        self.assertAlmostEqual(daily.loc[dates[2], "cash"], 0.90)
        self.assertAlmostEqual(daily.loc[dates[2], "turnover"], 1.90)

        for date in (dates[2], dates[3], dates[4]):
            self.assertEqual(daily.loc[date, "n_holdings"], 0)
            self.assertAlmostEqual(daily.loc[date, "cash_weight"], 1.0)

        self.assertTrue(daily.loc[dates[5], "rebalanced"])
        self.assertEqual(daily.loc[dates[5], "n_holdings"], 1)
        self.assertAlmostEqual(daily.loc[dates[5], "cash"], 0.0)
        self.assertAlmostEqual(daily.loc[dates[5], "rebalance_turnover"], 1.0)
        self.assertAlmostEqual(daily.loc[dates[5], "equity"], 0.945)

    def test_wide_barriers_match_base_equity_curve(self):
        pooled, _ = synthetic_pooled(
            [90, 100, 110, 121, 133.1, 146.41, 161.051]
        )

        base = build_momentum_equity_curve(
            pooled,
            lookback_col="return_63d",
            top_frac=0.5,
            rebalance_days=3,
            min_universe=2,
        )
        stopped, events = apply_position_stops(
            pooled,
            lookback_col="return_63d",
            top_frac=0.5,
            rebalance_days=3,
            profit_mult=1000.0,
            stop_mult=1000.0,
            min_universe=2,
        )

        self.assertTrue(events.empty)
        pd.testing.assert_index_equal(stopped.index, base.index)
        np.testing.assert_allclose(stopped["equity"], base["equity"])
        np.testing.assert_allclose(
            stopped["portfolio_return"], base["portfolio_return"]
        )
        np.testing.assert_allclose(stopped["turnover"], base["turnover"])


if __name__ == "__main__":
    unittest.main()
