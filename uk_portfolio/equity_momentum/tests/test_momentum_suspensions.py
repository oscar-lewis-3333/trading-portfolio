"""Suspension tests to copy into the project's tests folder with the update."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if (PROJECT / "src").is_dir():
    sys.path.insert(0, str(PROJECT / "src"))
import momentum_backtest as bt

NAMES = pd.Index(["A.L", "B.L"], name="ticker")


def vector(values):
    return pd.Series(values, index=NAMES, dtype=float)


def day():
    return pd.DataFrame({
        "open_gbp": [10., 20.], "close_gbp": [10., 20.],
        "dividend_gbp": [0., 0.], "open_reference_usable": True,
        "close_reference_usable": True, "known_suspended": False,
        "close_suspended": False, "unverified_quote": False,
    }, index=NAMES)


def halt(frame):
    frame = frame.copy()
    frame.loc["A.L", ["open_gbp", "close_gbp"]] = np.nan
    frame.loc["A.L", ["open_reference_usable", "close_reference_usable"]] = False
    frame.loc["A.L", ["known_suspended", "close_suspended"]] = True
    return frame


class SuspensionTests(unittest.TestCase):
    def test_one_missing_mark_is_flagged_and_next_price_recognises_loss(self):
        dates = pd.bdate_range("2020-01-02", periods=3, name="Date")
        missing = day()
        missing.loc["A.L", ["open_gbp", "close_gbp"]] = np.nan
        missing.loc["A.L", ["open_reference_usable", "close_reference_usable"]] = False
        resumed = day()
        resumed.loc["A.L", ["open_gbp", "close_gbp"]] = 8.
        market = pd.concat([day(), missing, resumed], keys=dates, names=["Date", "ticker"])
        targets = pd.DataFrame([[1., 0.]], index=dates[:1], columns=NAMES)
        result = bt.run_momentum_backtest(market, targets, dates, 100., 0., max_missing_valuation_sessions=1)
        np.testing.assert_allclose(result["daily"]["equity_gbp"], [100., 100., 80.])
        self.assertEqual(result["daily"]["missing_valuation_holdings_count"].tolist(), [0, 1, 0])
        np.testing.assert_allclose(result["daily"]["stale_weight"], [0., 1., 0.])
        self.assertTrue(pd.isna(market.loc[(dates[1], "A.L"), "close_gbp"]))

    def test_unexplained_gap_cannot_exceed_allowance_or_supply_a_fill(self):
        missing = day()
        missing.loc["A.L", ["open_gbp", "close_gbp"]] = np.nan
        missing.loc["A.L", ["open_reference_usable", "close_reference_usable"]] = False
        first = bt.step_portfolio_day(missing, vector([10, 0]), 0.,
                                     last_valuation_gbp=vector([10, 20]),
                                     max_missing_valuation_sessions=1)
        with self.assertRaisesRegex(ValueError, "allowance exceeded"):
            bt.step_portfolio_day(missing, first["units"], first["cash_gbp"],
                                  last_valuation_gbp=first["valuation_gbp"],
                                  missing_valuation_counts=first["missing_valuation_counts"],
                                  max_missing_valuation_sessions=1)
        with self.assertRaisesRegex(ValueError, "Unusable open_gbp"):
            bt.step_portfolio_day(missing, vector([10, 0]), 0., vector([0, 1]),
                                  last_valuation_gbp=vector([10, 20]),
                                  max_missing_valuation_sessions=1)

    def test_blocked_new_purchase_leaves_its_budget_in_cash(self):
        result = bt.step_portfolio_day(halt(day()), vector([0, 0]), 1_000., vector([.5, .5]), 100.)
        equity = 1_000 / 1.005
        self.assertEqual(result["units"].loc["A.L"], 0.)
        self.assertAlmostEqual(result["holdings_gbp"].loc["B.L"], equity / 2)
        self.assertAlmostEqual(result["cash_gbp"], equity / 2)
        self.assertEqual(result["blocked_target_count"], 1)
        self.assertEqual(result["stale_value_gbp"], 0.)

    def test_trapped_position_cannot_be_sold_to_fund_another_stock(self):
        result = bt.step_portfolio_day(
            halt(day()), vector([40, 5]), 0., vector([0, 1]), 100.,
            last_valuation_gbp=vector([10, 20]),
        )
        np.testing.assert_allclose(result["units"], [40., 5.])
        np.testing.assert_allclose(result["trades_gbp"], 0., atol=1e-10)
        self.assertAlmostEqual(result["stale_value_gbp"], 400.)
        self.assertAlmostEqual(result["cash_gbp"], 0.)
        self.assertAlmostEqual(result["total_cost_gbp"], 0.)

    def test_existing_underweight_position_is_not_topped_up(self):
        result = bt.step_portfolio_day(
            halt(day()), vector([10, 20]), 0., vector([.5, .5]), 0.,
            last_valuation_gbp=vector([10, 20]),
        )
        np.testing.assert_allclose(result["holdings_gbp"], [100., 250.])
        self.assertAlmostEqual(result["cash_gbp"], 150.)
        self.assertEqual(result["trades_gbp"].loc["A.L"], 0.)

    def test_all_frozen_liquidation_request_keeps_positions_and_cash(self):
        result = bt.rebalance_with_costs(
            vector([100, 200]), 50., vector([0, 0]), 100.,
            frozen=pd.Series(True, index=NAMES),
        )
        np.testing.assert_allclose(result["holdings_gbp"], [100., 200.])
        self.assertAlmostEqual(result["cash_gbp"], 50.)
        self.assertEqual(result["total_cost_gbp"], 0.)

    def test_vendor_quotes_during_known_halt_are_never_used(self):
        frame = halt(day())
        frame.loc["A.L", ["open_gbp", "close_gbp"]] = 1_000.
        frame.loc["A.L", ["open_reference_usable", "close_reference_usable"]] = True
        result = bt.step_portfolio_day(frame, vector([10, 0]), 0.,
                                       last_valuation_gbp=vector([10, 20]))
        self.assertEqual(result["equity_gbp"], 100.)
        self.assertEqual(result["stale_value_gbp"], 100.)

    def test_unverified_data_and_unknown_gaps_still_stop(self):
        for known in [False, True]:
            with self.subTest(known=known):
                frame = halt(day())
                if known:
                    frame.loc["A.L", "unverified_quote"] = True
                else:
                    frame.loc["A.L", ["known_suspended", "close_suspended"]] = False
                with self.assertRaisesRegex(ValueError, "Unusable close_gbp"):
                    bt.step_portfolio_day(frame, vector([10, 0]), 0.,
                                           last_valuation_gbp=vector([10, 20]))

    def test_held_suspension_without_a_prior_mark_cannot_be_valued(self):
        with self.assertRaisesRegex(ValueError, "No prior valuation"):
            bt.step_portfolio_day(halt(day()), vector([10, 0]), 0.)

    def test_intraday_halt_uses_observed_open_for_newly_bought_units(self):
        frame = day()
        frame.loc["A.L", "close_gbp"] = np.nan
        frame.loc["A.L", "close_reference_usable"] = False
        frame.loc["A.L", "close_suspended"] = True
        result = bt.step_portfolio_day(frame, vector([0, 0]), 100., vector([1, 0]), 0.)
        self.assertEqual(result["units"].loc["A.L"], 10.)
        self.assertEqual(result["stale_value_gbp"], 100.)

    def test_reopening_loss_is_recognised_and_blocked_sale_is_not_faked(self):
        dates = pd.bdate_range("2020-01-02", periods=4, name="Date")
        resumed = day()
        resumed.loc["A.L", ["open_gbp", "close_gbp"]] = 8.
        frames = [day(), halt(day()), halt(day()), resumed]
        market = pd.concat(frames, keys=dates, names=["Date", "ticker"])
        targets = pd.DataFrame([[1., 0.], [0., 1.]], index=dates[:2], columns=NAMES)
        result = bt.run_momentum_backtest(market, targets, dates, 100., 0.)
        np.testing.assert_allclose(result["daily"]["equity_gbp"], [100., 100., 100., 80.])
        np.testing.assert_allclose(result["daily"]["stale_weight"], [0., 1., 1., 0.])
        self.assertAlmostEqual(result["daily"]["net_return"].iloc[-1], -.2)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["final_units"].loc["A.L"], 10.)
        # A change in the eventual reopening cannot alter earlier stale marks.
        market.loc[(dates[-1], "A.L"), ["open_gbp", "close_gbp"]] = 20.
        other = bt.run_momentum_backtest(market, targets, dates, 100., 0.)
        pd.testing.assert_frame_equal(other["daily"].iloc[:3], result["daily"].iloc[:3])


if __name__ == "__main__":
    unittest.main()
