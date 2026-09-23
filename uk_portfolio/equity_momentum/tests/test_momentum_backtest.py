"""Hand-calculated accounting examples for the long-only momentum ledger."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import momentum_backtest as backtest

TICKERS = pd.Index(["A.L", "B.L"], name="ticker")


def series(values):
    return pd.Series(values, index=TICKERS, dtype=float)


def market_day():
    return pd.DataFrame({
        "open_gbp": [10., 20.],
        "close_gbp": [10., 20.],
        "dividend_gbp": [0., 0.],
        "open_reference_usable": [True, True],
        "close_reference_usable": [True, True],
    }, index=TICKERS)


def path_fixture():
    dates = pd.bdate_range("2020-01-02", periods=3, name="Date")
    first, second, third = market_day(), market_day(), market_day()
    first["close_gbp"] = [11., 20.]
    second["open_gbp"] = [12., 18.]
    second["close_gbp"] = [12., 18.]
    second["dividend_gbp"] = [0., 2.]
    third["open_gbp"] = [12., 18.]
    third["close_gbp"] = [12., 18.]
    market = pd.concat([first, second, third], keys=dates, names=["Date", "ticker"])
    targets = pd.DataFrame([[.5, .5], [0., 0.]], index=dates[[0, 2]], columns=TICKERS)
    return market, targets, dates


class RebalanceAccountingTests(unittest.TestCase):
    def test_initial_purchase_funds_its_own_costs(self):
        result = backtest.rebalance_with_costs(series([0, 0]), 1_000., series([.5, .5]), 100.)
        # 1% cost: purchase + .01 * purchase = initial cash.
        invested = 1_000 / 1.01
        np.testing.assert_allclose(result["holdings_gbp"], [invested / 2] * 2)
        self.assertAlmostEqual(result["total_cost_gbp"], invested * .01)
        self.assertAlmostEqual(result["cash_gbp"], 0.)
        self.assertAlmostEqual(result["equity_after_gbp"], invested)

    def test_full_sale_charges_sales_once_and_keeps_existing_cash(self):
        result = backtest.rebalance_with_costs(series([600, 400]), 50., series([0, 0]), 100.)
        np.testing.assert_allclose(result["trades_gbp"], [-600., -400.])
        self.assertAlmostEqual(result["total_cost_gbp"], 10.)
        self.assertAlmostEqual(result["cash_gbp"], 1_040.)
        self.assertTrue(result["holdings_gbp"].eq(0).all())

    def test_switching_stock_charges_both_purchase_and_sale(self):
        result = backtest.rebalance_with_costs(series([1_000, 0]), 0., series([0, 1]), 100.)
        # Sell A for 1000 less 10; B purchase must also fund its 1% cost.
        purchase = 990 / 1.01
        np.testing.assert_allclose(result["trades_gbp"], [-1_000., purchase])
        self.assertAlmostEqual(result["total_cost_gbp"], 10 + .01 * purchase)
        self.assertAlmostEqual(result["cash_gbp"], 0.)

    def test_unchanged_allocation_has_no_economic_trade_or_fee(self):
        result = backtest.rebalance_with_costs(series([600, 400]), 0., series([.6, .4]), 100.)
        np.testing.assert_allclose(result["trades_gbp"], 0., atol=1e-10)
        self.assertAlmostEqual(result["total_cost_gbp"], 0.)
        self.assertAlmostEqual(result["equity_after_gbp"], 1_000.)

    def test_partial_investment_preserves_target_cash_after_costs(self):
        result = backtest.rebalance_with_costs(series([0, 0]), 1_000., series([.6, 0]), 100.)
        equity = 1_000 / 1.006
        self.assertAlmostEqual(result["holdings_gbp"].iloc[0], .6 * equity)
        self.assertAlmostEqual(result["cash_gbp"], .4 * equity)
        self.assertAlmostEqual(result["total_cost_gbp"], .006 * equity)


class SessionAccountingTests(unittest.TestCase):
    def test_units_stay_fixed_while_values_drift_without_a_rebalance(self):
        day = market_day()
        day["close_gbp"] = [12., 18.]
        # A non-trading session needs closing marks, not opening fills.
        day["open_gbp"] = np.nan
        day["open_reference_usable"] = False
        units = series([10, 5])
        original = day.copy(deep=True)
        result = backtest.step_portfolio_day(day, units, 0.)
        pd.testing.assert_series_equal(result["units"], units)
        pd.testing.assert_frame_equal(day, original)
        np.testing.assert_allclose(result["holdings_gbp"], [120., 90.])
        self.assertAlmostEqual(result["equity_gbp"], 210.)
        self.assertEqual(result["total_cost_gbp"], 0.)
        self.assertTrue(result["trades_gbp"].eq(0).all())

    def test_dividend_entitlement_survives_sale_and_cannot_fund_same_open(self):
        day = market_day()
        day["open_gbp"] = [9., 18.]
        day["close_gbp"] = [9., 18.]
        day["dividend_gbp"] = [1., 2.]
        result = backtest.step_portfolio_day(day, series([10, 0]), 0., series([0, 1]), 0.)
        # Ten A units yield GBP10 even after sale. Newly bought B earns nothing.
        self.assertAlmostEqual(result["dividends_gbp"], 10.)
        self.assertAlmostEqual(result["cash_gbp"], 10.)
        self.assertAlmostEqual(result["units"].loc["B.L"], 5.)
        self.assertAlmostEqual(result["equity_gbp"], 100.)

    def test_new_purchase_on_ex_date_has_no_dividend_entitlement(self):
        day = market_day()
        day["dividend_gbp"] = [1., 2.]
        result = backtest.step_portfolio_day(day, series([0, 0]), 1_000., series([1, 0]), 0.)
        self.assertEqual(result["dividends_gbp"], 0.)
        self.assertAlmostEqual(result["equity_gbp"], 1_000.)

    def test_missing_required_quotes_or_dividends_fail_instead_of_becoming_cash(self):
        for column, value in [("close_gbp", np.nan), ("close_reference_usable", False),
                              ("dividend_gbp", np.nan)]:
            with self.subTest(column=column):
                day = market_day()
                day.loc["A.L", column] = value
                with self.assertRaises(ValueError):
                    backtest.step_portfolio_day(day, series([10, 0]), 0.)
        day = market_day()
        day.loc["A.L", "open_reference_usable"] = False
        with self.assertRaisesRegex(ValueError, "Unusable open_gbp"):
            backtest.step_portfolio_day(day, series([10, 0]), 0., series([0, 0]))

    def test_unused_bad_quotes_and_execution_day_volume_do_not_control_fills(self):
        day = market_day()
        day.loc["B.L", ["open_gbp", "close_gbp", "dividend_gbp"]] = np.nan
        day.loc["B.L", ["open_reference_usable", "close_reference_usable"]] = False
        day["reported_volume"] = 0.
        result = backtest.step_portfolio_day(day, series([0, 0]), 1_000., series([1, 0]), 0.)
        self.assertAlmostEqual(result["equity_gbp"], 1_000.)
        changed = day.copy()
        changed["reported_volume"] = 1e9
        other = backtest.step_portfolio_day(changed, series([0, 0]), 1_000., series([1, 0]), 0.)
        pd.testing.assert_series_equal(result["units"], other["units"])


class FullLedgerTests(unittest.TestCase):
    def test_hand_calculated_path_reconciles_returns_costs_dividends_and_weights(self):
        market, targets, dates = path_fixture()
        for bps in [0., 100.]:
            with self.subTest(cost_bps=bps):
                rate = bps / 10_000
                result = backtest.run_momentum_backtest(market, targets, dates, 1_000., bps)
                daily, weights, trades = result["daily"], result["weights"], result["trades"]
                # Day1 units 50 A and 25 B, divided by (1+rate) to fund entry fees.
                expected_equity = np.array([1_050., 1_100., 1_100. - 1_050. * rate]) / (1 + rate)
                np.testing.assert_allclose(daily["equity_gbp"], expected_equity)
                np.testing.assert_allclose(1_000 * (1 + daily["net_return"]).cumprod(), expected_equity)
                np.testing.assert_allclose(daily["dividends_gbp"], [0., 50. / (1 + rate), 0.])
                np.testing.assert_allclose(daily["total_cost_gbp"],
                                           np.array([1_000. * rate, 0., 1_050. * rate]) / (1 + rate))
                np.testing.assert_allclose(weights.sum(axis=1) + daily["cash_weight"], 1.)
                self.assertAlmostEqual(weights.iloc[0]["A.L"], 550 / 1_050)
                self.assertEqual(daily["holdings_count"].tolist(), [2, 2, 0])
                self.assertAlmostEqual(trades["cost_gbp"].sum(), daily["total_cost_gbp"].sum())
                self.assertEqual(len(trades), 4)
                self.assertTrue(result["final_units"].eq(0).all())
                self.assertAlmostEqual(result["final_cash_gbp"], expected_equity[-1])

    def test_future_prices_do_not_change_prior_daily_ledger(self):
        market, targets, dates = path_fixture()
        expected = backtest.run_momentum_backtest(market, targets, dates, 1_000., 10.)
        changed = market.copy()
        changed.loc[dates[-1], ["open_gbp", "close_gbp"]] = 200.
        actual = backtest.run_momentum_backtest(changed, targets, dates, 1_000., 10.)
        pd.testing.assert_frame_equal(actual["daily"].iloc[:2], expected["daily"].iloc[:2])
        pd.testing.assert_frame_equal(actual["weights"].iloc[:2], expected["weights"].iloc[:2])

    def test_cash_only_path_needs_no_stock_marks_or_trades(self):
        market, targets, dates = path_fixture()
        market[["open_gbp", "close_gbp", "dividend_gbp"]] = np.nan
        market[["open_reference_usable", "close_reference_usable"]] = False
        result = backtest.run_momentum_backtest(market, targets.iloc[:0], dates, 1_000., 10.)
        self.assertTrue(result["daily"]["equity_gbp"].eq(1_000.).all())
        self.assertTrue(result["daily"]["net_return"].eq(0.).all())
        self.assertTrue(result["trades"].empty)
        self.assertTrue(result["weights"].eq(0.).all().all())

    def test_skipped_sessions_and_missing_held_marks_are_not_silently_dropped(self):
        market, targets, dates = path_fixture()
        with self.assertRaisesRegex(ValueError, "every market session"):
            backtest.run_momentum_backtest(market, targets, dates[[0, 2]])
        market.loc[(dates[1], "A.L"), "close_gbp"] = np.nan
        with self.assertRaisesRegex(ValueError, "2020-01-03: Unusable close_gbp"):
            backtest.run_momentum_backtest(market, targets, dates)


if __name__ == "__main__":
    unittest.main()
