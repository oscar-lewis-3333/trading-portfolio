"""Hand-calculated ledger examples with changing prices and delayed execution."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversal_backtest import run_development_ledger


class ReversalLedgerTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"development_start": "2022-01-01", "development_end_exclusive": "2022-01-05",
                         "holdout_start": "2022-01-05"}
        self.dates = pd.date_range("2022-01-01", periods=4, tz="UTC")
        self.targets = pd.DataFrame({"A": [1., 0.], "B": [0., 1.], "CASH": [0., 0.]}, index=self.dates[::2])
        self.prices = pd.concat([
            pd.DataFrame({"timestamp": self.dates, "product_id": "A", "open_gbp": [10., 14., 18., 22.], "close_gbp": [12., 16., 20., 24.]}),
            pd.DataFrame({"timestamp": self.dates, "product_id": "B", "open_gbp": [7., 8., 9., 10.], "close_gbp": [8., 9., 11., 12.]})], ignore_index=True)
        self.schedule = pd.DataFrame({"day": self.dates, "execution_at": self.dates + pd.to_timedelta([5, 5, 8, 5], unit="min"),
                                      "status": "ready", "required_assets": 2})
        rows = []
        for i, row in self.schedule.iterrows():
            for asset, value in (("A", [11., 15., 19., 23.][i]), ("B", [8., 9., 10., 11.][i])):
                rows.append({"product_id": asset, "day": row.day, "execution_at": row.execution_at,
                             "reference_bar_start": row.execution_at - pd.Timedelta(minutes=1),
                             "assumed_available_at": row.execution_at, "reference_price_gbp": value})
        self.refs = pd.DataFrame(rows)

    def run_ledger(self, costs=False):
        return run_development_ledger(self.prices, self.targets, self.schedule, self.refs,
            self.protocol, rebalance_days=2, fee_bps=9. if costs else 0., other_cost_bps=3.5 if costs else 0.)

    def test_manual_zero_cost_path_and_delayed_trade(self):
        ledger, trades, positions = self.run_ledger()
        a_units = 1000 / 11
        b_units = a_units * 19 / 10
        np.testing.assert_allclose(ledger.equity_close_gbp, [a_units * 12, a_units * 16, b_units * 11, b_units * 12])
        np.testing.assert_allclose(positions.A, [a_units, a_units, 0, 0])
        np.testing.assert_allclose(positions.B, [0, 0, b_units, b_units])
        self.assertEqual(ledger.scheduled_rebalance.tolist(), [True, False, True, False])
        self.assertEqual(trades.execution_at.iloc[-1], self.dates[2] + pd.Timedelta(minutes=8))
        self.assertAlmostEqual(ledger.pre_execution_pnl_gbp.iloc[2], a_units * (19 - 18))
        self.assertAlmostEqual(ledger.previous_close_to_open_pnl_gbp.iloc[2], a_units * (18 - 16))
        self.assertAlmostEqual(ledger.net_return.iloc[0], a_units * 12 / 1000 - 1)

    def test_cost_path_and_independent_cash_positions_reconstruction(self):
        ledger, trades, positions = self.run_ledger(costs=True)
        c = .00125
        a_units = 1000 / (11 * (1 + c))
        b_units = (a_units * 19 * (1 - c) / (1 + c)) / 10
        self.assertAlmostEqual(ledger.equity_close_gbp.iloc[-1], b_units * 12)
        cash = 1000.
        held = pd.Series(0., index=["A", "B"])
        for day in self.dates:
            for row in trades.loc[trades.decision_at.eq(day)].itertuples(index=False):
                self.assertAlmostEqual(row.units_before, held[row.product_id])
                held[row.product_id] += row.signed_notional_gbp / row.reference_price_gbp
                cash -= row.signed_notional_gbp + row.fee_gbp + row.other_cost_gbp
                self.assertAlmostEqual(row.fee_gbp + row.other_cost_gbp, abs(row.signed_notional_gbp) * c)
            close = self.prices.loc[self.prices.timestamp.eq(day)].set_index("product_id").close_gbp
            self.assertAlmostEqual(cash + (held * close).sum(), ledger.loc[day, "equity_close_gbp"])
            np.testing.assert_allclose(held, positions.loc[day], atol=1e-12)
        self.assertTrue(ledger.loc[~ledger.scheduled_rebalance, "traded_notional_gbp"].eq(0).all())
        self.assertLess(ledger.daily_pnl_error_gbp.abs().max(), 1e-10)

    def test_empty_universe_cash_target_sells_and_keeps_cash(self):
        self.targets.loc[self.dates[2]] = [0., 0., 1.]
        ledger, trades, positions = self.run_ledger(costs=True)
        expected = 1000 / (11 * 1.00125) * 19 * .99875
        np.testing.assert_allclose(ledger.equity_close_gbp.iloc[2:], expected)
        self.assertTrue(positions.iloc[2:].eq(0).all().all())
        self.assertTrue(trades.loc[trades.decision_at.eq(self.dates[2]), "signed_notional_gbp"].lt(0).all())

    def test_all_cash_period_still_has_all_days(self):
        self.targets.loc[:, ["A", "B"]] = 0.
        self.targets["CASH"] = 1.
        ledger, trades, positions = self.run_ledger(costs=True)
        self.assertEqual(len(ledger), 4)
        self.assertTrue(ledger.equity_close_gbp.eq(1000.).all())
        self.assertTrue(trades.empty)

    def test_missing_scheduled_target_fails(self):
        self.targets = self.targets.iloc[:1]
        with self.assertRaisesRegex(ValueError, "full anchored"):
            self.run_ledger()

    def test_holdout_candle_is_rejected(self):
        extra = self.prices.iloc[[0]].copy()
        extra["timestamp"] = pd.Timestamp("2022-01-05", tz="UTC")
        self.prices = pd.concat([self.prices, extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "restricted to development"):
            self.run_ledger()

    def test_future_and_stale_references_are_rejected(self):
        self.refs.loc[0, "reference_bar_start"] = self.refs.loc[0, "execution_at"]
        self.refs.loc[0, "assumed_available_at"] = self.refs.loc[0, "execution_at"] + pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "Future, stale"):
            self.run_ledger()
        self.setUp()
        self.refs.loc[0, "reference_bar_start"] = self.dates[0] - pd.Timedelta(minutes=1)
        self.refs.loc[0, "assumed_available_at"] = self.dates[0]
        with self.assertRaisesRegex(ValueError, "Future, stale"):
            self.run_ledger()

    def test_missing_held_execution_and_closing_prices_fail(self):
        self.refs = self.refs.loc[~(self.refs.day.eq(self.dates[2]) & self.refs.product_id.eq("A"))]
        self.schedule.loc[2, "required_assets"] = 1
        with self.assertRaisesRegex(ValueError, "invalid execution prices"):
            self.run_ledger()
        self.setUp()
        self.prices = self.prices.loc[~(self.prices.timestamp.eq(self.dates[1]) & self.prices.product_id.eq("A"))]
        with self.assertRaisesRegex(ValueError, "invalid closing prices|invalid opening prices"):
            self.run_ledger()

    def test_future_daily_prices_cannot_change_earlier_ledger(self):
        first = self.run_ledger()[0]
        self.prices.loc[self.prices.timestamp.eq(self.dates[-1]), ["open_gbp", "close_gbp"]] *= 10
        second = self.run_ledger()[0]
        pd.testing.assert_frame_equal(first.iloc[:-1], second.iloc[:-1])


if __name__ == "__main__":
    unittest.main()
