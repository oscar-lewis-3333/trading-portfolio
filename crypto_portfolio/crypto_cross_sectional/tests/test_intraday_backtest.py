#checking the intraday ledger works correctly
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("CRYPTO_RESEARCH_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from intraday_backtest import run_intraday_ledger


def fixture():
    days = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    prices = []
    for i, day in enumerate(days):
        prices.extend([
            {"product_id": "BTC-USD", "timestamp": day,"open_gbp": 100.0 if i == 0 else (160.0 if i == 7 else 150.0), "close_gbp": 250.0 if i == 7 else 150.0},
            {"product_id": "ETH-USD", "timestamp": day, "open_gbp": 30.0 if i == 7 else 10.0, "close_gbp": 50.0 if i == 7 else 10.0}])
    decisions = days[[0, 7]]
    targets = pd.DataFrame([[1., 0., 0.], [0., 1., 0.]], index=decisions, columns=["BTC-USD", "ETH-USD", "CASH"])
    execution = decisions + pd.to_timedelta([5, 17], unit="min")
    schedule = pd.DataFrame({"day": decisions, "execution_at": execution, "status": ["ready", "ready"], "required_assets": [2, 2]})
    refs = []
    for i, (day, time) in enumerate(zip(decisions, execution)):
        for product, price in [("BTC-USD", 125.0 if i == 0 else 200.0), ("ETH-USD", 10.0 if i == 0 else 40.0)]:
            refs.append({"product_id": product, "day": day, "execution_at": time,
                         "reference_bar_start": time - pd.Timedelta(minutes=1),
                         "assumed_available_at": time, "reference_price_gbp": price})
    return pd.DataFrame(prices), targets, schedule, pd.DataFrame(refs)


class IntradayLedgerTests(unittest.TestCase):
    def test_entry_and_rotation_use_execution_marks(self):
        p, t, s, r = fixture()
        ledger, trades, positions = run_intraday_ledger(p, t, s, r, fee_bps=0, other_cost_bps=0)
        #buy 8 BTC at 125, mark at 150, later sell 200, buy 40 ETH at 40
        self.assertAlmostEqual(ledger.iloc[0].equity_close_gbp, 1200)
        self.assertAlmostEqual(ledger.iloc[-1].equity_close_gbp, 2000)
        self.assertAlmostEqual(positions.iloc[0]["BTC-USD"], 8)
        self.assertAlmostEqual(positions.iloc[-1]["ETH-USD"], 40)
        self.assertAlmostEqual(ledger.iloc[-1].pre_execution_pnl_gbp, 320)
        self.assertAlmostEqual(ledger.iloc[-1].previous_close_to_open_pnl_gbp, 80)
        self.assertAlmostEqual(ledger.iloc[-1].post_execution_or_nonrebalance_pnl_gbp, 400)
        self.assertEqual(trades.iloc[-1].execution_at, s.iloc[-1].execution_at)
        self.assertLess(ledger.accounting_error_gbp.abs().max(), 1e-9)
        self.assertLess(ledger.daily_pnl_error_gbp.abs().max(), 1e-9)

    def test_costs_on_initial_entry_and_both_rotation_legs(self):
        p, t, s, r = fixture()
        ledger, _, _ = run_intraday_ledger(p, t, s, r)
        c = 12.5 / 10000
        units = 1000 / (125 * (1 + c))
        before_rotation = units * 200
        after_rotation = before_rotation * (1 - c) / (1 + c)
        self.assertAlmostEqual(ledger.iloc[0].equity_close_gbp, units * 150)
        self.assertAlmostEqual(ledger.iloc[-1].equity_close_gbp, after_rotation * 50 / 40)
        self.assertAlmostEqual(ledger.iloc[-1].fee_gbp + ledger.iloc[-1].other_cost_gbp, c * (before_rotation + after_rotation))

    def test_exit_to_cash_preserves_gains_before_sale(self):
        p, t, s, r = fixture()
        t.iloc[-1] = [0., 0., 1.]
        ledger, _, positions = run_intraday_ledger(p, t, s, r, fee_bps=0, other_cost_bps=0)
        self.assertAlmostEqual(ledger.iloc[-1].cash_gbp, 1600)
        self.assertAlmostEqual(ledger.iloc[-1].equity_close_gbp, 1600)
        self.assertEqual(positions.iloc[-1].sum(), 0)

    def test_cash_only_needs_no_asset_references(self):
        p, t, s, r = fixture()
        t.loc[:, :] = [0., 0., 1.]
        s["required_assets"] = 0
        s["status"] = "cash_only"
        ledger, trades, positions = run_intraday_ledger(p, t, s, r.iloc[:0])
        np.testing.assert_allclose(ledger.equity_close_gbp, 1000)
        self.assertTrue(trades.empty)
        self.assertEqual(positions.to_numpy().sum(), 0)

    def test_departing_asset_still_needs_a_price(self):
        p, t, s, r = fixture()
        r = r.loc[~(r.day.eq(t.index[-1]) & r.product_id.eq("BTC-USD"))]
        s.loc[s.day.eq(t.index[-1]), "required_assets"] = 1
        with self.assertRaisesRegex(ValueError, "invalid execution prices"):
            run_intraday_ledger(p, t, s, r)

    def test_future_or_stale_reference_fails(self):
        for shift in [1, -10]:
            p, t, s, r = fixture()
            r.loc[0, "reference_bar_start"] += pd.Timedelta(minutes=shift)
            r.loc[0, "assumed_available_at"] = r.loc[0, "reference_bar_start"] + pd.Timedelta(minutes=1)
            with self.assertRaisesRegex(ValueError, "Future, stale"):
                run_intraday_ledger(p, t, s, r)

    def test_missing_weekly_decision_fails(self):
        p, t, s, r = fixture()
        with self.assertRaisesRegex(ValueError, "Missing Monday"):
            run_intraday_ledger(p, t.iloc[:1], s.iloc[:1], r.loc[r.day.eq(t.index[0])])

    def test_missing_held_close_fails_but_unheld_missing_prices_are_allowed(self):
        p, t, s, r = fixture()
        unheld = p.product_id.eq("ETH-USD") & p.timestamp.lt(t.index[-1])
        p.loc[unheld, ["open_gbp", "close_gbp"]] = np.nan
        run_intraday_ledger(p, t, s, r)
        p.loc[p.product_id.eq("BTC-USD") & p.timestamp.eq(t.index[0]), "close_gbp"] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid closing prices"):
            run_intraday_ledger(p, t, s, r)

    def test_future_prices_do_not_change_earlier_ledger_or_positions(self):
        p, t, s, r = fixture()
        ledger, _, positions = run_intraday_ledger(p, t, s, r)
        p.loc[p.timestamp.eq(t.index[-1]), ["open_gbp", "close_gbp"]] *= 3
        r.loc[r.day.eq(t.index[-1]), "reference_price_gbp"] *= 2
        changed, _, changed_positions = run_intraday_ledger(p, t, s, r)
        pd.testing.assert_frame_equal(ledger.iloc[:-1], changed.iloc[:-1])
        pd.testing.assert_frame_equal(positions.iloc[:-1], changed_positions.iloc[:-1])


if __name__ == "__main__":
    unittest.main()