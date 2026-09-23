"""Offline checks for Bitcoin timing, cash transitions and sweep accounting."""
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CRYPTO_RESEARCH_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

from bitcoin_filter import apply_bitcoin_filter, build_bitcoin_filter
from bitcoin_sweep import BITCOIN_LOOKBACKS, run_bitcoin_training_sweep
from intraday_backtest import run_intraday_ledger
from weekly_sweep import weekly_protocol


def bitcoin_history():
    days = pd.date_range("2023-01-01", "2024-01-15", tz="UTC")
    return pd.DataFrame({"product_id": "BTC-USD", "timestamp": days,
                         "close": 100.0 + np.arange(len(days))})


def execution_references(weeks, assets, prices):
    rows = []
    for day in weeks:
        for asset in assets:
            rows.append({"day": day, "product_id": asset,
                         "execution_at": day + pd.Timedelta(minutes=5),
                         "reference_bar_start": day + pd.Timedelta(minutes=4),
                         "assumed_available_at": day + pd.Timedelta(minutes=5),
                         "reference_price_gbp": prices[(day, asset)]})
    return pd.DataFrame(rows)


def synthetic_sweep_inputs():
    days = pd.date_range("2021-01-01", "2022-02-07", tz="UTC")
    assets = ["BTC-USD"] + [f"COIN{i:02d}-USD" for i in range(19)]
    parts = []
    k = np.arange(len(days))
    for j, asset in enumerate(assets):
        close = 100 * np.exp(.0005 * k + .12 * np.sin(k / 20 + j / 3))
        parts.append(pd.DataFrame({
            "product_id": asset, "timestamp": days, "close": close,
            "signal_available_at": days + pd.Timedelta(days=1),
            "universe_candidate": days >= pd.Timestamp("2021-07-01", tz="UTC"),
            "median_dollar_volume": 2e6 + j * 1000,
            "open_gbp": close * .999, "close_gbp": close,
        }))
    history = pd.concat(parts, ignore_index=True)
    prices = history[["product_id", "timestamp", "open_gbp", "close_gbp"]].copy()
    weeks = pd.date_range("2021-12-20", "2022-02-07", freq="W-MON", tz="UTC")
    schedule = pd.DataFrame({"day": weeks, "status": "ready", "required_assets": 20,
                             "execution_at": weeks + pd.Timedelta(minutes=5)})
    marks = prices.set_index(["timestamp", "product_id"])["open_gbp"].to_dict()
    refs = execution_references(weeks, assets, marks)
    return history, prices, schedule, refs


class OfflineCase(unittest.TestCase):
    def setUp(self):
        blocker = patch("requests.sessions.Session.request",
                        side_effect=AssertionError("Network forbidden in Bitcoin tests"))
        blocker.start()
        self.addCleanup(blocker.stop)


class BitcoinSignalTests(OfflineCase):
    def test_calendar_endpoints_for_every_swept_lookback(self):
        history = bitcoin_history()
        decision = pd.Timestamp("2024-01-01", tz="UTC")
        closes = history.set_index("timestamp")["close"]
        for lookback in BITCOIN_LOOKBACKS:
            with self.subTest(lookback=lookback):
                audit = build_bitcoin_filter(history, [decision], lookback)
                last = closes.loc[decision - pd.Timedelta(days=1)]
                first = closes.loc[decision - pd.Timedelta(days=lookback + 1)]
                self.assertAlmostEqual(audit.iloc[0].btc_return, last / first - 1)
                self.assertEqual(audit.iloc[0].btc_close_usd, last)
                self.assertEqual(audit.iloc[0].btc_reference_close_usd, first)

    def test_future_candles_and_universe_flags_do_not_change_signal(self):
        history = bitcoin_history()
        decision = pd.Timestamp("2024-01-01", tz="UTC")
        changed = history.copy()
        changed.loc[changed.timestamp.ge(decision), "close"] *= 1000
        changed["universe_candidate"] = False
        altcoin = changed.assign(product_id="ALT-USD", close=1.0)
        changed = pd.concat([changed, altcoin], ignore_index=True)
        for lookback in BITCOIN_LOOKBACKS:
            pd.testing.assert_frame_equal(
                build_bitcoin_filter(history, [decision], lookback),
                build_bitcoin_filter(changed, [decision], lookback),
            )

    def test_positive_zero_and_negative_return(self):
        decision = pd.Timestamp("2024-01-01", tz="UTC")
        for last, expected in [(101., True), (100., False), (99., False)]:
            history = bitcoin_history().assign(close=100.)
            history.loc[history.timestamp.eq(decision - pd.Timedelta(days=1)), "close"] = last
            self.assertEqual(bool(build_bitcoin_filter(history, [decision], 30).risk_on.iloc[0]), expected)

    def test_missing_candle_gap_duplicate_and_missing_bitcoin_fail(self):
        history = bitcoin_history()
        decision = pd.Timestamp("2024-01-01", tz="UTC")
        bad = [history.loc[history.timestamp.ne(decision - pd.Timedelta(days=1))],
               history.loc[history.timestamp.ne(decision - pd.Timedelta(days=10))],
               pd.concat([history, history.iloc[[-1]]], ignore_index=True),
               history.assign(product_id="ETH-USD")]
        for frame in bad:
            with self.assertRaises(ValueError):
                build_bitcoin_filter(frame, [decision], 30)

    def test_invalid_decision_calendars_fail(self):
        history = bitcoin_history()
        first = pd.Timestamp("2024-01-01", tz="UTC")
        for dates in [[], [first.tz_localize(None)], [first + pd.Timedelta(days=1)],
                      [first + pd.Timedelta(minutes=1)], [first, first],
                      [first + pd.Timedelta(days=7), first]]:
            with self.assertRaises(ValueError):
                build_bitcoin_filter(history, dates, 30)


class BitcoinTargetTests(OfflineCase):
    def test_overlay_preserves_on_targets_base_cash_and_inputs(self):
        weeks = pd.date_range("2024-01-01", periods=3, freq="W-MON", tz="UTC")
        targets = pd.DataFrame([[.6, .4, 0.], [.6, .4, 0.], [0., 0., 1.]],
                               index=weeks, columns=["A", "B", "CASH"])
        audit = pd.DataFrame({"risk_on": [True, False, True]}, index=weeks)
        before, audit_before = targets.copy(), audit.copy()
        filtered = apply_bitcoin_filter(targets, audit)
        np.testing.assert_allclose(filtered, [[.6, .4, 0.], [0., 0., 1.], [0., 0., 1.]])
        pd.testing.assert_frame_equal(targets, before)
        pd.testing.assert_frame_equal(audit, audit_before)

    def test_bad_alignment_flags_and_weights_fail(self):
        weeks = pd.date_range("2024-01-01", periods=2, freq="W-MON", tz="UTC")
        targets = pd.DataFrame({"BTC-USD": [1., 1.], "CASH": [0., 0.]}, index=weeks)
        valid = pd.DataFrame({"risk_on": [True, False]}, index=weeks)
        for audit in [valid.iloc[::-1], valid.iloc[:1], valid.assign(risk_on=[1, 0]),
                      valid.assign(risk_on=pd.array([True, pd.NA], dtype="boolean"))]:
            with self.assertRaises(ValueError):
                apply_bitcoin_filter(targets, audit)
        for value in [-.1, np.nan, np.inf, .8]:
            invalid = targets.copy()
            invalid.iloc[0, 0] = value
            with self.assertRaises(ValueError):
                apply_bitcoin_filter(invalid, valid)

    def test_entry_exit_and_reentry_match_analytic_cash_equations(self):
        days = pd.date_range("2024-01-01", periods=15, tz="UTC")
        weeks = days[[0, 7, 14]]
        prices = pd.DataFrame({"product_id": "BTC-USD", "timestamp": days,
                               "open_gbp": 150., "close_gbp": 150.})
        prices.loc[prices.timestamp.ge(weeks[1]), "close_gbp"] = 300.
        prices.loc[prices.timestamp.eq(weeks[2]), "close_gbp"] = 60.
        targets = pd.DataFrame({"BTC-USD": 1., "CASH": 0.}, index=weeks)
        audit = pd.DataFrame({"risk_on": [True, False, True]}, index=weeks)
        schedule = pd.DataFrame({"day": weeks, "status": "ready", "required_assets": 1,
                                 "execution_at": weeks + pd.Timedelta(minutes=5)})
        marks = {(d, "BTC-USD"): p for d, p in zip(weeks, [125., 200., 50.])}
        refs = execution_references(weeks, ["BTC-USD"], marks)
        ledger, trades, positions = run_intraday_ledger(
            prices, apply_bitcoin_filter(targets, audit), schedule, refs)
        cost = .00125
        initial_units = 1000 / (125 * (1 + cost))
        sale_cash = initial_units * 200 * (1 - cost)
        final_units = sale_cash / (50 * (1 + cost))
        self.assertAlmostEqual(positions.loc[weeks[0], "BTC-USD"], initial_units)
        self.assertAlmostEqual(ledger.loc[weeks[1], "cash_gbp"], sale_cash)
        self.assertAlmostEqual(positions.loc[weeks[2], "BTC-USD"], final_units)
        self.assertAlmostEqual(ledger.iloc[-1].equity_close_gbp, final_units * 60)
        np.testing.assert_allclose(ledger.loc[days[8:14], "net_return"], 0., atol=1e-14)
        self.assertEqual(positions.loc[weeks[1], "BTC-USD"], 0.)
        self.assertEqual(len(trades), 3)
        self.assertTrue(trades.fee_gbp.gt(0).all())


class BitcoinSweepTests(OfflineCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic_sweep_inputs()
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")), \
                contextlib.redirect_stdout(io.StringIO()):
            cls.summary, cls.runs, cls.audits = run_bitcoin_training_sweep(
                *cls.inputs, weekly_protocol(), training_end="2022-02-01")

    def test_full_grid_scoring_window_and_compounded_returns(self):
        self.assertEqual(len(self.summary), 8)
        self.assertEqual(len(self.runs), 16)
        self.assertEqual(set(self.summary.btc_lookback_days), {0, *BITCOIN_LOOKBACKS})
        self.assertTrue(self.summary.observations.eq(31).all())
        for label in self.summary.index:
            ledger = self.runs[(label, "momentum")][0]
            scored = ledger.loc["2022-01-01":"2022-01-31", "net_return"]
            self.assertAlmostEqual(self.summary.loc[label, "total_return"],
                                   np.expm1(np.log1p(scored).sum()))

    def test_later_data_mutation_cannot_change_any_training_output(self):
        cutoff = pd.Timestamp("2022-02-01", tz="UTC")
        history, prices, schedule, refs = [frame.copy() for frame in self.inputs]
        history.loc[history.timestamp.ge(cutoff), "close"] = -999.
        prices.loc[prices.timestamp.ge(cutoff), ["open_gbp", "close_gbp"]] = -999.
        refs.loc[refs.day.ge(cutoff), "reference_price_gbp"] = -999.
        with contextlib.redirect_stdout(io.StringIO()):
            summary, runs, audits = run_bitcoin_training_sweep(
                history, prices, schedule, refs, weekly_protocol(), training_end="2022-02-01")
        pd.testing.assert_frame_equal(self.summary, summary)
        for key in self.runs:
            for before, after in zip(self.runs[key], runs[key]):
                pd.testing.assert_frame_equal(before, after)
        for key in self.audits:
            pd.testing.assert_frame_equal(self.audits[key], audits[key])

    def test_reconstruct_cash_units_and_wealth_from_signed_trade_logs(self):
        closes = self.inputs[1].pivot(index="timestamp", columns="product_id", values="close_gbp")
        for (label, name), (ledger, trades, positions) in self.runs.items():
            with self.subTest(filter=label, portfolio=name):
                quantities = trades.assign(delta=trades.signed_notional_gbp / trades.reference_price_gbp)
                changes = quantities.pivot_table(index="decision_at", columns="product_id", values="delta", aggfunc="sum")
                units = changes.reindex(index=ledger.index, columns=positions.columns).fillna(0).cumsum()
                np.testing.assert_allclose(units, positions, rtol=1e-11, atol=1e-10)
                outflows = (trades.signed_notional_gbp + trades.fee_gbp + trades.other_cost_gbp)
                cash = 1000 - outflows.groupby(trades.decision_at).sum().reindex(ledger.index, fill_value=0).cumsum()
                np.testing.assert_allclose(cash, ledger.cash_gbp, rtol=1e-11, atol=1e-9)
                wealth = cash + (units * closes.reindex(index=ledger.index, columns=units.columns)).sum(axis=1)
                np.testing.assert_allclose(wealth, ledger.equity_close_gbp, rtol=1e-11, atol=1e-9)
                np.testing.assert_allclose(trades.fee_gbp, trades.signed_notional_gbp.abs() * .0009)
                np.testing.assert_allclose(trades.other_cost_gbp, trades.signed_notional_gbp.abs() * .00035)
                if label != "unfiltered":
                    audit = self.audits[label]
                    self.assertTrue(positions.loc[audit.index[~audit.risk_on]].eq(0).all().all())

    def test_missing_decision_and_truncated_prices_fail(self):
        history, prices, schedule, refs = self.inputs
        with self.assertRaisesRegex(ValueError, "every Monday"):
            run_bitcoin_training_sweep(history, prices, schedule.iloc[1:], refs,
                                       weekly_protocol(), training_end="2022-02-01")
        truncated = prices.loc[prices.timestamp.lt(pd.Timestamp("2022-01-31", tz="UTC"))]
        with self.assertRaisesRegex(ValueError, "cutoff"):
            run_bitcoin_training_sweep(history, truncated, schedule, refs,
                                       weekly_protocol(), training_end="2022-02-01")


if __name__ == "__main__":
    unittest.main()
