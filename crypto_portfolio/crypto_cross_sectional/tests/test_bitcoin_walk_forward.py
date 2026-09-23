"""Offline checks for annual Bitcoin selection, separate from ledger tests."""
import contextlib
import copy
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

import bitcoin_walk_forward as walk_forward
from bitcoin_sweep import BITCOIN_LOOKBACKS
from weekly_sweep import weekly_protocol


def training_table(winner=120, observations=730):
    days = [0, *BITCOIN_LOOKBACKS]
    table = pd.DataFrame({
        "btc_lookback_days": days,
        "sharpe_zero_cash_rate": [100.] + [.1] * len(BITCOIN_LOOKBACKS),
        "annualised_one_way_turnover": 10.,
        "observations": observations,
        "total_return": .25,
        "max_drawdown": -.20,
        "sharpe_difference": 0.,
    }, index=pd.Index(["unfiltered"] + [f"btc_{d}d" for d in BITCOIN_LOOKBACKS], name="filter"))
    table.loc[f"btc_{winner}d", "sharpe_zero_cash_rate"] = 1.5
    return table


class OfflineCase(unittest.TestCase):
    def setUp(self):
        blocker = patch("requests.sessions.Session.request",
                        side_effect=AssertionError("Network forbidden in selection tests"))
        blocker.start()
        self.addCleanup(blocker.stop)


class BitcoinRankingTests(OfflineCase):
    def test_strategy_sharpe_wins_and_unfiltered_cannot_be_selected(self):
        table = training_table()
        table.loc["btc_90d", "sharpe_difference"] = 1000.
        ranked = walk_forward.rank_bitcoin_filters(table)
        self.assertEqual(ranked.index[0], "btc_120d")
        self.assertNotIn("unfiltered", ranked.index)
        self.assertEqual(set(ranked.btc_lookback_days), set(BITCOIN_LOOKBACKS))

    def test_rounding_then_turnover_then_shorter_lookback_break_ties(self):
        table = training_table()
        table.loc["btc_120d", "sharpe_zero_cash_rate"] = 1.5 + 1e-13
        table.loc[["btc_30d", "btc_60d", "btc_90d"], "sharpe_zero_cash_rate"] = 1.5
        table.loc[["btc_60d", "btc_90d"], "annualised_one_way_turnover"] = 9.
        ranked = walk_forward.rank_bitcoin_filters(table)
        self.assertEqual(ranked.index[:4].tolist(),
                         ["btc_60d", "btc_90d", "btc_30d", "btc_120d"])

    def test_rank_does_not_mutate_input_or_depend_on_row_order(self):
        table = training_table()
        before = table.copy(deep=True)
        expected = walk_forward.rank_bitcoin_filters(table)
        actual = walk_forward.rank_bitcoin_filters(table.sample(frac=1, random_state=12))
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(table, before)

    def test_nonfinite_scores_are_excluded_and_no_valid_filter_fails(self):
        table = training_table()
        table.loc["btc_120d", "sharpe_zero_cash_rate"] = np.nan
        table.loc["btc_90d", "sharpe_zero_cash_rate"] = np.inf
        table.loc["btc_60d", "annualised_one_way_turnover"] = np.inf
        ranked = walk_forward.rank_bitcoin_filters(table)
        self.assertTrue({"btc_120d", "btc_90d", "btc_60d"}.isdisjoint(ranked.index))
        table.loc[table.btc_lookback_days.ne(0), "sharpe_zero_cash_rate"] = np.nan
        with self.assertRaisesRegex(ValueError, "No Bitcoin filter"):
            walk_forward.rank_bitcoin_filters(table)

    def test_negative_scores_still_select_best_actual_filter(self):
        table = training_table()
        table.loc[table.btc_lookback_days.ne(0), "sharpe_zero_cash_rate"] = -2.
        table.loc["btc_180d", "sharpe_zero_cash_rate"] = -.5
        self.assertEqual(walk_forward.rank_bitcoin_filters(table).index[0], "btc_180d")

    def test_missing_columns_incomplete_grid_and_duplicates_fail(self):
        table = training_table()
        duplicate_label = table.copy()
        duplicate_label.index = ["same", "same", *table.index[2:]]
        duplicate_lookback = table.copy()
        duplicate_lookback.loc["btc_180d", "btc_lookback_days"] = 120
        outside_grid = table.copy()
        outside_grid.loc["btc_180d", "btc_lookback_days"] = 181
        for bad in [table.drop(columns="sharpe_zero_cash_rate"),
                    table.drop(index="btc_180d"), table.drop(index="unfiltered"),
                    duplicate_label, duplicate_lookback, outside_grid]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                walk_forward.rank_bitcoin_filters(bad)


class AnnualBitcoinSelectionTests(OfflineCase):
    def run_selector(self, tables, protocol=None):
        inputs = tuple(object() for _ in range(4))
        protocol = weekly_protocol() if protocol is None else protocol
        with patch.object(walk_forward, "run_bitcoin_training_sweep",
                          side_effect=[(t, {}, {}) for t in tables]) as sweep, \
                contextlib.redirect_stdout(io.StringIO()):
            result = walk_forward.select_annual_bitcoin_filters(*inputs, protocol)
        return result, sweep, inputs, protocol

    def test_each_fold_uses_its_own_cutoff_and_winner(self):
        tables = [training_table(winner, n) for winner, n in
                  [(120, 730), (90, 1096), (180, 1461)]]
        copies = [t.copy(deep=True) for t in tables]
        (choices, saved), sweep, inputs, protocol = self.run_selector(tables)
        self.assertEqual(choices.btc_lookback_days.tolist(), [120, 90, 180])
        self.assertEqual(choices.index.tolist(), [1, 2, 3])
        self.assertEqual(sweep.call_count, 3)
        for call, fold, table, original in zip(sweep.call_args_list, protocol["folds"], tables, copies):
            self.assertEqual(call.args[:4], inputs)
            self.assertIs(call.args[4], protocol)
            self.assertEqual(call.kwargs, {"training_end": fold["training_end"]})
            for column in ("training_start", "training_end", "test_start", "test_end"):
                self.assertEqual(choices.loc[fold["fold"], column],
                                 pd.Timestamp(fold[column], tz="UTC"))
            pd.testing.assert_frame_equal(saved[fold["fold"]], original)
            pd.testing.assert_frame_equal(table, original)
        np.testing.assert_allclose(choices.training_sharpe, 1.5)
        np.testing.assert_allclose(choices.training_return, .25)
        np.testing.assert_allclose(choices.training_max_drawdown, -.20)
        np.testing.assert_allclose(choices.training_turnover, 10.)

    def test_later_training_results_cannot_rewrite_earlier_choice(self):
        first_tables = [training_table(120, 730), training_table(90, 1096), training_table(180, 1461)]
        second_tables = [first_tables[0].copy(), training_table(7, 1096), training_table(14, 1461)]
        first = self.run_selector(first_tables)[0][0]
        second = self.run_selector(second_tables)[0][0]
        pd.testing.assert_series_equal(first.loc[1], second.loc[1])
        self.assertEqual(second.btc_lookback_days.tolist(), [120, 7, 14])

    def test_wrong_observation_count_in_any_candidate_fails(self):
        table = training_table()
        table.loc["btc_7d", "observations"] = 729
        with self.assertRaisesRegex(ValueError, "observation counts"):
            self.run_selector([table])

    def test_incomplete_grid_fails_before_selecting(self):
        with self.assertRaisesRegex(ValueError, "complete sweep grid"):
            self.run_selector([training_table().drop(index="btc_7d")])

    def test_changed_protocol_is_rejected_before_training(self):
        protocol = copy.deepcopy(weekly_protocol())
        protocol["folds"][0]["training_end"] = "2024-01-08"
        with patch.object(walk_forward, "run_bitcoin_training_sweep") as sweep:
            with self.assertRaises(ValueError):
                walk_forward.select_annual_bitcoin_filters(None, None, None, None, protocol)
            sweep.assert_not_called()


def bitcoin_evaluation_fixture():
    days = pd.date_range("2023-01-01", "2026-08-31", tz="UTC")
    clock = np.arange(len(days))
    pieces = []
    for i in range(20):
        asset = "BTC-USD" if i == 0 else f"ASSET{i:02d}-USD"
        close = 100 * np.exp(.0002 * clock + .15 * np.sin(clock / (13 + i) + i))
        pieces.append(pd.DataFrame({
            "product_id": asset, "timestamp": days, "close": close,
            "signal_available_at": days + pd.Timedelta(days=1),
            "universe_candidate": clock >= 180, "median_dollar_volume": 2e6 + i,
            "open_gbp": close * .999, "close_gbp": close,
        }))
    panel = pd.concat(pieces, ignore_index=True)
    prices = panel[["product_id", "timestamp", "open_gbp", "close_gbp"]].copy()
    weeks = pd.date_range("2024-01-01", "2026-08-31", freq="W-MON", tz="UTC")
    schedule = pd.DataFrame({"day": weeks, "execution_at": weeks + pd.Timedelta(minutes=5),
                             "status": "ready", "required_assets": 20})
    refs = prices.loc[prices.timestamp.isin(weeks), ["product_id", "timestamp", "open_gbp"]].rename(
        columns={"timestamp": "day", "open_gbp": "reference_price_gbp"})
    refs["execution_at"] = refs.day + pd.Timedelta(minutes=5)
    refs["reference_bar_start"] = refs.day + pd.Timedelta(minutes=4)
    refs["assumed_available_at"] = refs.execution_at
    choices = pd.DataFrame([
        {**fold, "btc_lookback_days": lookback}
        for fold, lookback in zip(weekly_protocol()["folds"], [120, 7, 180])
    ]).set_index("fold")
    for column in ("training_start", "training_end", "test_start", "test_end"):
        choices[column] = pd.to_datetime(choices[column], utc=True)
    return panel, prices, schedule, refs, choices


class BitcoinEvaluationTests(OfflineCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.prices, cls.schedule, cls.refs, cls.choices = bitcoin_evaluation_fixture()
        inputs = (cls.panel, cls.prices, cls.schedule, cls.refs, cls.choices)
        before = [frame.copy(deep=True) for frame in inputs]
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")), \
                patch.object(walk_forward, "run_intraday_ledger", wraps=walk_forward.run_intraday_ledger) as ledger:
            cls.runs, cls.audit, cls.targets = walk_forward.run_bitcoin_walk_forward(
                *inputs, weekly_protocol())
            cls.ledger_calls = ledger.call_args_list
        for frame, original in zip(inputs, before):
            pd.testing.assert_frame_equal(frame, original)

    def test_annual_gate_changes_on_mondays_using_completed_calendar_days(self):
        firsts = self.audit.groupby("selection_fold").first()
        self.assertEqual(firsts.decision_at.tolist(), list(pd.to_datetime(
            ["2024-01-01", "2025-01-06", "2026-01-05"], utc=True)))
        self.assertEqual(firsts.btc_lookback_days.tolist(), [120, 7, 180])
        self.assertTrue(self.audit.coin_lookback_days.eq(90).all())
        self.assertTrue(self.audit.top_fraction.eq(.1).all())
        btc = self.panel.loc[self.panel.product_id.eq("BTC-USD")].set_index("timestamp").close
        for row in self.audit.itertuples():
            last = btc.loc[row.decision_at - pd.Timedelta(days=1)]
            first = btc.loc[row.decision_at - pd.Timedelta(days=int(row.btc_lookback_days) + 1)]
            self.assertAlmostEqual(row.btc_return, last / first - 1)
            self.assertEqual(row.risk_on, last > first)
        at = self.audit.set_index("decision_at")
        self.assertEqual(at.loc["2024-12-30", "btc_lookback_days"], 120)
        self.assertEqual(at.loc["2025-12-29", "btc_lookback_days"], 7)

    def test_identical_gate_for_strategy_and_benchmark_preserves_on_targets(self):
        gate = self.audit.set_index("decision_at").risk_on
        self.assertTrue(gate.any() and (~gate).any())
        for name, count in [("momentum", 2), ("equal_weight", 20)]:
            filtered, base = self.targets[f"filtered_{name}"], self.targets[f"unfiltered_{name}"]
            pd.testing.assert_frame_equal(filtered.loc[gate], base.loc[gate])
            self.assertTrue(filtered.loc[~gate, "CASH"].eq(1).all())
            self.assertTrue(filtered.loc[~gate].drop(columns="CASH").eq(0).all().all())
            self.assertTrue(base.drop(columns="CASH").gt(0).sum(axis=1).eq(count).all())

    def test_later_prices_and_choices_cannot_change_earlier_targets(self):
        panel, choices = self.panel.copy(), self.choices.copy()
        cutoff = pd.Timestamp("2025-01-01", tz="UTC")
        panel.loc[panel.timestamp.ge(cutoff), "close"] *= 100
        choices.loc[[2, 3], "btc_lookback_days"] = [30, 60]
        changed, audit = walk_forward.build_bitcoin_walk_forward_targets(
            panel, choices, self.schedule, weekly_protocol())
        for name, original in self.targets.items():
            pd.testing.assert_frame_equal(original.loc[original.index < cutoff],
                                          changed[name].loc[changed[name].index < cutoff])
        pd.testing.assert_frame_equal(self.audit.loc[self.audit.decision_at < cutoff],
                                      audit.loc[audit.decision_at < cutoff])

    def test_invalid_choices_and_omitted_monday_fail(self):
        cases = [self.choices.drop(columns="btc_lookback_days"), self.choices.drop(index=3)]
        for bad_day in [0, 13]:
            bad = self.choices.copy()
            bad.loc[1, "btc_lookback_days"] = bad_day
            cases.append(bad)
        future = self.choices.copy()
        future.loc[1, "training_end"] = pd.Timestamp("2024-02-01", tz="UTC")
        cases.append(future)
        for choices in cases:
            with self.assertRaises(ValueError):
                walk_forward.build_bitcoin_walk_forward_targets(
                    self.panel, choices, self.schedule, weekly_protocol())
        with self.assertRaisesRegex(ValueError, "every Monday"):
            walk_forward.build_bitcoin_walk_forward_targets(
                self.panel, self.choices, self.schedule.iloc[1:], weekly_protocol())

    def test_four_continuous_ledgers_carry_cash_and_units_across_year_boundaries(self):
        self.assertEqual(len(self.ledger_calls), 4)
        expected_days = pd.date_range("2024-01-01", "2026-08-31", tz="UTC")
        for call in self.ledger_calls:
            self.assertEqual(call.kwargs, {"initial_cash": 1000., "fee_bps": 9., "other_cost_bps": 3.5})
        for ledger, trades, positions in self.runs.values():
            self.assertTrue(ledger.index.equals(expected_days))
            np.testing.assert_allclose((1 + ledger.net_return).cumprod() * 1000,
                                       ledger.equity_close_gbp, rtol=1e-11)
            for date in ["2025-01-01", "2026-01-01"]:
                day = pd.Timestamp(date, tz="UTC")
                previous = day - pd.Timedelta(days=1)
                np.testing.assert_allclose(positions.loc[day], positions.loc[previous])
                self.assertAlmostEqual(ledger.loc[day, "cash_gbp"], ledger.loc[previous, "cash_gbp"])
                self.assertEqual(ledger.loc[day, "traded_notional_gbp"], 0.)

    def test_signed_trades_independently_reconstruct_cash_units_and_daily_equity(self):
        closes = self.prices.pivot(index="timestamp", columns="product_id", values="close_gbp")
        gate = self.audit.set_index("decision_at").risk_on
        for name, (ledger, trades, positions) in self.runs.items():
            quantities = trades.assign(delta=trades.signed_notional_gbp / trades.reference_price_gbp)
            changes = quantities.pivot_table(index="decision_at", columns="product_id", values="delta", aggfunc="sum")
            units = changes.reindex(index=ledger.index, columns=positions.columns).fillna(0).cumsum()
            outflows = trades.signed_notional_gbp + trades.fee_gbp + trades.other_cost_gbp
            cash = 1000 - outflows.groupby(trades.decision_at).sum().reindex(ledger.index, fill_value=0).cumsum()
            equity = cash + (units * closes.reindex(index=ledger.index, columns=units.columns)).sum(axis=1)
            np.testing.assert_allclose(units, positions, rtol=1e-11, atol=1e-10)
            np.testing.assert_allclose(cash, ledger.cash_gbp, rtol=1e-11, atol=1e-9)
            np.testing.assert_allclose(equity, ledger.equity_close_gbp, rtol=1e-11, atol=1e-9)
            np.testing.assert_allclose(trades.fee_gbp, trades.signed_notional_gbp.abs() * .0009)
            np.testing.assert_allclose(trades.other_cost_gbp, trades.signed_notional_gbp.abs() * .00035)
            if name.startswith("filtered_"):
                self.assertTrue(positions.loc[gate.index[~gate]].eq(0).all().all())
                idle = positions.abs().sum(axis=1).lt(1e-12) & ledger.traded_notional_gbp.eq(0)
                np.testing.assert_allclose(ledger.loc[idle, "net_return"], 0., atol=1e-12)

    def test_incomplete_evaluation_end_fails(self):
        truncated = self.prices.loc[self.prices.timestamp.lt(pd.Timestamp("2026-08-31", tz="UTC"))]
        with self.assertRaisesRegex(ValueError, "complete end date"):
            walk_forward.run_bitcoin_walk_forward(
                self.panel, truncated, self.schedule, self.refs, self.choices, weekly_protocol())


if __name__ == "__main__":
    unittest.main()
