
#ensuring 2024-2025 replays as expected
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import socket
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_uk_extension_features import PROJECT, NOTEBOOK, run_stage
import reversal_namespace  # widens the trading_portfolio package path
import trading_portfolio.reversal_signals.src.reversal_preparation as reversal_preparation
import reversal_validation
import reversal_walk_forward


def saved_cell(name, namespace):
    matches = []
    for number, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        names = {node.id for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        if name in names:
            matches.append((number, source))
    if len(matches) != 1:
        raise AssertionError(f"Expected one saved cell defining {name}")
    number, source = matches[0]
    with redirect_stdout(io.StringIO()):
        exec(compile(source, f"{NOTEBOOK}:cell-{number}", "exec"), namespace)


class EvaluationStartTests(unittest.TestCase):
    def run_targets(self, on_cash_date):
        dates = pd.bdate_range("2023-12-19", "2024-01-12").difference(
            pd.to_datetime(["2023-12-25", "2023-12-26", "2024-01-01"])).rename("Date")
        cash = pd.Timestamp("2023-12-29")
        index = pd.MultiIndex.from_product([dates, ["A.L", "B.L"]], names=["Date", "ticker"])
        features = pd.DataFrame({
            "signal_time": index.get_level_values("Date").tz_localize("UTC") + pd.Timedelta(hours=16),
            "eligible": True, "raw_reversal_score": [.2, -.1] * len(dates),
        }, index=index)
        # The 3-session schedule either includes the final December close or
        # next rebalances in January; the selection itself remains in December.
        anchor = 0 if on_cash_date else 1
        calendar = dates[anchor:-4:3]
        choices = pd.DataFrame({
            "selection_date": [dates[0]], "test_start": [dates[1]], "test_end": [dates[-1]],
            "allow_new_entries": [True], "formation_sessions": [1], "holding_sessions": [3],
            "top_frac": [.5], "configuration_id": [27],
        }, index=pd.Index([27], name="fold"))
        ns = dict(pd=pd, np=np, uk_extension_sessions=dates,
                  uk_extension_choices=choices, uk_extension_features={1: features},
                  uk_extension_formation_windows=[1],
                  uk_extension_decision_dates={3: calendar},
                  uk_extension_frozen_spec={"minimum_eligible": 1},
                  reversal_walk_forward=reversal_walk_forward)
        saved_cell("uk_extension_decisions", ns)
        self.assertEqual(ns["uk_extension_cash_date"], cash)
        self.assertEqual(ns["uk_extension_decisions"].index.get_level_values("Date").min(), cash)
        return ns

    def test_pre_year_selection_and_cash_date_signal_are_retained(self):
        ns = self.run_targets(True)
        cash_targets = ns["uk_extension_decisions"].xs(ns["uk_extension_cash_date"], level="Date")
        self.assertEqual(cash_targets.target_weight.sum(), 1.)
        self.assertTrue(cash_targets.configuration_id.eq(27).all())
        self.assertTrue(cash_targets.holding_sessions.eq(3).all())

    def test_cash_marker_does_not_create_an_unscheduled_rebalance(self):
        ns = self.run_targets(False)
        targets = ns["uk_extension_decisions"]
        initial = targets.xs(ns["uk_extension_cash_date"], level="Date")
        self.assertTrue(initial.target_weight.eq(0).all())
        self.assertFalse(initial.eligible.any())
        self.assertTrue(targets.loc[(pd.Timestamp("2024-01-02"), "A.L"), "target_weight"] > 0)


class CachedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook_hash = reversal_preparation.sha256(NOTEBOOK)
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline checkpoint")):
            prepared = reversal_preparation.prepare_uk_extension(PROJECT)
            ns = run_stage(prepared["prices"], prepared["schedule"], prepared["manifest"]["frozen_strategy"])
            ns.update(_REPO_ROOT=PROJECT.parent, hashlib=hashlib, json=json,
                      uk_extension_data=prepared, uk_extension_market=prepared["market"],
                      reversal_validation=reversal_validation)
            saved_cell("uk_extension_decision_dates", ns)
            saved_cell("uk_extension_candidate_cache", ns)
            cls.cache_hashes = {p: reversal_preparation.sha256(p)
                                for p in ns["uk_extension_candidate_cache"].glob("configuration_*.pkl")}
            if len(cls.cache_hashes) != 36:
                raise AssertionError("All 36 cached candidates are required")
            with patch.object(reversal_validation, "evaluate_execution_configuration",
                              side_effect=AssertionError("Candidate cache unexpectedly missed")):
                saved_cell("uk_extension_net_excess", ns)
            saved_cell("uk_extension_choices", ns)
            saved_cell("uk_extension_decisions", ns)
            saved_cell("uk_extension_results", ns)
        cls.ns = ns

    def test_replay_matches_saved_totals_and_starts_without_inherited_assets(self):
        ns = self.ns
        self.assertEqual(ns["uk_extension_cash_date"], pd.Timestamp("2023-12-29"))
        for name, expected in [("Walk-forward", 12713.81), ("Matched benchmark", 10266.48)]:
            with self.subTest(portfolio=name):
                result = ns["uk_extension_results"][name]
                daily = result["daily"]
                self.assertEqual(len(daily), 508)
                self.assertEqual(daily.index[-1], pd.Timestamp("2025-12-31"))
                self.assertEqual(daily.iloc[0].cash_gbp, 10000.)
                self.assertTrue(daily.iloc[0][["stock_value_gbp", "positions", "dividend_reserve_gbp", "cost_gbp"]].eq(0).all())
                fills = result["orders"].loc[lambda x: x.status.eq("assumed_fill")]
                self.assertEqual(fills.Date.min(), pd.Timestamp("2024-01-02"))
                self.assertAlmostEqual(daily.iloc[-1].equity_gbp, expected, delta=.005)
                self.assertEqual(len(result["final_positions"]), 0)
                self.assertEqual(daily.iloc[-1].stale_value_gbp, 0.)

    def test_fees_and_cash_balance_reconcile_to_executed_orders(self):
        rate = self.ns["uk_extension_wf_rules"]["cost_per_side_bps"] / 10000.
        for name, result in self.ns["uk_extension_results"].items():
            with self.subTest(portfolio=name):
                daily, orders = result["daily"], result["orders"]
                fills = orders.loc[orders.status.eq("assumed_fill")]
                np.testing.assert_allclose(fills.cost_gbp, fills.notional_gbp * rate, atol=1e-9)
                expected_cash_change = np.where(fills.side.eq("buy"), -fills.notional_gbp - fills.cost_gbp,
                                                fills.notional_gbp - fills.cost_gbp)
                np.testing.assert_allclose(fills.cash_change_gbp, expected_cash_change, atol=1e-9)
                expected_cash = 10000. + fills.groupby("Date").cash_change_gbp.sum().reindex(daily.index, fill_value=0).cumsum()
                np.testing.assert_allclose(daily.cash_gbp, expected_cash, atol=1e-7)
                expected_fees = fills.groupby("Date").cost_gbp.sum().reindex(daily.index, fill_value=0)
                np.testing.assert_allclose(daily.cost_gbp, expected_fees, atol=1e-9)
                self.assertTrue(orders.loc[~orders.status.eq("assumed_fill"), ["cost_gbp", "cash_change_gbp"]].eq(0).all().all())
                self.assertTrue(daily.cash_gbp.ge(-1e-8).all())

    def test_holdings_reconcile_to_signed_trade_units(self):
        for name, result in self.ns["uk_extension_results"].items():
            with self.subTest(portfolio=name):
                fills = result["orders"].loc[lambda x: x.status.eq("assumed_fill")].copy()
                fills["signed_units"] = np.where(fills.side.eq("buy"), fills.units, -fills.units)
                units = result["holdings"].units.unstack("ticker").reindex(result["daily"].index).fillna(0.)
                trade_units = fills.pivot_table(index="Date", columns="ticker", values="signed_units", aggfunc="sum")
                expected = trade_units.reindex(index=units.index, columns=units.columns).fillna(0.).cumsum()
                np.testing.assert_allclose(units, expected, atol=1e-7, rtol=1e-10)
                holdings = result["holdings"]
                np.testing.assert_allclose(holdings.value_gbp, holdings.units * holdings.mark_gbp)
                values = holdings.groupby(level="Date").value_gbp.sum().reindex(units.index, fill_value=0)
                np.testing.assert_allclose(result["daily"].stock_value_gbp, values, atol=1e-8)

    def test_overnight_dividends_and_total_equity_reconcile(self):
        market = self.ns["uk_extension_eval_market"]
        for name, result in self.ns["uk_extension_results"].items():
            with self.subTest(portfolio=name):
                daily = result["daily"]
                units = result["holdings"].units.unstack("ticker").reindex(daily.index).fillna(0.)
                previous_units = units.shift(1).fillna(0.)
                dividends = market.dividend_gbp.unstack("ticker").reindex(index=units.index, columns=units.columns)
                entitlement = np.where(previous_units.gt(0), previous_units * dividends, 0.).sum(axis=1)
                self.assertTrue(np.isfinite(entitlement).all())
                np.testing.assert_allclose(daily.dividends_accrued_gbp, entitlement, atol=1e-8)
                np.testing.assert_allclose(daily.dividend_reserve_gbp, entitlement.cumsum(), atol=1e-8)
                np.testing.assert_allclose(daily.equity_gbp, daily.cash_gbp + daily.stock_value_gbp + daily.dividend_reserve_gbp)

    def test_orders_use_valid_opens_and_buys_follow_prior_close_targets(self):
        ns = self.ns
        market, schedule = ns["uk_extension_eval_market"], ns["uk_extension_eval_schedule"]
        for name, targets in [("Walk-forward", ns["uk_extension_decisions"]),
                              ("Matched benchmark", ns["uk_extension_benchmark_decisions"])]:
            fills = ns["uk_extension_results"][name]["orders"].loc[lambda x: x.status.eq("assumed_fill")]
            keys = pd.MultiIndex.from_frame(fills[["Date", "ticker"]])
            quotes = market.loc[keys]
            self.assertTrue(quotes.open_reference_usable.all())
            self.assertFalse(quotes.known_suspended.any())
            np.testing.assert_allclose(fills.price_gbp, quotes.open_gbp)
            np.testing.assert_array_equal(fills.volume_warning, ~quotes.positive_reported_volume)
            buys = fills.loc[fills.side.eq("buy")]
            positions = schedule.index.get_indexer(buys.Date)
            self.assertTrue((positions > 0).all())
            prior = schedule.index[positions - 1]
            planned_keys = pd.MultiIndex.from_arrays([prior, buys.ticker], names=["Date", "ticker"])
            planned = targets.loc[planned_keys]
            self.assertTrue(planned.target_weight.gt(0).all())
            np.testing.assert_array_equal(planned.signal_time, schedule.loc[prior, "market_close"])

    def test_benchmark_matches_planned_exposure_and_future_returns_cannot_change_initial_choice(self):
        ns = self.ns
        strategy = ns["uk_extension_decisions"].target_weight.groupby(level="Date").sum()
        benchmark = ns["uk_extension_benchmark_decisions"].target_weight.groupby(level="Date").sum()
        np.testing.assert_allclose(strategy, benchmark, atol=1e-12)
        self.assertTrue(strategy.le(1 + 1e-12).all())
        changed = ns["uk_extension_net_excess"].copy()
        changed.loc[changed.index >= "2024-01-01", 0] = 100.
        choices = reversal_walk_forward.select_walk_forward_configurations(
            changed, ns["uk_extension_folds"], ns["uk_extension_grid"], ns["uk_extension_wf_rules"])
        pd.testing.assert_frame_equal(choices.loc[:27], ns["uk_extension_choices"].loc[:27])
        self.assertEqual(choices.loc[28, "configuration_id"], 0)

    def test_frozen_sources_notebook_and_candidate_caches_are_unchanged(self):
        self.assertEqual(reversal_preparation.sha256(NOTEBOOK), self.notebook_hash)
        for path, digest in self.cache_hashes.items():
            self.assertEqual(reversal_preparation.sha256(path), digest)
        protected = json.loads((PROJECT / "data/uk_extension_2023_2025_v1/preparation_backup_v1/protected_sha256.json").read_text())
        for path, digest in protected.items():
            reversal_preparation.verify_frozen_source(PROJECT / path, digest)

    @classmethod
    def tearDownClass(cls):
        report = {}
        for name, result in cls.ns["uk_extension_results"].items():
            daily, orders = result["daily"], result["orders"]
            fills = orders.loc[orders.status.eq("assumed_fill")]
            report[name] = dict(final_equity_gbp=float(daily.equity_gbp.iloc[-1]),
                                costs_gbp=float(daily.cost_gbp.sum()), executed_orders=len(fills),
                                zero_volume_flagged_fills=int(fills.volume_warning.sum()),
                                blocked_orders=int(orders.status.eq("blocked").sum()),
                                maximum_stale_value_pct=float((100 * daily.stale_value_gbp / daily.equity_gbp).max()))
        print("Execution audit: " + json.dumps(report))


if __name__ == "__main__":
    unittest.main()
