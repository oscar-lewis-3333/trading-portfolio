"""Selection, switching and cost checks using the frozen development data."""
import math
import statistics
from functools import lru_cache

import numpy as np
import pandas as pd

from _support import (OfflineTestCase, SYMBOLS, DEVELOPMENT_END, reference_targets,
                      reference_equity, crypto_portfolio, crypto_backtest,
                      crypto_reporting)
from test_sweep import sweep_fixture, EVALUATION_START
import crypto_validation
import crypto_sweep


def reference_scores(runs, windows):
    start = max(run["ledger"].index.min() for run in runs.values()) + pd.Timedelta(days=1)
    scores = {}
    for fold, window in windows.iterrows():
        for days, run in runs.items():
            returns = [row.net_return for date, row in run["ledger"].iterrows()
                       if max(start, window.calibration_start) <= date < window.calibration_end]
            scores[fold, days] = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(365.25)
    return scores


@lru_cache(maxsize=1)
def selection_fixture():
    prices, windows, _, _, runs = sweep_fixture()
    selections, scores = crypto_validation.select_lookbacks_by_past_sharpe(runs, windows)
    targets = crypto_portfolio.build_selected_momentum_targets(runs[90]["targets"], SYMBOLS, runs, selections)
    ledger, trades = crypto_backtest.run_daily_backtest(prices, targets, SYMBOLS)
    return prices, windows, runs, selections, scores, targets, ledger, trades


class SelectionTests(OfflineTestCase):
    def test_past_scores_and_selections_match_independent_statistics(self):
        _, windows, runs, selections, scores, _, _, _ = selection_fixture()
        independent = reference_scores(runs, windows)
        for (fold, days), value in independent.items():
            self.assertAlmostEqual(scores.loc[(fold, days), "training_sharpe"], value, places=12)
        chosen = [max(runs, key=lambda days: (independent[fold, days], days)) for fold in windows.index]
        self.assertEqual(chosen, [14, 30, 30])
        self.assertEqual(list(selections.lookback_days), chosen)

    def test_future_returns_cannot_change_an_earlier_choice(self):
        _, windows, runs, selections, _, _, _, _ = selection_fixture()
        for fold, window in windows.iterrows():
            changed = {}
            for days, run in runs.items():
                history = run["ledger"].copy()
                history.loc[history.index >= window.calibration_end, "net_return"] = 100.0 + days
                changed[days] = {**run, "ledger": history}
            actual, _ = crypto_validation.select_lookbacks_by_past_sharpe(changed, windows.loc[[fold]])
            pd.testing.assert_frame_equal(actual, selections.loc[[fold]])

    def test_exact_score_ties_choose_longer_lookback(self):
        _, windows, runs, _, _, _, _, _ = selection_fixture()
        tied = {14: runs[14], 30: runs[14]}
        actual, _ = crypto_validation.select_lookbacks_by_past_sharpe(tied, windows.loc[[1]])
        self.assertEqual(actual.loc[1, "lookback_days"], 30)

    def test_selected_targets_and_equity_match_calendar_and_algebra(self):
        prices, windows, runs, _, _, actual_targets, ledger, _ = selection_fixture()
        scores = reference_scores(runs, windows)
        choices = {fold: max(runs, key=lambda days: (scores[fold, days], days)) for fold in windows.index}
        expected = reference_targets(prices, "weekly")
        closes = prices.set_index(["symbol", "timestamp"])["close"]
        for i, row in expected.iterrows():
            weights = [0.25, 0.25]
            for fold, window in windows.iterrows():
                if window.test_start <= row.decision_at < window.test_end:
                    last = row.decision_at - pd.Timedelta(days=1)
                    before = last - pd.Timedelta(days=choices[fold])
                    weights = [0.5 if closes.loc[symbol, last] > closes.loc[symbol, before] else 0.0
                               for symbol in SYMBOLS]
            expected.loc[i, SYMBOLS] = weights
            expected.loc[i, "GBP_cash"] = 1 - sum(weights)
        pd.testing.assert_frame_equal(actual_targets[expected.columns], expected)
        independent_equity = reference_equity(prices, expected)
        np.testing.assert_allclose(ledger.equity_close, independent_equity, rtol=1e-11, atol=1e-7)

    def test_costs_use_real_trades_and_no_holdings_reset(self):
        _, _, _, _, _, _, ledger, trades = selection_fixture()
        np.testing.assert_allclose(trades.fee, trades.signed_notional.abs() * 0.0009)
        np.testing.assert_allclose(trades.other_cost, trades.signed_notional.abs() * 0.00035)
        self.assertLess(ledger.rebalance_accounting_error.abs().max(), 1e-8)
        quiet = ~ledger.scheduled_rebalance
        columns = ["cash_gbp"] + [f"units_{symbol}" for symbol in SYMBOLS]
        self.assertTrue(ledger[columns].diff().loc[quiet].eq(0).all().all())
        for year in (2021, 2022, 2023):
            self.assertFalse(ledger.loc[pd.Timestamp(f"{year}-01-01", tz="UTC"), "scheduled_rebalance"])

    def test_misaligned_candidate_execution_rejected(self):
        _, _, runs, selections, _, _, _, _ = selection_fixture()
        changed = dict(runs)
        candidate = runs[14]["targets"].copy()
        mask = candidate.decision_at.eq(pd.Timestamp("2021-01-04", tz="UTC"))
        candidate.loc[mask, "execution_at"] += pd.Timedelta(days=1)
        changed[14] = {**runs[14], "targets": candidate}
        with self.assertRaises(ValueError):
            crypto_portfolio.build_selected_momentum_targets(runs[90]["targets"], SYMBOLS, changed, selections)

    def test_selected_reported_results_and_initial_boundary_difference(self):
        _, windows, runs, _, _, _, ledger, _ = selection_fixture()
        result = crypto_reporting.summarise_backtest_period(ledger, EVALUATION_START, DEVELOPMENT_END)
        self.assertEqual(round(100 * result.total_return, 2), 106.20)
        self.assertEqual(round(100 * result.cagr, 2), 27.30)
        self.assertEqual(round(result.sharpe_zero_cash_rate, 3), 0.739)
        for (_, window), expected in zip(windows.iterrows(), (50.14, -22.72, 77.71)):
            annual = crypto_reporting.summarise_backtest_period(ledger, window.test_start, window.test_end)
            self.assertEqual(round(100 * annual.total_return, 2), expected)
        # Once the first common target has been executed, 2021 daily returns agree.
        dates = (ledger.index >= pd.Timestamp("2021-01-06", tz="UTC")) & (ledger.index < pd.Timestamp("2022-01-01", tz="UTC"))
        np.testing.assert_allclose(ledger.loc[dates, "net_return"],
                                   runs[14]["ledger"].reindex(ledger.index).loc[dates, "net_return"], atol=1e-12, rtol=1e-10)


class CostSensitivityTests(OfflineTestCase):
    def test_cost_scenarios_match_independent_equity_and_recorded_returns(self):
        prices, windows, _, _, runs = sweep_fixture()
        summary, cost_runs = crypto_sweep.run_cost_sensitivity(
            prices, {"momentum_30d": runs[30]["targets"]}, SYMBOLS, windows)
        for cost, expected_return in ((25.0, 417.32), (40.0, 404.84), (60.0, 388.68)):
            ledger = cost_runs[cost, "momentum_30d"]["ledger"]
            expected = reference_equity(prices, runs[30]["targets"], rate=cost / 20000)
            np.testing.assert_allclose(ledger.equity_close, expected, rtol=1e-11, atol=1e-7)
            self.assertEqual(round(summary.loc[(cost, "momentum_30d"), "total_return"] * 100, 2), expected_return)

    def test_invalid_cost_scenarios_rejected(self):
        prices, windows, _, _, runs = sweep_fixture()
        for costs in ((), (25, 25), (float("nan"),), (float("inf"),), (17,)):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                crypto_sweep.run_cost_sensitivity(prices, {"momentum": runs[30]["targets"]}, SYMBOLS, windows, costs)
