"""End-to-end checks of the now-evaluated, frozen 2024-2025 holdout."""
import hashlib
import json
import math
import statistics
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pandas as pd

from _support import (OfflineTestCase, PROJECT, SYMBOLS, saved_prices,
                      reference_equity, crypto_reporting)
from test_selection import selection_fixture
from test_annual_evaluation import fixture as annual_fixture
import crypto_validation
import crypto_holdout

START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2026-01-01", tz="UTC")
PROTOCOL = PROJECT / "research/holdout_protocol_v1.json"


@lru_cache(maxsize=1)
def holdout_fixture():
    prices = saved_prices()
    _, _, development_runs, earlier_choices, _, _, selected, _ = selection_fixture()
    development_benchmarks = annual_fixture()[5]
    primary = crypto_holdout.run_primary_holdout(prices, PROTOCOL, {
        "fixed_30d": development_runs[30]["ledger"],
        "half_crypto_half_cash": development_benchmarks["half_crypto_half_cash"],
    })
    # Capture candidate histories already calculated by the runner, without
    # changing its selection or repeating its full simulation.
    with patch.object(crypto_validation, "select_lookbacks_by_past_sharpe",
                      wraps=crypto_validation.select_lookbacks_by_past_sharpe) as selector:
        secondary = crypto_holdout.run_secondary_holdout(prices, primary, {
            "annual_past_sharpe_selection": selected,
            "weekly_equal_weight": development_benchmarks["weekly_equal_weight"],
        }, earlier_choices)
        candidates = selector.call_args.kwargs["sweep_runs"]
        windows = selector.call_args.kwargs["windows"]
    return prices, primary, secondary, candidates, windows


@lru_cache(maxsize=1)
def independent_choices():
    _, _, _, candidates, windows = holdout_fixture()
    choices, scores = {}, {}
    for fold, window in windows.iterrows():
        for days, run in candidates.items():
            history = run["ledger"]
            sample = list(history.loc[(history.index >= window.calibration_start)
                                      & (history.index < window.calibration_end), "net_return"])
            scores[fold, days] = statistics.mean(sample) / statistics.stdev(sample) * math.sqrt(365.25)
        choices[window.test_start.year] = max(candidates, key=lambda days: (scores[fold, days], days))
    return choices, scores


class HoldoutTests(OfflineTestCase):
    def test_protocol_inference_and_primary_decision_reproduced(self):
        _, primary, _, _, _ = holdout_fixture()
        record = json.loads(PROTOCOL.read_text())
        canonical = json.dumps(record["protocol"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(record["protocol_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
        self.assertEqual(len(primary["returns"]), 731)
        self.assertFalse(primary["primary_criterion_met"])
        for block, low, high in ((28, -0.605, 1.374), (14, -0.573, 1.413), (56, -0.586, 1.365)):
            row = primary["inference"].loc[block]
            self.assertEqual(round(row.sharpe_difference, 3), 0.451)
            self.assertEqual(round(row.ci_lower, 3), low)
            self.assertEqual(round(row.ci_upper, 3), high)

    def test_past_only_selections_match_independent_scores(self):
        _, _, secondary, _, _ = holdout_fixture()
        choices, scores = independent_choices()
        self.assertEqual(choices, {2021: 14, 2022: 30, 2023: 30, 2024: 30, 2025: 30})
        for key, score in scores.items():
            self.assertAlmostEqual(secondary["selection_scores"].loc[key, "training_sharpe"], score, places=12)
        self.assertEqual(list(secondary["holdout_selections"].lookback_days), [30, 30])

    def test_future_returns_cannot_change_2024_or_2025_choice(self):
        _, _, secondary, candidates, windows = holdout_fixture()
        for fold in (4, 5):
            cutoff = windows.loc[fold, "calibration_end"]
            altered = {}
            for days, run in candidates.items():
                ledger = run["ledger"].copy()
                ledger.loc[ledger.index >= cutoff, "net_return"] = 50.0 + days
                altered[days] = {**run, "ledger": ledger}
            actual, _ = crypto_validation.select_lookbacks_by_past_sharpe(altered, windows.loc[[fold]])
            pd.testing.assert_frame_equal(actual, secondary["holdout_selections"].loc[[fold]])

    def test_all_target_schedules_and_cost_equities_match_independent_engine(self):
        prices, _, secondary, _, _ = holdout_fixture()
        choices, _ = independent_choices()
        closes = prices.set_index(["symbol", "timestamp"])["close"]
        for name, actual in secondary["targets"].items():
            # Dates follow the original declared initialization schedules.
            first_decision = pd.Timestamp("2019-02-04" if name == "fixed_30d" else "2019-04-08", tz="UTC")
            rows = []
            for decision in pd.date_range(first_decision, END, freq="W-MON"):
                execution = decision + pd.Timedelta(days=1)
                if execution >= END:
                    continue
                if name == "half_crypto_half_cash" or (name == "annual_past_sharpe_selection" and decision.year < 2021):
                    weights = [0.25, 0.25]
                elif name == "weekly_equal_weight":
                    weights = [0.5, 0.5]
                else:
                    days = 30 if name == "fixed_30d" else choices[decision.year]
                    last = decision - pd.Timedelta(days=1)
                    earlier = last - pd.Timedelta(days=days)
                    weights = [0.5 if closes.loc[(symbol, last)] > closes.loc[(symbol, earlier)] else 0.0 for symbol in SYMBOLS]
                rows.append({"decision_at": decision, "execution_at": execution,
                             **dict(zip(SYMBOLS, weights)), "GBP_cash": 1 - sum(weights)})
            expected = pd.DataFrame(rows)
            pd.testing.assert_frame_equal(actual[expected.columns].reset_index(drop=True), expected)
            for cost in (25.0, 40.0, 60.0):
                with self.subTest(strategy=name, cost=cost):
                    independent = reference_equity(prices, expected, rate=cost / 20000)
                    ledger = secondary["runs"][cost, name]["ledger"]
                    np.testing.assert_allclose(ledger.equity_close, independent, rtol=1e-11, atol=1e-7)

    def test_year_boundary_return_and_rebased_equity_reconcile(self):
        _, _, secondary, _, _ = holdout_fixture()
        for name in secondary["targets"]:
            ledger = secondary["runs"][25.0, name]["ledger"]
            period = ledger.loc[(ledger.index >= START) & (ledger.index < END)]
            before = ledger.loc[START - pd.Timedelta(days=1), "equity_close"]
            self.assertAlmostEqual(period.net_return.iloc[0], period.equity_close.iloc[0] / before - 1, places=12)
            curve = secondary["equity"][name]
            self.assertEqual(curve.index[0], START)
            self.assertEqual(curve.index[-1], END)
            self.assertEqual(len(curve), 732)
            self.assertEqual(curve.iloc[0], 1000)
            np.testing.assert_allclose(curve.iloc[1:], 1000 * (1 + period.net_return).cumprod(), rtol=1e-12)
            annual_growth = (1 + secondary["annual_summary"].xs(name, level="strategy").total_return).prod()
            self.assertAlmostEqual(annual_growth, curve.iloc[-1] / 1000, places=12)

    def test_equal_normalised_performance_is_not_equal_capital(self):
        _, _, secondary, _, _ = holdout_fixture()
        for cost in (25.0, 40.0, 60.0):
            fixed = secondary["runs"][cost, "fixed_30d"]["ledger"]
            annual = secondary["runs"][cost, "annual_past_sharpe_selection"]["ledger"]
            fixed = fixed.loc[fixed.index >= START]
            annual = annual.loc[annual.index >= START]
            np.testing.assert_allclose(fixed.net_return, annual.net_return, atol=1e-12, rtol=1e-10)
            self.assertFalse(np.isclose(fixed.equity_close.iloc[-1], annual.equity_close.iloc[-1]))

    def test_reported_cost_and_annual_returns_reproduced(self):
        _, _, secondary, _, _ = holdout_fixture()
        expected = {
            25.0: (96.50, 37.59, 96.50, 61.27),
            40.0: (92.75, 37.40, 92.75, 60.99),
            60.0: (87.85, 37.15, 87.85, 60.63),
        }
        names = ("fixed_30d", "half_crypto_half_cash", "annual_past_sharpe_selection", "weekly_equal_weight")
        for cost, values in expected.items():
            for name, value in zip(names, values):
                self.assertEqual(round(secondary["cost_summary"].loc[(cost, name), "total_return"] * 100, 2), value)
        annual = {"fixed_30d": (75.35, 12.06), "annual_past_sharpe_selection": (75.35, 12.06),
                  "half_crypto_half_cash": (42.99, -3.78), "weekly_equal_weight": (85.26, -12.95)}
        for name, values in annual.items():
            for year, value in zip((2024, 2025), values):
                self.assertEqual(round(secondary["annual_summary"].loc[(year, name), "total_return"] * 100, 2), value)

    def test_plot_uses_holdout_values_and_drawdowns(self):
        _, _, secondary, _, _ = holdout_fixture()
        equity = secondary["equity"]
        figure = crypto_holdout.plot_holdout_equity(equity)
        self.assertEqual(len(figure.axes), 2)
        for column, name in enumerate(equity.columns):
            np.testing.assert_allclose(figure.axes[0].lines[column].get_ydata(), equity[name])
            np.testing.assert_allclose(figure.axes[1].lines[column].get_ydata(), equity[name] / equity[name].cummax() - 1)
