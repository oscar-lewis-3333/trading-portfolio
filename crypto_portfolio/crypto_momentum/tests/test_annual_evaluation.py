"""Offline checks for calibration, continuous holdings and annual reporting."""
import math
import statistics
from functools import lru_cache

import numpy as np
import pandas as pd

from _support import (
    OfflineTestCase, SYMBOLS, START, DEVELOPMENT_END, development_prices,
    reference_equity, reference_targets, crypto_features, crypto_portfolio,
    crypto_backtest, crypto_reporting,
)
import crypto_validation


def utc(value):
    return pd.Timestamp(value, tz="UTC")


@lru_cache(maxsize=1)
def fixture():
    prices = development_prices()
    features = crypto_features.add_momentum_features(prices, 90)
    targets = crypto_portfolio.build_weekly_targets(features, SYMBOLS, DEVELOPMENT_END)
    momentum, _ = crypto_backtest.run_daily_backtest(prices, targets, SYMBOLS)
    windows = crypto_validation.build_annual_evaluation_windows(
        START, utc("2021-01-01"), DEVELOPMENT_END)
    allocations = crypto_validation.calibrate_benchmark_allocations(momentum, windows)
    calibrated = crypto_portfolio.build_calibrated_mix_targets(
        targets, SYMBOLS, windows, allocations)
    schedules = {
        "momentum": targets,
        "buy_and_hold": crypto_portfolio.build_benchmark_targets(targets, SYMBOLS, "buy_and_hold"),
        "weekly_equal_weight": crypto_portfolio.build_benchmark_targets(targets, SYMBOLS, "weekly"),
        "half_crypto_half_cash": crypto_portfolio.build_constant_mix_targets(targets, SYMBOLS),
        "calibrated_crypto_cash": calibrated,
    }
    ledgers = {"momentum": momentum}
    for name, schedule in schedules.items():
        if name != "momentum":
            ledgers[name], _ = crypto_backtest.run_daily_backtest(prices, schedule, SYMBOLS)
    return prices, targets, windows, allocations, schedules, ledgers


class CalibrationTests(OfflineTestCase):
    def test_windows_and_calendar_counts(self):
        _, _, windows, allocations, _, _ = fixture()
        self.assertEqual(list(windows.test_start.dt.year), [2021, 2022, 2023])
        self.assertTrue(windows.calibration_start.eq(START).all())
        self.assertTrue(windows.calibration_end.eq(windows.test_start).all())
        self.assertEqual(list(allocations.calibration_days), [633, 998, 1363])

    def test_calibration_matches_independent_python_average(self):
        _, _, windows, allocations, _, ledgers = fixture()
        for fold, window in windows.iterrows():
            observations = [1 - row.cash_gbp / row.equity_close
                            for date, row in ledgers["momentum"].iterrows()
                            if window.calibration_start <= date < window.calibration_end]
            self.assertAlmostEqual(allocations.loc[fold, "crypto_weight"],
                                   statistics.mean(observations), places=12)

    def test_future_exposure_cannot_change_earlier_allocation(self):
        _, _, windows, allocations, _, ledgers = fixture()
        for fold, window in windows.iterrows():
            changed = ledgers["momentum"].copy(deep=True)
            later = changed.index >= window.calibration_end
            changed.loc[later, "cash_gbp"] = changed.loc[later, "equity_close"]
            actual = crypto_validation.calibrate_benchmark_allocations(changed, windows.loc[[fold]])
            pd.testing.assert_frame_equal(actual, allocations.loc[[fold]])

    def test_missing_history_and_unavailable_close_rejected(self):
        _, _, windows, _, _, ledgers = fixture()
        original = ledgers["momentum"]
        gap = original.drop(utc("2020-06-01"))
        delayed = original.copy()
        delayed.loc[utc("2020-12-31"), "valuation_at"] = utc("2021-01-02")
        for invalid in (gap, delayed):
            with self.subTest(case=len(invalid)), self.assertRaises(ValueError):
                crypto_validation.calibrate_benchmark_allocations(invalid, windows.loc[[1]])

    def test_window_inputs_reject_naive_nonannual_and_reversed_dates(self):
        for first, test, end in [
            ("2019-01-01", utc("2021-01-01"), DEVELOPMENT_END),
            (START, utc("2021-02-01"), DEVELOPMENT_END),
            (START, DEVELOPMENT_END, utc("2021-01-01")),
        ]:
            with self.subTest(test=test), self.assertRaises(ValueError):
                crypto_validation.build_annual_evaluation_windows(first, test, end)


class CalibratedScheduleTests(OfflineTestCase):
    def test_expected_first_executions_and_initial_weights(self):
        _, _, _, allocations, schedules, _ = fixture()
        targets = schedules["calibrated_crypto_cash"]
        expected = [utc("2021-01-05"), utc("2022-01-04"), utc("2023-01-03")]
        for fold, execution in enumerate(expected, 1):
            selected = targets.loc[targets.calibration_fold == fold]
            self.assertEqual(selected.execution_at.iloc[0], execution)
            np.testing.assert_allclose(selected[SYMBOLS].sum(axis=1), allocations.loc[fold, "crypto_weight"])
        initial = targets.loc[targets.calibration_fold == 0]
        np.testing.assert_allclose(initial[SYMBOLS], 0.25)
        np.testing.assert_allclose(initial.GBP_cash, 0.5)

    def test_year_boundary_uses_decision_not_execution_date(self):
        # Monday 31 December -> Tuesday 1 January must retain old weights.
        windows = crypto_validation.build_annual_evaluation_windows(
            utc("2017-01-01"), utc("2019-01-01"), utc("2020-01-01"))
        decisions = pd.to_datetime(["2018-12-31", "2019-01-07"], utc=True)
        schedule = pd.DataFrame({"decision_at": decisions,
                                 "execution_at": decisions + pd.Timedelta(days=1)})
        allocation = pd.DataFrame({"last_valuation_at": [utc("2019-01-01")],
                                   "crypto_weight": [0.8]}, index=[1])
        actual = crypto_portfolio.build_calibrated_mix_targets(schedule, SYMBOLS, windows, allocation)
        np.testing.assert_allclose(actual[SYMBOLS].sum(axis=1), [0.5, 0.8])
        self.assertEqual(list(actual.calibration_fold), [0, 1])

    def test_later_allocation_changes_do_not_change_earlier_targets(self):
        _, targets, windows, allocations, schedules, _ = fixture()
        changed = allocations.copy()
        changed.loc[3, "crypto_weight"] = 0.0
        actual = crypto_portfolio.build_calibrated_mix_targets(targets, SYMBOLS, windows, changed)
        original = schedules["calibrated_crypto_cash"]
        earlier = original.decision_at < utc("2023-01-01")
        pd.testing.assert_frame_equal(actual.loc[earlier], original.loc[earlier])

    def test_cash_endpoints_and_invalid_weight(self):
        _, targets, _, _, _, _ = fixture()
        for weight in (0.0, 1.0):
            actual = crypto_portfolio.build_constant_mix_targets(targets, SYMBOLS, weight)
            np.testing.assert_allclose(actual[SYMBOLS].sum(axis=1), weight)
            np.testing.assert_allclose(actual.GBP_cash, 1 - weight)
        with self.assertRaises(ValueError):
            crypto_portfolio.build_constant_mix_targets(targets, SYMBOLS, float("nan"))

    def test_continuous_benchmarks_match_separate_allocation_solver(self):
        prices, _, windows, _, _, ledgers = fixture()
        for mode in ("half_crypto_half_cash", "calibrated_crypto_cash"):
            independent = reference_targets(prices, "weekly")
            for i, row in independent.iterrows():
                weight = 0.5
                if mode == "calibrated_crypto_cash":
                    for _, window in windows.iterrows():
                        if window.test_start <= row.decision_at < window.test_end:
                            history = ledgers["momentum"].loc[
                                (ledgers["momentum"].index >= window.calibration_start)
                                & (ledgers["momentum"].index < window.calibration_end)]
                            weight = statistics.mean(1 - r.cash_gbp / r.equity_close
                                                     for _, r in history.iterrows())
                independent.loc[i, SYMBOLS] = weight / 2
                independent.loc[i, "GBP_cash"] = 1 - weight
            expected = reference_equity(prices, independent)
            np.testing.assert_allclose(ledgers[mode].equity_close, expected, rtol=1e-11, atol=1e-7)

    def test_holdings_and_cash_carry_through_january_without_reset(self):
        _, _, windows, _, _, ledgers = fixture()
        ledger = ledgers["calibrated_crypto_cash"]
        for _, window in windows.iterrows():
            first = window.test_start
            previous = first - pd.Timedelta(days=1)
            self.assertFalse(ledger.loc[first, "scheduled_rebalance"])
            for column in ["cash_gbp"] + [f"units_{s}" for s in SYMBOLS]:
                self.assertEqual(ledger.loc[first, column], ledger.loc[previous, column])


class AnnualReportingTests(OfflineTestCase):
    @staticmethod
    def boundary_ledger():
        dates = pd.date_range("2020-12-31", periods=3, tz="UTC")
        return pd.DataFrame({
            "equity_close": [1000.0, 900.0, 990.0],
            "equity_open": [1000.0, 1100.0, 900.0],
            "valuation_at": dates + pd.Timedelta(days=1),
            "net_return": [0.0, -0.1, 0.1],
            "cash_gbp": [0.0, 0.0, 0.0],
            "fee": [0.0, 0.0, 0.0], "other_cost": [0.0, 0.0, 0.0],
            "traded_notional": [0.0, 0.0, 0.0],
        }, index=dates)

    def test_previous_close_not_january_open_anchors_return_and_drawdown(self):
        result = crypto_reporting.summarise_backtest_period(
            self.boundary_ledger(), utc("2021-01-01"), utc("2021-01-03"))
        self.assertAlmostEqual(result.total_return, -0.01)
        self.assertAlmostEqual(result.max_drawdown, -0.10)
        self.assertAlmostEqual(result.annualised_volatility, statistics.stdev([-0.1, 0.1]) * math.sqrt(365.25))

    def test_missing_day_prior_close_and_corrupted_return_are_rejected(self):
        original = self.boundary_ledger()
        corrupted = original.copy()
        corrupted.loc[utc("2021-01-01"), "net_return"] = 0.0
        for invalid in (original.iloc[1:], original.drop(utc("2021-01-02")), corrupted):
            with self.subTest(rows=len(invalid)), self.assertRaises(ValueError):
                crypto_reporting.summarise_backtest_period(invalid, utc("2021-01-01"), utc("2021-01-03"))

    def test_all_annual_metrics_match_independent_calculation(self):
        _, _, windows, _, _, ledgers = fixture()
        for name, ledger in ledgers.items():
            for _, window in windows.iterrows():
                with self.subTest(strategy=name, year=window.test_start.year):
                    actual = crypto_reporting.summarise_backtest_period(ledger, window.test_start, window.test_end)
                    base = float(ledger.loc[window.test_start - pd.Timedelta(days=1), "equity_close"])
                    sample = ledger.loc[(ledger.index >= window.test_start) & (ledger.index < window.test_end)]
                    values = list(sample.equity_close)
                    returns = [b / a - 1 for a, b in zip([base] + values[:-1], values)]
                    peak, worst = base, 0.0
                    for value in values:
                        peak = max(peak, value)
                        worst = min(worst, value / peak - 1)
                    self.assertAlmostEqual(actual.total_return, values[-1] / base - 1, places=12)
                    self.assertAlmostEqual(actual.annualised_volatility, statistics.stdev(returns) * math.sqrt(365.25), places=12)
                    self.assertAlmostEqual(actual.sharpe_zero_cash_rate, statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(365.25), places=12)
                    self.assertAlmostEqual(actual.max_drawdown, worst, places=12)

    def test_annual_returns_compound_to_continuous_period_return(self):
        _, _, windows, _, _, ledgers = fixture()
        for name, ledger in ledgers.items():
            growth = 1.0
            for _, window in windows.iterrows():
                growth *= 1 + crypto_reporting.summarise_backtest_period(
                    ledger, window.test_start, window.test_end).total_return
            expected = ledger.loc[utc("2023-12-31"), "equity_close"] / ledger.loc[utc("2020-12-31"), "equity_close"]
            self.assertAlmostEqual(growth, expected, places=12)

    def test_source_frames_are_preserved(self):
        _, targets, windows, allocations, _, ledgers = fixture()
        originals = [x.copy(deep=True) for x in (targets, windows, allocations, ledgers["momentum"])]
        crypto_validation.calibrate_benchmark_allocations(ledgers["momentum"], windows)
        crypto_portfolio.build_calibrated_mix_targets(targets, SYMBOLS, windows, allocations)
        crypto_reporting.summarise_backtest_period(ledgers["momentum"], utc("2021-01-01"), utc("2022-01-01"))
        for actual, original in zip((targets, windows, allocations, ledgers["momentum"]), originals):
            pd.testing.assert_frame_equal(actual, original)
