"""Independent checks of the ten-lookback development sweep."""
from functools import lru_cache

import numpy as np
import pandas as pd

from _support import (OfflineTestCase, SYMBOLS, START, DEVELOPMENT_END,
                      development_prices, reference_equity)
import crypto_sweep
import crypto_validation

LOOKBACKS = (1, 3, 7, 14, 21, 30, 60, 90, 120, 180)
EVALUATION_START = pd.Timestamp("2021-01-01", tz="UTC")


@lru_cache(maxsize=1)
def sweep_fixture():
    prices = development_prices()
    windows = crypto_validation.build_annual_evaluation_windows(
        START, EVALUATION_START, DEVELOPMENT_END)
    summary, annual, runs = crypto_sweep.run_lookback_sweep(
        prices, SYMBOLS, LOOKBACKS, windows)
    return prices, windows, summary, annual, runs


class SweepTests(OfflineTestCase):
    def test_every_rule_matches_calendar_signals_and_independent_equity(self):
        prices, _, _, _, runs = sweep_fixture()
        closes = prices.set_index(["symbol", "timestamp"])["close"]
        for days, result in runs.items():
            with self.subTest(lookback=days):
                decisions = pd.date_range(
                    START + pd.Timedelta(days=days + 1), DEVELOPMENT_END, freq="W-MON")
                rows = []
                for decision in decisions:
                    execution = decision + pd.Timedelta(days=1)
                    if execution >= DEVELOPMENT_END:
                        continue
                    last = decision - pd.Timedelta(days=1)
                    previous = last - pd.Timedelta(days=days)
                    weights = {symbol: 0.5 if closes.loc[(symbol, last)] > closes.loc[(symbol, previous)] else 0.0
                               for symbol in SYMBOLS}
                    rows.append({"decision_at": decision, "execution_at": execution,
                                 **weights, "GBP_cash": 1 - sum(weights.values())})
                expected_targets = pd.DataFrame(rows)
                pd.testing.assert_frame_equal(result["targets"], expected_targets)
                expected_equity = reference_equity(prices, expected_targets)
                self.assertTrue(result["ledger"].index.equals(expected_equity.index))
                np.testing.assert_allclose(result["ledger"].equity_close, expected_equity,
                                           rtol=1e-11, atol=1e-7)

    def test_common_period_and_annual_compounding(self):
        _, _, summary, annual, runs = sweep_fixture()
        expected_dates = pd.date_range(EVALUATION_START, DEVELOPMENT_END,
                                       freq="D", inclusive="left")
        for days, result in runs.items():
            ledger = result["ledger"]
            actual = ledger.loc[ledger.index >= EVALUATION_START]
            self.assertTrue(actual.index.equals(expected_dates))
            base = ledger.loc[EVALUATION_START - pd.Timedelta(days=1), "equity_close"]
            growth = actual.equity_close.iloc[-1] / base
            self.assertAlmostEqual(summary.loc[days, "total_return"], growth - 1, places=12)
            self.assertAlmostEqual((1 + annual.loc[days, "total_return"]).prod(), growth, places=12)

    def test_reported_aggregate_and_annual_returns_reproduced(self):
        _, _, summary, annual, _ = sweep_fixture()
        # Supplementary regression checks; independent oracles are above.
        expected = {
            1: (14.02, (37.37, -41.27, 41.34)),
            3: (26.96, (84.32, -56.57, 58.58)),
            7: (31.49, (89.48, -38.12, 12.15)),
            14: (-7.24, (67.45, -41.83, -4.78)),
            21: (119.78, (91.25, -16.63, 37.85)),
            30: (417.32, (277.72, -22.93, 77.71)),
            60: (132.12, (156.47, -40.11, 51.13)),
            90: (87.23, (142.04, -39.07, 26.97)),
            120: (198.89, (169.45, -22.90, 43.87)),
            180: (207.69, (150.87, -6.24, 30.81)),
        }
        for days, (total, yearly) in expected.items():
            self.assertEqual(round(100 * summary.loc[days, "total_return"], 2), total)
            for year, value in zip((2021, 2022, 2023), yearly):
                self.assertEqual(round(100 * annual.loc[(days, year), "total_return"], 2), value)
        self.assertEqual(round(100 * summary.loc[30, "cagr"], 2), 73.01)
        self.assertEqual(round(summary.loc[30, "sharpe_zero_cash_rate"], 3), 1.369)
        self.assertEqual(round(100 * summary.loc[30, "max_drawdown"], 2), -37.66)

    def test_synthetic_later_prices_are_excluded(self):
        prices, windows, summary, annual, runs = sweep_fixture()
        future = prices.groupby("symbol", sort=False).tail(1).copy()
        future["timestamp"] = DEVELOPMENT_END
        future[["open", "high", "low", "close"]] = np.nan
        changed = pd.concat([prices, future], ignore_index=True)
        actual_summary, actual_annual, actual_runs = crypto_sweep.run_lookback_sweep(
            changed, SYMBOLS, (30,), windows)
        pd.testing.assert_frame_equal(actual_summary, summary.loc[[30]])
        pd.testing.assert_frame_equal(actual_annual, annual.loc[[(30, y) for y in (2021, 2022, 2023)]])
        pd.testing.assert_frame_equal(actual_runs[30]["ledger"], runs[30]["ledger"])

    def test_invalid_lookbacks_rejected(self):
        prices, windows, _, _, _ = sweep_fixture()
        for invalid in ((), (0,), (-1,), (True,), (1.5,), (30, 30)):
            with self.subTest(lookbacks=invalid), self.assertRaises(ValueError):
                crypto_sweep.run_lookback_sweep(prices, SYMBOLS, invalid, windows)
