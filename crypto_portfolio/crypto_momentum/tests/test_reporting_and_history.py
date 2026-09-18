import math
import statistics

import numpy as np
import pandas as pd

from _support import (OfflineTestCase, SYMBOLS, DEVELOPMENT_END, development_prices,
                      reference_targets, reference_equity, crypto_features,
                      crypto_portfolio, crypto_backtest, crypto_reporting)


class ReportingTests(OfflineTestCase):
    @staticmethod
    def ledger_for(equity):
        previous = [1000.0] + equity[:-1]
        dates = pd.date_range("2020-01-01", periods=len(equity), tz="UTC")
        return pd.DataFrame({
            "equity_open": previous, "equity_close": equity,
            "valuation_at": dates + pd.Timedelta(days=1),
            "net_return": [x / y - 1 for x, y in zip(equity, previous)],
            "cash_gbp": np.asarray(equity) * 0.25,
            "fee": [0.9] + [0] * (len(equity) - 1),
            "other_cost": [0.35] + [0] * (len(equity) - 1),
            "traded_notional": [1000] + [0] * (len(equity) - 1),
        }, index=dates)

    def test_metrics_match_hand_calculation(self):
        ledger = self.ledger_for([900.0, 950.0, 800.0])
        summary = crypto_reporting.summarise_backtest(ledger)
        returns = [-0.1, 950 / 900 - 1, 800 / 950 - 1]
        expected_volatility = statistics.stdev(returns) * math.sqrt(365.25)
        expected_sharpe = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(365.25)
        self.assertAlmostEqual(summary.total_return, -0.2)
        self.assertAlmostEqual(summary.cagr, 0.8 ** (365.25 / 3) - 1)
        self.assertAlmostEqual(summary.annualised_volatility, expected_volatility)
        self.assertAlmostEqual(summary.sharpe_zero_cash_rate, expected_sharpe)
        self.assertAlmostEqual(summary.mean_crypto_weight, 0.75)
        self.assertAlmostEqual(summary.annualised_one_way_turnover, 365.25 / 3)
        self.assertAlmostEqual(summary.total_cost_gbp, 1.25)

    def test_drawdown_includes_starting_capital_and_entry_loss(self):
        summary = crypto_reporting.summarise_backtest(self.ledger_for([900.0, 950.0, 800.0]))
        self.assertAlmostEqual(summary.max_drawdown, -0.2)

    def test_zero_volatility_has_undefined_sharpe(self):
        summary = crypto_reporting.summarise_backtest(self.ledger_for([1000.0] * 3))
        self.assertEqual(summary.annualised_volatility, 0)
        self.assertTrue(np.isnan(summary.sharpe_zero_cash_rate))

    def test_incorrect_initial_cash_rejected(self):
        with self.assertRaises(ValueError):
            crypto_reporting.summarise_backtest(self.ledger_for([1000.0] * 3), initial_cash=2000)


class HistoricalIntegrationTests(OfflineTestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = development_prices()
        features = crypto_features.add_momentum_features(cls.prices, 90)
        targets = crypto_portfolio.build_weekly_targets(features, SYMBOLS, DEVELOPMENT_END)
        cls.runs = {}
        cls.references = {}
        for mode in ("momentum", "buy_and_hold", "weekly"):
            actual_targets = targets if mode == "momentum" else crypto_portfolio.build_benchmark_targets(targets, SYMBOLS, mode)
            cls.runs[mode] = crypto_backtest.run_daily_backtest(cls.prices, actual_targets, SYMBOLS)
            cls.references[mode] = reference_equity(cls.prices, reference_targets(cls.prices, mode))

    def test_every_equity_observation_matches_independent_engine(self):
        for mode, (ledger, _) in self.runs.items():
            with self.subTest(mode=mode):
                self.assertEqual(len(ledger), 1728)
                self.assertTrue(ledger.index.equals(self.references[mode].index))
                np.testing.assert_allclose(ledger.equity_close, self.references[mode], rtol=1e-11, atol=1e-7)

    def test_buy_and_hold_matches_direct_units_formula(self):
        ledger, trades = self.runs["buy_and_hold"]
        opening = self.prices.loc[self.prices.timestamp == ledger.index.min()].set_index("symbol")["open"]
        closing = self.prices.loc[self.prices.timestamp == ledger.index.max()].set_index("symbol")["close"]
        expected = sum((1000 / 1.00125) * 0.5 * closing[s] / opening[s] for s in SYMBOLS)
        self.assertAlmostEqual(ledger.equity_close.iloc[-1], expected, places=8)
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual((ledger.fee + ledger.other_cost).sum(), 1000 * 0.00125 / 1.00125)

    def test_ledger_identities_and_costs(self):
        for mode, (ledger, trades) in self.runs.items():
            with self.subTest(mode=mode):
                np.testing.assert_allclose((1 + ledger.net_return).cumprod() * 1000, ledger.equity_close, rtol=1e-12)
                self.assertLess(ledger.rebalance_accounting_error.abs().max(), 1e-8)
                quiet = ~ledger.scheduled_rebalance
                self.assertTrue(ledger.loc[quiet, ["fee", "other_cost", "traded_notional"]].eq(0).all().all())
                self.assertTrue(ledger[[f"units_{s}" for s in SYMBOLS]].diff().loc[quiet].eq(0).all().all())
                self.assertTrue(ledger.cash_gbp.ge(0).all())
                np.testing.assert_allclose(trades.fee, trades.signed_notional.abs() * 0.0009)
                np.testing.assert_allclose(trades.other_cost, trades.signed_notional.abs() * 0.00035)
                self.assertTrue(trades.execution_at.lt(DEVELOPMENT_END).all())

    def test_reported_results_reproduced_as_regression_checks(self):
        # These recorded results supplement, rather than replace, independent oracles.
        expected = {
            "momentum": (4569.275873, 37.87, 57.09, 0.858, -63.40),
            "buy_and_hold": (10581.666798, 64.65, 74.33, 1.051, -74.47),
            "weekly": (11815.032902, 68.53, 72.96, 1.089, -73.04),
        }
        for mode, (ledger, _) in self.runs.items():
            with self.subTest(mode=mode):
                summary = crypto_reporting.summarise_backtest(ledger)
                equity, cagr, vol, sharpe, drawdown = expected[mode]
                self.assertAlmostEqual(summary.final_equity_gbp, equity, places=5)
                self.assertEqual(round(summary.cagr * 100, 2), cagr)
                self.assertEqual(round(summary.annualised_volatility * 100, 2), vol)
                self.assertEqual(round(summary.sharpe_zero_cash_rate, 3), sharpe)
                self.assertEqual(round(summary.max_drawdown * 100, 2), drawdown)

    def test_sharpe_uses_arithmetic_mean_not_cagr(self):
        for mode, (ledger, _) in self.runs.items():
            with self.subTest(mode=mode):
                summary = crypto_reporting.summarise_backtest(ledger)
                expected = statistics.mean(ledger.net_return) * 365.25 / summary.annualised_volatility
                self.assertAlmostEqual(summary.sharpe_zero_cash_rate, expected, places=12)
