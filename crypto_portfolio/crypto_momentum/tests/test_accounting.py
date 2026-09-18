import numpy as np
import pandas as pd

from _support import (OfflineTestCase, SYMBOLS, crypto_backtest,
                      reference_rebalance, toy_prices, toy_targets)


class AccountingTests(OfflineTestCase):
    def test_cash_entry_and_liquidation_closed_form(self):
        prices = toy_prices([[100, 200]] * 3, [[100, 200]] * 3)
        targets = toy_targets([[0.5, 0.5], [0, 0]], [0, 2])
        ledger, _ = crypto_backtest.run_daily_backtest(prices, targets, SYMBOLS)
        c = 0.00125
        np.testing.assert_allclose(ledger.equity_close,
            [1000 / (1 + c), 1000 / (1 + c), 1000 * (1 - c) / (1 + c)])
        np.testing.assert_allclose(ledger.fee + ledger.other_cost,
            [1000 * c / (1 + c), 0, 1000 * c / (1 + c)], atol=1e-10)
        self.assertTrue(ledger.iloc[-1][[f"units_{s}" for s in SYMBOLS]].eq(0).all())

    def test_no_trade_when_weights_and_prices_unchanged(self):
        prices = toy_prices([[100, 200]] * 3, [[100, 200]] * 3)
        targets = toy_targets([[0.5, 0.5], [0.5, 0.5]], [0, 2])
        ledger, _ = crypto_backtest.run_daily_backtest(prices, targets, SYMBOLS)
        self.assertLess(abs(ledger.iloc[-1].traded_notional), 1e-9)

    def test_partial_investment_closed_form(self):
        prices = toy_prices([[100, 200]] * 3, [[100, 200]] * 3)
        ledger, _ = crypto_backtest.run_daily_backtest(prices, toy_targets([[0.5, 0]], [0]), SYMBOLS)
        expected = 1000 / (1 + 0.5 * 0.00125)
        np.testing.assert_allclose(ledger.equity_close, expected)
        np.testing.assert_allclose(ledger.cash_gbp, 0.5 * expected)

    def test_cash_only_is_constant(self):
        prices = toy_prices([[100, 200], [80, 500], [20, 400]], [[101, 199], [70, 300], [10, 350]])
        ledger, _ = crypto_backtest.run_daily_backtest(prices, toy_targets([[0, 0]], [0]), SYMBOLS)
        np.testing.assert_allclose(ledger.equity_close, 1000)
        np.testing.assert_allclose(ledger.traded_notional, 0)
        np.testing.assert_allclose(ledger.fee + ledger.other_cost, 0)

    def test_mark_to_market_preserves_units(self):
        prices = toy_prices([[100, 200], [110, 200], [90, 220]], [[100, 200], [120, 200], [90, 220]])
        ledger, _ = crypto_backtest.run_daily_backtest(prices, toy_targets([[0.5, 0.5]], [0]), SYMBOLS, fee_bps=0, other_cost_bps=0)
        np.testing.assert_allclose(ledger.equity_open, [1000, 1050, 1000])
        np.testing.assert_allclose(ledger.equity_close, [1000, 1100, 1000])
        np.testing.assert_allclose(ledger.net_return, [0, 0.1, 1000 / 1100 - 1])
        np.testing.assert_allclose(ledger["units_BTC-GBP"], 5)
        np.testing.assert_allclose(ledger["units_ETH-GBP"], 2.5)

    def test_turnover_counts_buys_and_sells_not_their_net(self):
        prices = toy_prices([[100, 100], [200, 100], [200, 100]], [[100, 100], [200, 100], [200, 100]])
        targets = toy_targets([[0.5, 0.5], [0.5, 0.5]], [0, 1])
        ledger, trades = crypto_backtest.run_daily_backtest(prices, targets, SYMBOLS, fee_bps=0, other_cost_bps=0)
        np.testing.assert_allclose(ledger.iloc[1].traded_notional, 500)
        np.testing.assert_allclose(trades.iloc[2:].signed_notional, [-250, 250])

    def test_mixed_rebalances_match_independent_linear_solution(self):
        rng = np.random.default_rng(724)
        for case in range(100):
            with self.subTest(case=case):
                prices = pd.Series(rng.uniform(1, 10000, 2), index=SYMBOLS)
                units = pd.Series(rng.uniform(0, 10, 2), index=SYMBOLS)
                cash = float(rng.uniform(0, 5000))
                weights = pd.Series(rng.dirichlet(np.ones(3))[:2], index=SYMBOLS)
                rate = float(rng.uniform(0, 0.01))
                expected, delta = reference_rebalance((units * prices).to_numpy(), cash, weights.to_numpy(), rate)
                q, money, trades = crypto_backtest.rebalance_portfolio(units, cash, prices, weights, fee_bps=rate * 10000, other_cost_bps=0)
                np.testing.assert_allclose(q * prices, weights * expected, rtol=1e-11, atol=1e-8)
                np.testing.assert_allclose(money, (1 - weights.sum()) * expected, rtol=1e-11, atol=1e-8)
                np.testing.assert_allclose(trades.signed_notional, delta, rtol=1e-11, atol=1e-8)
                np.testing.assert_allclose(money + (q * prices).sum() + trades.fee.sum(), cash + (units * prices).sum(), rtol=1e-12)

    def test_invalid_schedules_and_prices_rejected(self):
        prices = toy_prices([[100, 200]] * 3, [[100, 200]] * 3)
        targets = toy_targets([[0.5, 0.5]], [0])
        dates = sorted(prices.timestamp.unique())
        cases = [
            (prices, targets.assign(decision_at=targets.execution_at)),
            (prices, targets.assign(decision_at=pd.NaT)),
            (prices.loc[prices.timestamp != dates[1]], targets),
            (prices.loc[prices.timestamp != dates[0]], targets),
            (prices, targets.assign(GBP_cash=0.1)),
            (prices, pd.concat([targets, targets], ignore_index=True)),
            (prices.assign(open=0), targets),
            (prices.assign(close=np.nan), targets),
        ]
        for case, (p, t) in enumerate(cases):
            with self.subTest(case=case), self.assertRaises(ValueError):
                crypto_backtest.run_daily_backtest(p, t, SYMBOLS)

    def test_input_and_past_rows_unchanged(self):
        prices = toy_prices([[100, 200]] * 3, [[100, 200]] * 3)
        targets = toy_targets([[0.5, 0.5]], [0])
        original_p, original_t = prices.copy(deep=True), targets.copy(deep=True)
        baseline, _ = crypto_backtest.run_daily_backtest(prices, targets, SYMBOLS)
        pd.testing.assert_frame_equal(prices, original_p)
        pd.testing.assert_frame_equal(targets, original_t)
        changed = prices.copy()
        changed.loc[changed.timestamp == changed.timestamp.max(), "close"] *= 3
        revised, _ = crypto_backtest.run_daily_backtest(changed, targets, SYMBOLS)
        pd.testing.assert_frame_equal(baseline.iloc[:-1], revised.iloc[:-1])

    def test_capital_scaling_is_linear_in_fractional_model(self):
        prices = toy_prices([[100, 200]] * 3, [[110, 190], [120, 180], [130, 170]])
        targets = toy_targets([[0.5, 0.5], [0, 0.5]], [0, 2])
        small, _ = crypto_backtest.run_daily_backtest(prices, targets, SYMBOLS, initial_cash=1000)
        large, _ = crypto_backtest.run_daily_backtest(prices, targets, SYMBOLS, initial_cash=10000)
        np.testing.assert_allclose(large.equity_close, 10 * small.equity_close)
        np.testing.assert_allclose(large.net_return, small.net_return)
