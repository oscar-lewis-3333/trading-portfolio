import pandas as pd

from _support import (OfflineTestCase, SYMBOLS, DEVELOPMENT_END, development_prices,
                      crypto_features, crypto_portfolio, reference_targets)


class SignalTests(OfflineTestCase):
    def test_momentum_matches_calendar_ratio(self):
        prices = development_prices()
        before = prices.copy(deep=True)
        features = crypto_features.add_momentum_features(prices, 90)
        lookup = prices.set_index(["symbol", "timestamp"])["close"]
        for symbol, group in features.groupby("symbol"):
            self.assertEqual(group.momentum_return.isna().sum(), 90)
            self.assertTrue(group.trend_positive.iloc[:90].isna().all())
            valid = group.loc[group.momentum_return.notna()]
            self.assertEqual(valid.timestamp.iloc[0], pd.Timestamp("2019-04-01", tz="UTC"))
            for row in valid.itertuples():
                expected = lookup.loc[(symbol, row.timestamp)] / lookup.loc[(symbol, row.timestamp - pd.Timedelta(days=90))] - 1
                self.assertAlmostEqual(row.momentum_return, expected, places=12)
                self.assertEqual(row.signal_available_at, row.timestamp + pd.Timedelta(days=1))
                self.assertEqual(row.trend_positive, expected > 0)
        pd.testing.assert_frame_equal(prices, before)

    def test_weekly_targets_match_independent_dates(self):
        prices = development_prices()
        features = crypto_features.add_momentum_features(prices)
        before = features.copy(deep=True)
        actual = crypto_portfolio.build_weekly_targets(features, SYMBOLS, DEVELOPMENT_END)
        pd.testing.assert_frame_equal(actual, reference_targets(prices))
        pd.testing.assert_frame_equal(features, before)
        self.assertEqual(len(actual), 247)
        self.assertTrue(actual.decision_at.dt.dayofweek.eq(0).all())
        self.assertTrue(actual.execution_at.dt.dayofweek.eq(1).all())
        self.assertTrue(actual.execution_at.lt(DEVELOPMENT_END).all())

    def test_benchmark_schedules_are_unconditional(self):
        prices = development_prices()
        signals = reference_targets(prices)
        for mode in ("buy_and_hold", "weekly"):
            actual = crypto_portfolio.build_benchmark_targets(signals, SYMBOLS, mode)
            pd.testing.assert_frame_equal(actual, reference_targets(prices, mode))
        altered = signals.copy()
        altered[SYMBOLS] = 0.0
        altered["GBP_cash"] = 1.0
        for mode in ("buy_and_hold", "weekly"):
            pd.testing.assert_frame_equal(
                crypto_portfolio.build_benchmark_targets(signals, SYMBOLS, mode),
                crypto_portfolio.build_benchmark_targets(altered, SYMBOLS, mode),
            )

    def test_future_prices_do_not_change_past_targets(self):
        prices = development_prices()
        cutoff = pd.Timestamp("2022-01-01", tz="UTC")
        original = crypto_features.add_momentum_features(prices)
        changed = prices.copy(deep=True)
        changed.loc[changed.timestamp >= cutoff, "close"] *= 0.01
        revised = crypto_features.add_momentum_features(changed)
        a = crypto_portfolio.build_weekly_targets(original, SYMBOLS, DEVELOPMENT_END)
        b = crypto_portfolio.build_weekly_targets(revised, SYMBOLS, DEVELOPMENT_END)
        pd.testing.assert_frame_equal(a.loc[a.decision_at <= cutoff], b.loc[b.decision_at <= cutoff])

    def test_shuffled_input_is_equivalent(self):
        prices = development_prices()
        pd.testing.assert_frame_equal(
            crypto_features.add_momentum_features(prices),
            crypto_features.add_momentum_features(prices.sample(frac=1, random_state=7)),
        )

    def test_gap_duplicate_and_invalid_lookback_rejected(self):
        prices = development_prices()
        for changed in (prices.drop(index=100), pd.concat([prices, prices.iloc[[100]]], ignore_index=True)):
            with self.assertRaises(ValueError):
                crypto_features.add_momentum_features(changed)
        for lookback in (True, 0, -1, 1.5):
            with self.subTest(lookback=lookback), self.assertRaises(ValueError):
                crypto_features.add_momentum_features(prices, lookback)
