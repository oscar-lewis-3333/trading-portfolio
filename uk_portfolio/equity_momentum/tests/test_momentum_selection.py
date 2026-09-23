"""Offline examples for signal timing, data gaps and monthly stock selection."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import momentum_backtest as backtest
import momentum_features as features


def price_fixture():
    dates = pd.bdate_range("2020-01-02", periods=10, name="Date")
    index = pd.MultiIndex.from_product(
        [dates, ["A.L", "B.L"]], names=["Date", "ticker"]
    )
    frame = pd.DataFrame(index=index)
    # Distinct histories expose accidental shifts across stocks.
    frame["adj_close"] = np.column_stack([
        [10., 12., 15., 18., 20., 24., 30., 40., 50., 60.],
        [100., 95., 90., 85., 80., 75., 70., 65., 60., 55.],
    ]).ravel()
    frame["Volume"] = 10.
    frame["traded_value_gbp"] = 100.
    frame["observed_row"] = True
    frame["close_reference_usable"] = True
    return frame, pd.DataFrame(index=dates)


def selection_fixture():
    dates = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-28"])
    index = pd.MultiIndex.from_product(
        [dates, ["A.L", "B.L", "C.L", "D.L", "E.L"]],
        names=["Date", "ticker"],
    )
    frame = pd.DataFrame({"eligible": True, "raw_momentum_score": 0.}, index=index)
    # All eligible scores are negative: relative momentum must still invest.
    frame.loc[dates[0], "raw_momentum_score"] = [-.2, -.1, -.1, -.3, 10.]
    frame.loc[(dates[0], "E.L"), "eligible"] = False
    frame.loc[dates[2], "eligible"] = False
    calendar = pd.DataFrame(
        {"execution_date": pd.to_datetime(["2020-02-03", "2020-03-02"])},
        index=pd.DatetimeIndex(dates[[0, 2]], name="signal_date"),
    )
    return frame, calendar


class FeatureTimingTests(unittest.TestCase):
    def test_momentum_endpoints_and_warmup_are_per_stock(self):
        prices, schedule = price_fixture()
        out = features.build_raw_momentum_features(prices, schedule, 5, 2)
        dates = schedule.index
        self.assertTrue(out.loc[:dates[4], "raw_momentum_score"].isna().all())
        self.assertAlmostEqual(out.loc[(dates[5], "A.L"), "raw_momentum_score"], .8)
        self.assertAlmostEqual(out.loc[(dates[5], "B.L"), "raw_momentum_score"], -.15)
        self.assertEqual(out.loc[(dates[5], "A.L"), "formation_start_price"], 10.)
        self.assertEqual(out.loc[(dates[5], "A.L"), "formation_end_price"], 18.)
        zero_skip = features.build_raw_momentum_features(prices, schedule, 5, 0)
        self.assertAlmostEqual(zero_skip.loc[(dates[5], "A.L"), "raw_momentum_score"], 1.4)

    def test_internal_gap_blocks_score_until_gap_leaves_formation(self):
        prices, schedule = price_fixture()
        dates = schedule.index
        prices.loc[(dates[1], "A.L"), "adj_close"] = np.nan
        out = features.build_raw_momentum_features(prices, schedule, 5, 2)
        # Both endpoints exist, but the interior price is missing.
        self.assertTrue(pd.isna(out.loc[(dates[5], "A.L"), "raw_momentum_score"]))
        self.assertFalse(out.loc[(dates[6], "A.L"), "formation_history_complete"])
        self.assertTrue(out.loc[(dates[7], "A.L"), "formation_history_complete"])
        self.assertAlmostEqual(out.loc[(dates[5], "B.L"), "raw_momentum_score"], -.15)

    def test_skipped_month_and_future_prices_do_not_enter_score(self):
        prices, schedule = price_fixture()
        dates = schedule.index
        expected = features.build_raw_momentum_features(prices, schedule, 5, 2)
        changed = prices.copy()
        changed.loc[dates[4]:, "adj_close"] *= 1_000.
        actual = features.build_raw_momentum_features(changed, schedule, 5, 2)
        pd.testing.assert_frame_equal(actual.loc[:dates[5]], expected.loc[:dates[5]])
        prefix = features.build_raw_momentum_features(
            prices.loc[:dates[6]], schedule.loc[:dates[6]], 5, 2
        )
        pd.testing.assert_frame_equal(prefix, expected.loc[:dates[6]])

    def test_liquidity_distinguishes_zero_from_missing_volume(self):
        prices, schedule = price_fixture()
        dates = schedule.index
        prices.loc[(dates[1], "A.L"), ["Volume", "traded_value_gbp"]] = 0.
        out = features.build_liquidity_features(prices, schedule, 3)
        self.assertFalse(out.loc[(dates[1], "A.L"), "liquidity_history_complete"])
        self.assertAlmostEqual(out.loc[(dates[2], "A.L"), "positive_volume_fraction"], 2 / 3)
        self.assertEqual(out.loc[(dates[2], "A.L"), "median_traded_value_gbp"], 100.)
        self.assertTrue(out.loc[(dates[2], "A.L"), "liquidity_history_complete"])
        for column, value in [("Volume", np.nan), ("Volume", -1.),
                              ("traded_value_gbp", np.inf),
                              ("observed_row", False), ("close_reference_usable", False)]:
            with self.subTest(column=column, value=value):
                changed = prices.copy()
                changed.loc[(dates[1], "A.L"), column] = value
                actual = features.build_liquidity_features(changed, schedule, 3)
                self.assertFalse(actual.loc[(dates[2], "A.L"), "liquidity_history_complete"])
                self.assertTrue(actual.loc[(dates[4], "A.L"), "liquidity_history_complete"])
                pd.testing.assert_frame_equal(actual.xs("B.L", level="ticker"),
                                              out.xs("B.L", level="ticker"))

    def test_liquidity_uses_current_close_but_never_future_rows(self):
        prices, schedule = price_fixture()
        cutoff = schedule.index[5]
        expected = features.build_liquidity_features(prices, schedule, 3)
        changed = prices.copy()
        changed.loc[schedule.index[6]:, "traded_value_gbp"] = 1e9
        actual = features.build_liquidity_features(changed, schedule, 3)
        pd.testing.assert_frame_equal(actual.loc[:cutoff], expected.loc[:cutoff])
        prefix = features.build_liquidity_features(prices.loc[:cutoff], schedule.loc[:cutoff], 3)
        pd.testing.assert_frame_equal(prefix, expected.loc[:cutoff])

    def test_missing_calendar_rows_are_rejected_instead_of_compressing_time(self):
        prices, schedule = price_fixture()
        broken = prices.drop(index=(schedule.index[1], "A.L"))
        for builder in [features.build_liquidity_features, features.build_raw_momentum_features]:
            with self.subTest(builder=builder.__name__):
                with self.assertRaisesRegex(ValueError, "every scheduled session"):
                    builder(broken, schedule)


class MonthlySelectionTests(unittest.TestCase):
    def test_calendar_uses_next_session_and_does_not_complete_partial_month(self):
        dates = pd.DatetimeIndex(["2015-12-30", "2015-12-31", "2016-01-04",
                                  "2016-01-28", "2016-01-29", "2016-02-01",
                                  "2016-02-02"], name="Date")
        actual = backtest.build_monthly_rebalance_calendar(pd.DataFrame(index=dates))
        expected = pd.DataFrame(
            {"execution_date": pd.to_datetime(["2016-01-04", "2016-02-01"])},
            index=pd.DatetimeIndex(["2015-12-31", "2016-01-29"], name="signal_date"),
        )
        pd.testing.assert_frame_equal(actual, expected)
        self.assertTrue(backtest.build_monthly_rebalance_calendar(
            pd.DataFrame(index=dates[:1])).empty)

    def test_ties_use_ticker_and_negative_relative_winners_still_invest(self):
        frame, calendar = selection_fixture()
        targets, summary = backtest.build_momentum_targets(frame, calendar, top_frac=.2)
        first = targets.iloc[0]
        self.assertEqual(first[first.gt(0)].to_dict(), {"B.L": 1.})
        self.assertEqual(summary.iloc[0]["eligible_stocks"], 4)
        shuffled = frame.sample(frac=1, random_state=41)
        pd.testing.assert_frame_equal(
            backtest.build_momentum_targets(shuffled, calendar, .2)[0], targets)

    def test_rounding_equal_weights_and_empty_period_cash_reconcile(self):
        frame, calendar = selection_fixture()
        targets, summary = backtest.build_momentum_targets(frame, calendar, top_frac=.3)
        self.assertEqual(targets.iloc[0][targets.iloc[0].gt(0)].to_dict(),
                         {"B.L": .5, "C.L": .5})
        self.assertEqual(summary["selected_stocks"].tolist(), [2, 0])
        self.assertEqual(summary["target_cash_weight"].tolist(), [0., 1.])
        np.testing.assert_allclose(targets.sum(axis=1) + summary["target_cash_weight"], 1.)
        self.assertTrue(targets.iloc[1].eq(0).all())

    def test_execution_day_information_cannot_change_earlier_target(self):
        frame, calendar = selection_fixture()
        expected = backtest.build_momentum_targets(frame, calendar)[0]
        execution = calendar.iloc[0]["execution_date"]
        changed = frame.copy()
        changed.loc[execution, "raw_momentum_score"] = [100., -100., -100., 0., 200.]
        changed.loc[execution, "eligible"] = False
        actual = backtest.build_momentum_targets(changed, calendar)[0]
        pd.testing.assert_frame_equal(actual, expected)

    def test_invalid_eligible_score_fails_but_ineligible_nan_is_allowed(self):
        frame, calendar = selection_fixture()
        first = calendar.index[0]
        frame.loc[(first, "E.L"), "raw_momentum_score"] = np.nan
        backtest.build_momentum_targets(frame, calendar)
        frame.loc[(first, "A.L"), "raw_momentum_score"] = np.nan
        with self.assertRaisesRegex(ValueError, "Eligible stocks have invalid scores"):
            backtest.build_momentum_targets(frame, calendar)

    def test_same_day_execution_and_duplicate_feature_rows_are_rejected(self):
        frame, calendar = selection_fixture()
        bad_calendar = calendar.copy()
        bad_calendar.iloc[0, 0] = bad_calendar.index[0]
        with self.assertRaisesRegex(ValueError, "Invalid signal/execution calendar"):
            backtest.build_momentum_targets(frame, bad_calendar)
        with self.assertRaisesRegex(ValueError, "unique"):
            backtest.build_momentum_targets(pd.concat([frame, frame.iloc[:1]]), calendar)


if __name__ == "__main__":
    unittest.main()
