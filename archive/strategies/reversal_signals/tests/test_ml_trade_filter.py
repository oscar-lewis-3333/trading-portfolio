#timing and exposure contracts shared by future ML studies
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ml_trade_filter as ml
from reversal_ml_data import build_context_features


class ObservationTimingTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2020-01-02", tz="UTC")
        self.end = pd.Timestamp("2020-01-04", tz="UTC")
        self.data = pd.DataFrame({
            "signal_time": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"], utc=True),
            "exit_time": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-05", "2020-01-04"], utc=True),
            "features_ready": [True, True, True, False],
            "outcome_observed": [True, True, False, True],
        })

    def test_training_excludes_future_labels_and_respects_exact_cutoff(self):
        annual = ml.observable_split(self.data, self.start, self.end)
        continuous = ml.observable_split(self.data, self.start, self.end, include_exit_at_start=True)
        self.assertEqual(annual["train_mask"].tolist(), [False] * 4)
        self.assertEqual(continuous["train_mask"].tolist(), [True, False, False, False])

    def test_missing_outcomes_and_features_do_not_remove_candidate_trades(self):
        split = ml.observable_split(self.data, self.start, self.end)
        self.assertEqual(split["candidate_mask"].tolist(), [False, False, True, True])
        changed = self.data.assign(outcome_observed=False)
        other = ml.observable_split(changed, self.start, self.end)
        pd.testing.assert_series_equal(split["candidate_mask"], other["candidate_mask"])

    def test_annual_resets_exclude_cross_year_exits_but_continuous_refits_do_not(self):
        annual = ml.observable_split(self.data, self.start, self.end, require_exit_before_end=True)
        continuous = ml.observable_split(self.data, self.start, self.end)
        self.assertEqual(int(annual["candidate_mask"].sum()), 0)
        self.assertEqual(int(continuous["candidate_mask"].sum()), 2)


class AllocationTests(unittest.TestCase):
    def setUp(self):
        index = pd.MultiIndex.from_product([[0], pd.to_datetime(["2020-01-01", "2020-01-02"]), ["A", "B", "C"]],
                                           names=["configuration_id", "Date", "ticker"])
        self.base = pd.Series(.05, index=index)
        self.forecast = pd.Series([.02, -.01, np.nan, -.01, 0., -.03], index=index)
        self.ready = pd.Series([True, True, False, True, True, True], index=index)

    def test_rejected_slots_stay_cash_missing_features_retain_the_base_trade(self):
        filtered = ml.filter_weights(self.base, self.forecast, self.ready)
        self.assertEqual(filtered.tolist(), [.05, 0., .05, 0., 0., 0.])
        self.assertEqual(self.base.tolist(), [.05] * 6)

    def test_control_matches_each_date_including_complete_abstention(self):
        filtered = ml.filter_weights(self.base, self.forecast, self.ready)
        control = ml.exposure_matched_control(self.base, filtered, group_levels=["configuration_id", "Date"])
        np.testing.assert_allclose(control.iloc[:3], [.1 / 3] * 3)
        np.testing.assert_array_equal(control.iloc[3:], [0., 0., 0.])

    def test_missing_forecast_for_ready_trade_is_an_error(self):
        self.ready.iloc[2] = True
        with self.assertRaisesRegex(ValueError, "Missing forecast"):
            ml.filter_weights(self.base, self.forecast, self.ready)


class CausalFeatureAndModelTests(unittest.TestCase):
    def test_future_prices_do_not_change_prior_features_and_gaps_are_preserved(self):
        dates = pd.bdate_range("2020-01-01", periods=50, name="Date")
        index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["Date", "ticker"])
        prices = pd.DataFrame({"adj_close": np.linspace(100, 130, len(index)), "Volume": 100.}, index=index)
        prices.loc[(dates[25], "A"), "adj_close"] = np.nan
        eligible = pd.Series(True, index=index)
        original = build_context_features(prices, eligible)
        altered = prices.copy()
        altered.loc[altered.index.get_level_values("Date") > dates[35], "adj_close"] *= 10
        changed = build_context_features(altered, eligible)
        pd.testing.assert_frame_equal(original.loc[:dates[35]], changed.loc[:dates[35]])
        self.assertTrue(np.isnan(original.loc[(dates[25], "A"), "return_1"]))
        self.assertTrue(np.isnan(original.loc[(dates[26], "A"), "return_1"]))

    def test_prediction_outcomes_cannot_affect_fitting(self):
        training = pd.DataFrame({"feature": np.linspace(-1, 1, 100),
                                 "forward_return": np.linspace(-.1, .1, 100)})
        candidates = pd.DataFrame({"feature": [0., .5, np.nan], "features_ready": [True, True, False],
                                   "forward_return": [100., -100., 100.]})
        for family in ("ridge", "xgb"):
            with self.subTest(family=family):
                first, audit = ml.fit_predict(training, candidates, ["feature"], family=family)
                second, _ = ml.fit_predict(training, candidates.assign(forward_return=0), ["feature"], family=family)
                pd.testing.assert_series_equal(first, second, check_exact=True)
                self.assertTrue(np.isnan(first.iloc[2]))
                self.assertEqual(audit["prediction_rows"], 2)


if __name__ == "__main__":
    unittest.main()
