"""Anchored target switching, cash initialization and departing-asset coverage."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rebalance_schedule import build_development_grid
from reversal_walk_forward import development_walk_forward_rules, build_development_folds
from walk_forward_targets import build_walk_forward_targets, audit_walk_forward_references


class WalkForwardTargetTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"development_start": "2022-01-01", "development_end_exclusive": "2025-01-01",
                         "holdout_start": "2025-01-01"}
        self.rules = development_walk_forward_rules(self.protocol)
        self.grid = build_development_grid()
        self.choices = build_development_folds(self.protocol, self.rules)
        a = "return_90d_bottom_20pct_rebalance_1d"
        b = "return_90d_bottom_30pct_rebalance_7d"
        c = "return_90d_bottom_30pct_rebalance_1d"
        self.set_choices([a, a, b, b, c, b, b, b])
        self.targets = {}
        for config_id, config in self.grid.iterrows():
            dates = pd.date_range("2022-01-01", "2025-01-01", freq=f"{int(config.rebalance_days)}D", tz="UTC", inclusive="left")
            self.targets[config_id] = {
                "strategy": pd.DataFrame({"A": 1., "B": 0., "CASH": 0.}, index=dates),
                "benchmark": pd.DataFrame({"A": .5, "B": .5, "CASH": 0.}, index=dates)}

    def set_choices(self, ids):
        self.choices["config_id"] = ids
        for column in ("lookback_days", "top_fraction", "rebalance_days"):
            self.choices[column] = [self.grid.loc[i, column] for i in ids]
        anchor = pd.Timestamp("2022-01-01", tz="UTC")
        self.choices["first_scheduled_decision_at"] = [
            r.selection_at + pd.Timedelta(days=(-(r.selection_at - anchor).days) % int(r.rebalance_days))
            for r in self.choices.itertuples()]

    def build(self):
        return build_walk_forward_targets(self.choices, self.targets, self.protocol, self.rules)

    def test_expected_events_and_no_boundary_resets(self):
        strategy, benchmark, events = self.build()
        self.assertEqual(len(strategy), 338)
        self.assertTrue(strategy.index.equals(benchmark.index))
        self.assertEqual(events.groupby("fold").size().tolist(), [90, 91, 14, 13, 91, 13, 13, 13])
        self.assertNotIn(pd.Timestamp("2024-04-01", tz="UTC"), events.index)
        before = events.loc[:"2024-04-06"].tail(2).index
        self.assertEqual(before.tolist(), [pd.Timestamp("2024-03-31", tz="UTC"), pd.Timestamp("2024-04-06", tz="UTC")])
        self.assertEqual(events.configuration_changed.sum(), 4)
        self.assertFalse(events.loc[pd.Timestamp("2023-10-07", tz="UTC"), "configuration_changed"])

    def test_initial_wait_is_cash_without_moving_weekly_anchor(self):
        ids = self.choices.config_id.tolist()
        ids[0] = "return_90d_bottom_30pct_rebalance_7d"
        self.set_choices(ids)
        strategy, benchmark, events = self.build()
        self.assertEqual(strategy.index[0], pd.Timestamp("2023-01-01", tz="UTC"))
        self.assertEqual(strategy.iloc[0].CASH, 1.)
        self.assertEqual(benchmark.iloc[0].CASH, 1.)
        self.assertEqual(strategy.index[1], pd.Timestamp("2023-01-07", tz="UTC"))
        self.assertEqual(events.event_kind.iloc[0], "initial_cash")

    def test_cash_target_kept_and_inputs_unchanged(self):
        key = self.choices.config_id.iloc[0]
        date = pd.Timestamp("2023-02-01", tz="UTC")
        for role in ("strategy", "benchmark"):
            self.targets[key][role].loc[date] = [0., 0., 1.]
        before = self.targets[key]["strategy"].copy(deep=True)
        strategy, benchmark, events = self.build()
        self.assertEqual(strategy.loc[date, "CASH"], 1.)
        self.assertEqual(benchmark.loc[date, "CASH"], 1.)
        self.assertEqual(events.loc[date, "strategy_assets"], 0)
        pd.testing.assert_frame_equal(before, self.targets[key]["strategy"])

    def test_wrong_fold_or_first_decision_is_rejected(self):
        self.choices.loc[1, "training_days"] = 999
        with self.assertRaisesRegex(ValueError, "complete quarterly"):
            self.build()
        self.setUp()
        self.choices.loc[6, "first_scheduled_decision_at"] = self.choices.loc[6, "selection_at"]
        with self.assertRaisesRegex(ValueError, "anchored calendar"):
            self.build()

    def test_missing_candidate_date_rejected(self):
        key = self.choices.config_id.iloc[0]
        self.targets[key]["strategy"] = self.targets[key]["strategy"].iloc[1:]
        with self.assertRaisesRegex(ValueError, "omit or add"):
            self.build()


class WalkForwardCoverageTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.to_datetime(["2024-03-31", "2024-04-06"], utc=True)
        self.strategy = pd.DataFrame({"OLD": [1., 0.], "NEW": [0., 1.], "CASH": [0., 0.]}, index=self.days)
        self.benchmark = self.strategy.copy()
        self.refs = pd.DataFrame([self.ref("OLD", self.days[0]), self.ref("NEW", self.days[1])])

    def ref(self, product, day):
        return {"product_id": product, "day": day, "execution_at": day + pd.Timedelta(minutes=5),
                "reference_bar_start": day + pd.Timedelta(minutes=4),
                "assumed_available_at": day + pd.Timedelta(minutes=5), "reference_price_gbp": 100.}

    def test_previous_event_exit_is_required_even_after_calendar_switch(self):
        coverage, summary = audit_walk_forward_references(self.strategy, self.benchmark, self.refs)
        self.assertEqual(summary.required_references, 3)
        self.assertEqual(summary.missing_references, 1)
        missing = coverage.loc[coverage.status.eq("missing_reference")].iloc[0]
        self.assertEqual(missing.product_id, "OLD")
        self.assertEqual(missing.day, self.days[1])
        self.assertTrue(missing.exit_only)

    def test_complete_exit_coverage_and_stale_mark_detection(self):
        self.refs = pd.concat([self.refs, pd.DataFrame([self.ref("OLD", self.days[1])])], ignore_index=True)
        _, summary = audit_walk_forward_references(self.strategy, self.benchmark, self.refs)
        self.assertEqual(summary.ready_references, 3)
        self.refs.loc[2, "reference_bar_start"] = self.days[1] - pd.Timedelta(minutes=1)
        self.refs.loc[2, "assumed_available_at"] = self.days[1]
        _, summary = audit_walk_forward_references(self.strategy, self.benchmark, self.refs)
        self.assertEqual(summary.invalid_references, 1)

    def test_all_cash_has_no_reference_requirements(self):
        self.strategy.loc[:, ["OLD", "NEW"]] = 0.
        self.strategy["CASH"] = 1.
        coverage, summary = audit_walk_forward_references(self.strategy, self.strategy.copy(), self.refs)
        self.assertTrue(coverage.empty)
        self.assertEqual(summary.required_references, 0)

    def test_benchmark_only_holdings_are_included(self):
        self.benchmark["EXTRA"] = 0.
        self.benchmark.loc[self.days[0], ["OLD", "EXTRA"]] = [.5, .5]
        coverage, summary = audit_walk_forward_references(self.strategy, self.benchmark, self.refs)
        extra = coverage.loc[coverage.product_id.eq("EXTRA")]
        self.assertEqual(len(extra), 2)
        self.assertTrue(extra.exit_only.iloc[1])


if __name__ == "__main__":
    unittest.main()
