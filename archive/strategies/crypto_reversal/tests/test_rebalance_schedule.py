"""Calendar-only tests: no prices, outcomes, or execution assumptions."""
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rebalance_schedule import build_development_grid, build_development_schedules


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "development_start": "2022-01-01",
            "development_end_exclusive": "2025-01-01",
            "holdout_start": "2025-01-01",
            "universe": {"min_assets": 20, "max_assets": 30},
        }
        dates = pd.date_range("2022-01-01", "2024-12-31", tz="UTC", name="decision_at")
        self.coverage = pd.DataFrame({"active": True, "universe_size": 20}, index=dates)

    def test_expected_counts_boundaries_and_spacing(self):
        schedules, audit = build_development_schedules(self.coverage, self.protocol)
        self.assertEqual(audit["scheduled_decisions"].to_dict(), {1: 1096, 2: 548, 3: 366, 7: 157})
        last_dates = {1: "2024-12-31", 2: "2024-12-30", 3: "2024-12-31", 7: "2024-12-28"}
        for interval, group in schedules.groupby("rebalance_days"):
            self.assertEqual(group.iloc[0]["decision_at"], pd.Timestamp("2022-01-01", tz="UTC"))
            self.assertEqual(group.iloc[-1]["decision_at"], pd.Timestamp(last_dates[interval], tz="UTC"))
            self.assertTrue(group["decision_at"].diff().dropna().eq(pd.Timedelta(days=interval)).all())
            self.assertEqual(group["rebalance_number"].tolist(), list(range(len(group))))
            self.assertTrue((group["decision_at"] - group["signal_candle_start"]).eq(pd.Timedelta(days=1)).all())
        self.assertTrue(schedules["decision_at"].lt(pd.Timestamp("2025-01-01", tz="UTC")).all())

    def test_full_cartesian_grid_has_all_60_configurations(self):
        grid = build_development_grid()
        expected = {
            (lookback, fraction, interval)
            for lookback in [1, 5, 10, 30, 90]
            for fraction in [.1, .2, .3]
            for interval in [1, 2, 3, 7]
        }
        self.assertEqual(len(grid), 60)
        self.assertTrue(grid.index.is_unique)
        self.assertEqual(set(grid.itertuples(index=False, name=None)), expected)
        self.assertEqual(grid.groupby("rebalance_days").size().to_dict(), {1: 15, 2: 15, 3: 15, 7: 15})
        pd.testing.assert_frame_equal(grid, build_development_grid())

    def test_grid_is_independent_and_every_configuration_has_a_schedule(self):
        grid = build_development_grid()
        schedules, audit = build_development_schedules(self.coverage, self.protocol)
        self.assertTrue(grid["rebalance_days"].isin(audit.index).all())
        for interval in grid["rebalance_days"].unique():
            self.assertTrue(schedules["rebalance_days"].eq(interval).any())
        grid.iloc[0, grid.columns.get_loc("lookback_days")] = 999
        self.assertEqual(build_development_grid().iloc[0]["lookback_days"], 1)

    def test_inactive_date_is_retained_and_does_not_shift_next_decision(self):
        self.coverage.loc["2022-01-03"] = [False, 0]
        schedules, audit = build_development_schedules(self.coverage, self.protocol)
        two_day = schedules.loc[schedules["rebalance_days"].eq(2)].set_index("decision_at")
        self.assertEqual(two_day.loc["2022-01-03", "status"], "insufficient_breadth")
        self.assertEqual(two_day.index[2], pd.Timestamp("2022-01-05", tz="UTC"))
        self.assertEqual(audit.loc[2, "inactive_decisions"], 1)

    def test_leap_day_and_weekends_remain_calendar_days(self):
        protocol = deepcopy(self.protocol)
        protocol.update(development_start="2024-02-27", development_end_exclusive="2024-03-05")
        coverage = self.coverage.loc["2024-02-27":"2024-03-04"]
        schedules, _ = build_development_schedules(coverage, protocol, intervals=(2,))
        expected = pd.to_datetime(["2024-02-27", "2024-02-29", "2024-03-02", "2024-03-04"], utc=True)
        self.assertEqual(schedules["decision_at"].tolist(), expected.tolist())

    def test_missing_duplicate_or_unsorted_calendar_is_rejected(self):
        bad_cases = [self.coverage.drop(self.coverage.index[4]), self.coverage.iloc[::-1],
                     pd.concat([self.coverage, self.coverage.iloc[:1]])]
        for coverage in bad_cases:
            with self.subTest(rows=len(coverage)):
                with self.assertRaisesRegex(ValueError, "every development day"):
                    build_development_schedules(coverage, self.protocol)

    def test_invalid_grid_and_holdout_overlap_are_rejected(self):
        for intervals in [(), (0,), (-1,), (True,), (1.5,), (1, 1)]:
            with self.subTest(intervals=intervals):
                with self.assertRaisesRegex(ValueError, "distinct positive"):
                    build_development_schedules(self.coverage, self.protocol, intervals)
        bad = dict(self.protocol, holdout_start="2024-12-31")
        with self.assertRaisesRegex(ValueError, "holdout starts"):
            build_development_schedules(self.coverage, bad)

    def test_inconsistent_universe_is_rejected(self):
        bad = self.coverage.copy()
        bad.iloc[0, bad.columns.get_loc("universe_size")] = 19
        with self.assertRaisesRegex(ValueError, "activity rules"):
            build_development_schedules(bad, self.protocol)
        bad = self.coverage.copy()
        bad.iloc[0, bad.columns.get_loc("active")] = False
        with self.assertRaisesRegex(ValueError, "activity rules"):
            build_development_schedules(bad, self.protocol)

    def test_inputs_unchanged_and_activity_cannot_move_calendar(self):
        original = self.coverage.copy(deep=True)
        protocol_before = deepcopy(self.protocol)
        baseline, _ = build_development_schedules(self.coverage, self.protocol)
        pd.testing.assert_frame_equal(self.coverage, original)
        self.assertEqual(self.protocol, protocol_before)
        changed = self.coverage.copy()
        changed.loc["2023-01-01":] = [False, 0]
        other, _ = build_development_schedules(changed, self.protocol)
        pd.testing.assert_frame_equal(
            baseline[["rebalance_days", "decision_at", "rebalance_number"]],
            other[["rebalance_days", "decision_at", "rebalance_number"]],
        )


if __name__ == "__main__":
    unittest.main()
