"""Walk-forward dates, past-only scoring, paired benchmarks and policy persistence."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rebalance_schedule import build_development_grid
from reversal_walk_forward import (development_walk_forward_rules,
    freeze_development_walk_forward, build_development_folds, select_development_configurations)


class DevelopmentWalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"development_start": "2022-01-01",
                         "development_end_exclusive": "2025-01-01", "holdout_start": "2025-01-01"}
        self.rules = development_walk_forward_rules(self.protocol)
        self.grid = build_development_grid()
        self.days = pd.date_range("2022-01-01", "2025-01-01", tz="UTC", inclusive="left")
        self.strategy = pd.DataFrame(-.002, index=self.days, columns=self.grid.index)
        self.benchmark = pd.DataFrame(0., index=self.days, columns=self.grid.index)
        self.a, self.b = self.grid.index[:2]
        self.strategy.loc[:, self.a] = 0.
        self.strategy.loc[self.days.year == 2022, self.a] = .001
        self.strategy.loc[:, self.b] = 0.
        self.strategy.loc[self.days.year == 2023, self.b] = .004

    def select(self):
        return select_development_configurations(self.strategy, self.benchmark, self.grid,
                                                 self.protocol, self.rules)

    def test_eight_quarters_expanding_training_and_leap_year(self):
        folds = build_development_folds(self.protocol, self.rules)
        self.assertEqual(len(folds), 8)
        self.assertEqual(folds.test_days.tolist(), [90, 91, 92, 92, 91, 91, 92, 92])
        self.assertEqual(folds.training_days.tolist(), [365, 455, 546, 638, 730, 821, 912, 1004])
        self.assertTrue(folds.last_training_valuation_at.eq(folds.selection_at).all())
        self.assertTrue(folds.test_end_exclusive.iloc[:-1].reset_index(drop=True).equals(
            folds.test_start.iloc[1:].reset_index(drop=True)))
        self.assertEqual(folds.test_end_exclusive.iloc[-1], pd.Timestamp("2025-01-01", tz="UTC"))

    def test_known_past_winners_and_training_scores(self):
        choices, scores = self.select()
        self.assertEqual(choices.loc[1, "config_id"], self.a)
        self.assertAlmostEqual(choices.loc[1, "training_mean_net_excess_bps"], 10.)
        self.assertEqual(choices.loc[5, "config_id"], self.b)
        self.assertAlmostEqual(choices.loc[5, "training_mean_net_excess_bps"], 20.)
        self.assertEqual(scores.shape, (8, 60))

    def test_future_mutation_cannot_change_earlier_choices_or_scores(self):
        before, scores = self.select()
        cutoff = pd.Timestamp("2023-07-01", tz="UTC")
        self.strategy.loc[self.days >= cutoff, :] = .9
        self.benchmark.loc[self.days >= cutoff, :] = -.5
        after, new_scores = self.select()
        pd.testing.assert_frame_equal(before.loc[:3], after.loc[:3])
        pd.testing.assert_frame_equal(scores.loc[:3], new_scores.loc[:3])

    def test_decision_day_return_is_excluded_but_previous_day_is_included(self):
        self.strategy.loc[:, :] = 0.
        self.strategy.loc[pd.Timestamp("2023-01-01", tz="UTC"), self.b] = 100.
        self.strategy.loc[pd.Timestamp("2022-12-31", tz="UTC"), self.a] = .1
        choices, _ = self.select()
        self.assertEqual(choices.loc[1, "config_id"], self.a)
        self.assertAlmostEqual(choices.loc[1, "training_mean_net_excess_bps"], .1 / 365 * 10000)

    def test_selection_uses_matched_excess_not_raw_strategy_return(self):
        self.strategy.loc[:, self.a] = .001
        self.strategy.loc[:, self.b] = .002
        self.benchmark.loc[:, self.b] = .0019
        choices, _ = self.select()
        self.assertTrue(choices.config_id.eq(self.a).all())

    def test_exact_tie_is_independent_of_column_order_and_negative_score_is_retained(self):
        self.strategy.loc[:, :] = -.001
        self.strategy = self.strategy[self.strategy.columns[::-1]]
        choices, _ = self.select()
        self.assertTrue(choices.config_id.eq(min(self.grid.index)).all())
        self.assertTrue(choices.positive_training_mean.eq(False).all())
        self.assertFalse(choices.config_id.isna().any())

    def test_first_decision_preserves_seven_day_anchor(self):
        winner = "return_90d_bottom_30pct_rebalance_7d"
        self.strategy[winner] = .01
        choices, _ = self.select()
        self.assertEqual(choices.loc[1, "first_scheduled_decision_at"], pd.Timestamp("2023-01-07", tz="UTC"))
        offsets = (choices.first_scheduled_decision_at - pd.Timestamp("2022-01-01", tz="UTC")).dt.days
        self.assertTrue(offsets.mod(7).eq(0).all())
        self.assertTrue(choices.first_scheduled_decision_at.ge(choices.selection_at).all())

    def test_missing_day_missing_config_and_nonfinite_return_fail(self):
        self.strategy = self.strategy.iloc[1:]
        with self.assertRaisesRegex(ValueError, "all development days"):
            self.select()
        self.setUp()
        self.benchmark = self.benchmark.drop(columns=self.b)
        with self.assertRaisesRegex(ValueError, "all 60"):
            self.select()
        self.setUp()
        self.strategy.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.select()

    def test_holdout_rows_are_rejected(self):
        self.strategy.loc[pd.Timestamp("2025-01-01", tz="UTC")] = 0.
        with self.assertRaisesRegex(ValueError, "all development days"):
            self.select()

    def test_saved_policy_is_repeatable_and_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            first = freeze_development_walk_forward(directory, self.protocol)
            path = Path(directory) / "research/development_walk_forward_v1.json"
            before = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(first, freeze_development_walk_forward(directory, self.protocol))
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "Existing walk-forward policy differs"):
                freeze_development_walk_forward(directory, self.protocol)
            self.assertEqual(path.read_text(), '{}')


if __name__ == "__main__":
    unittest.main()
