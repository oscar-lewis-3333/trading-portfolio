"""End-to-end grid coverage, portfolio counts, and empty-universe handling."""
import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reversal_targets
from rebalance_schedule import build_development_grid, build_development_schedules
from reversal_data import build_daily_universe
import test_reversal_targets as target_fixtures


class SweepTargetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = target_fixtures.ReversalTargetTests()
        self.fixture.setUp()
        self.grid = build_development_grid()

    def build(self, grid=None):
        f = self.fixture
        return reversal_targets.build_sweep_targets(
            f.panel, f.members, f.schedules, f.protocol,
            self.grid if grid is None else grid,
        )

    def test_all_sixty_match_hand_calculated_weights_and_calendar_counts(self):
        targets, audit = self.build()
        self.assertEqual(set(targets), set(self.grid.index))
        pd.testing.assert_index_equal(audit.index, self.grid.index)
        self.assertEqual(len(targets), 60)
        for config_id, config in self.grid.iterrows():
            row = audit.loc[config_id]
            expected_days = math.ceil(6 / config.rebalance_days)
            expected_assets = math.ceil(19 * config.top_fraction)
            self.assertEqual(row.scheduled_decisions, expected_days)
            self.assertEqual(row.min_selected_assets, expected_assets)
            self.assertEqual(row.max_selected_assets, expected_assets)
            self.assertEqual(row.cash_decisions, 0)
            strategy, benchmark = targets[config_id]["strategy"], targets[config_id]["benchmark"]
            pd.testing.assert_index_equal(strategy.index, benchmark.index)
            selected = [f"C{i:02d}-USD" for i in range(expected_assets)]
            np.testing.assert_allclose(strategy[selected], 1 / expected_assets)
            self.assertTrue(strategy.drop(columns=selected).eq(0).all().all())
            np.testing.assert_allclose(benchmark.drop(columns="CASH"), 1 / 19)
            self.assertLess(row.max_weight_sum_error, 1e-12)

    def test_zero_eligible_assets_are_cash_for_every_configuration(self):
        f = self.fixture
        protocol = deepcopy(f.protocol)
        protocol["excluded_products"] = f.prices.product_id.unique().tolist()
        panel, members, coverage = build_daily_universe(f.prices, protocol)
        schedules, _ = build_development_schedules(coverage, protocol)
        targets, audit = reversal_targets.build_sweep_targets(panel, members, schedules, protocol, self.grid)
        self.assertTrue(audit.cash_decisions.eq(audit.scheduled_decisions).all())
        self.assertTrue(audit.max_selected_assets.eq(0).all())
        for pair in targets.values():
            self.assertEqual(pair["strategy"].columns.tolist(), ["CASH"])
            self.assertTrue(pair["strategy"].CASH.eq(1).all())
            pd.testing.assert_frame_equal(pair["strategy"], pair["benchmark"])

    def test_duplicate_ids_or_combinations_fail_before_building(self):
        duplicate_ids = pd.concat([self.grid.iloc[:1], self.grid.iloc[:1]])
        duplicate_parameters = self.grid.iloc[[0, 0]].copy()
        duplicate_parameters.index = ["first", "second"]
        cases = [self.grid.iloc[:0], duplicate_ids, duplicate_parameters,
                 self.grid.drop(columns="lookback_days")]
        for grid in cases:
            with self.subTest(rows=len(grid)):
                with patch.object(reversal_targets, "build_configuration_targets") as build_one:
                    with self.assertRaises(ValueError):
                        self.build(grid)
                    build_one.assert_not_called()

    def test_inputs_and_returned_portfolios_are_independent(self):
        before = self.grid.copy(deep=True)
        protocol = deepcopy(self.fixture.protocol)
        targets, _ = self.build(self.grid.iloc[:2])
        pd.testing.assert_frame_equal(before, self.grid)
        self.assertEqual(protocol, self.fixture.protocol)
        first, second = list(targets)
        other = targets[second]["strategy"].copy(deep=True)
        benchmark = targets[first]["benchmark"].copy(deep=True)
        targets[first]["strategy"].iloc[0, 0] = 99.
        pd.testing.assert_frame_equal(targets[second]["strategy"], other)
        pd.testing.assert_frame_equal(targets[first]["benchmark"], benchmark)


if __name__ == "__main__":
    unittest.main()
