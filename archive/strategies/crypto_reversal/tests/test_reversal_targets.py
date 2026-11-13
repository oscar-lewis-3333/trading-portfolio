"""Target weights for small universes, frozen schedules, and causal signals."""
import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversal_data import build_daily_universe
from rebalance_schedule import build_development_grid, build_development_schedules
from reversal_targets import build_configuration_targets


class ReversalTargetTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "development_start": "2022-01-01",
            "development_end_exclusive": "2022-01-07",
            "holdout_start": "2022-01-07",
            "universe": {"min_assets": 1, "max_assets": 30},
            "eligibility": {"history_days": 180, "liquidity_days": 30,
                            "min_median_dollar_volume": 1000000.},
            "excluded_products": [],
            "signal": {"formation_days": 1, "selection_fraction": .2,
                       "negative_return_required": False},
        }
        dates = pd.date_range("2021-06-01", "2022-01-06", tz="UTC")
        self.prices = pd.DataFrame([
            {"product_id": f"C{asset:02d}-USD", "timestamp": day,
             "close": 100. * (1 + .0001 * asset) ** offset,
             "volume": 20000.}
            for asset in range(19)
            for offset, day in enumerate(dates)
        ])
        self.panel, self.members, self.coverage = build_daily_universe(self.prices, self.protocol)
        self.schedules, _ = build_development_schedules(self.coverage, self.protocol)

    def run_config(self, lookback=5, fraction=.2, interval=3):
        configuration = pd.Series({"lookback_days": lookback, "top_fraction": fraction, "rebalance_days": interval})
        return build_configuration_targets(self.panel, self.members, self.schedules, self.protocol, configuration)

    def test_nineteen_assets_select_two_four_or_six_without_cash(self):
        self.assertTrue(self.coverage["active"].all())
        self.assertTrue(self.coverage["universe_size"].eq(19).all())
        for fraction, expected in [(.1, 2), (.2, 4), (.3, 6)]:
            with self.subTest(fraction=fraction):
                strategy, benchmark, rankings, audit = self.run_config(fraction=fraction)
                self.assertTrue(audit["selected_assets"].eq(expected).all())
                self.assertTrue(strategy["CASH"].eq(0).all())
                np.testing.assert_allclose(strategy.sum(axis=1), 1.)
                np.testing.assert_allclose(benchmark.drop(columns="CASH"), 1 / 19)
                held = strategy.columns[strategy.iloc[0].gt(0)].tolist()
                self.assertEqual(held, [f"C{i:02d}-USD" for i in range(expected)])
                np.testing.assert_allclose(strategy[held], 1 / expected)

    def test_one_and_two_asset_universes_stay_invested(self):
        for count in (1, 2):
            prices = self.prices.loc[self.prices.product_id.isin([f"C{i:02d}-USD" for i in range(count)])]
            panel, members, coverage = build_daily_universe(prices, self.protocol)
            schedules, _ = build_development_schedules(coverage, self.protocol)
            strategy, benchmark, _, audit = build_configuration_targets(
                panel, members, schedules, self.protocol,
                {"lookback_days": 1, "top_fraction": .1, "rebalance_days": 1})
            self.assertTrue(audit.selected_assets.eq(1).all())
            self.assertTrue(strategy.CASH.eq(0).all())
            np.testing.assert_allclose(benchmark.sum(axis=1), 1.)

    def test_all_sixty_configurations_have_correct_counts_and_dates(self):
        for _, config in build_development_grid().iterrows():
            strategy, benchmark, _, audit = build_configuration_targets(
                self.panel, self.members, self.schedules, self.protocol, config)
            expected_dates = pd.date_range("2022-01-01", "2022-01-07", tz="UTC",
                                          freq=f"{int(config.rebalance_days)}D", inclusive="left", name="decision_at")
            pd.testing.assert_index_equal(strategy.index, expected_dates)
            pd.testing.assert_index_equal(strategy.index, benchmark.index)
            self.assertTrue(audit.selected_assets.eq(math.ceil(19 * config.top_fraction)).all())

    def test_empty_universe_targets_cash_then_resumes_normal_selection(self):
        schedules = self.schedules.copy()
        empty_day = pd.Timestamp("2022-01-01", tz="UTC")
        mask = schedules.decision_at.eq(empty_day)
        schedules.loc[mask, "active"] = False
        schedules.loc[mask, "universe_size"] = 0
        members = self.members.loc[self.members.decision_at.ne(empty_day)]
        strategy, benchmark, _, audit = build_configuration_targets(
            self.panel, members, schedules, self.protocol,
            {"lookback_days": 5, "top_fraction": .2, "rebalance_days": 3})
        self.assertEqual(strategy.loc[empty_day, "CASH"], 1.)
        self.assertEqual(benchmark.loc[empty_day, "CASH"], 1.)
        self.assertEqual(strategy.loc[empty_day].drop("CASH").sum(), 0.)
        self.assertEqual(audit.loc[empty_day, "selected_assets"], 0)
        self.assertEqual(strategy.loc["2022-01-04", "CASH"], 0.)
        self.assertEqual(audit.loc["2022-01-04", "selected_assets"], 4)

    def test_entirely_empty_eligible_universe_has_cash_only_targets(self):
        protocol = deepcopy(self.protocol)
        protocol["excluded_products"] = self.prices.product_id.unique().tolist()
        panel, members, coverage = build_daily_universe(self.prices, protocol)
        schedules, _ = build_development_schedules(coverage, protocol)
        strategy, benchmark, rankings, audit = build_configuration_targets(
            panel, members, schedules, protocol,
            {"lookback_days": 5, "top_fraction": .2, "rebalance_days": 3})
        self.assertEqual(strategy.columns.tolist(), ["CASH"])
        self.assertTrue(strategy.CASH.eq(1).all())
        pd.testing.assert_frame_equal(strategy, benchmark)
        self.assertTrue(rankings.empty)
        self.assertTrue(audit.selected_assets.eq(0).all())

    def test_stale_membership_and_schedule_are_rejected(self):
        config = {"lookback_days": 5, "top_fraction": .2, "rebalance_days": 3}
        with self.assertRaisesRegex(ValueError, "Membership counts"):
            build_configuration_targets(self.panel, self.members.iloc[1:], self.schedules, self.protocol, config)
        with self.assertRaisesRegex(ValueError, "full ordered"):
            build_configuration_targets(self.panel, self.members, self.schedules.iloc[1:], self.protocol,
                dict(config, rebalance_days=1))
        with self.assertRaisesRegex(ValueError, "revised protocol"):
            build_configuration_targets(self.panel, self.members, self.schedules,
                dict(self.protocol, universe={"min_assets": 20, "max_assets": 30}), config)

    def test_future_prices_cannot_change_earlier_targets(self):
        first, _, _, _ = self.run_config()
        altered = self.prices.copy()
        cutoff = pd.Timestamp("2022-01-02", tz="UTC")
        changed_rows = altered.timestamp.ge(cutoff) & altered.product_id.eq("C18-USD")
        altered.loc[changed_rows, "close"] /= 2
        panel, members, coverage = build_daily_universe(altered, self.protocol)
        schedules, _ = build_development_schedules(coverage, self.protocol)
        later, _, _, _ = build_configuration_targets(panel, members, schedules, self.protocol,
            {"lookback_days": 5, "top_fraction": .2, "rebalance_days": 3})
        pd.testing.assert_frame_equal(first.loc[:cutoff], later.loc[:cutoff])

    def test_declining_membership_uses_new_count_not_cash(self):
        prices = self.prices.loc[~(self.prices.product_id.eq("C18-USD")
            & self.prices.timestamp.ge(pd.Timestamp("2022-01-02", tz="UTC")))]
        panel, members, coverage = build_daily_universe(prices, self.protocol)
        schedules, _ = build_development_schedules(coverage, self.protocol)
        strategy, benchmark, _, audit = build_configuration_targets(panel, members, schedules, self.protocol,
            {"lookback_days": 5, "top_fraction": .2, "rebalance_days": 1})
        self.assertEqual(audit.loc["2022-01-03", "universe_size"], 18)
        self.assertEqual(audit.loc["2022-01-03", "selected_assets"], 4)
        self.assertTrue(strategy.CASH.eq(0).all())
        self.assertEqual(benchmark.loc["2022-01-03", "C18-USD"], 0.)

    def test_inputs_are_not_mutated(self):
        protocol = deepcopy(self.protocol)
        members = self.members.copy(deep=True)
        schedules = self.schedules.copy(deep=True)
        self.run_config()
        self.assertEqual(self.protocol, protocol)
        pd.testing.assert_frame_equal(self.members, members)
        pd.testing.assert_frame_equal(self.schedules, schedules)


if __name__ == "__main__":
    unittest.main()
