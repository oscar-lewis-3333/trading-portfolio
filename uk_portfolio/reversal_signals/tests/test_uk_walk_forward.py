
#testing the walk-forward method works as intended

import unittest

import numpy as np
import pandas as pd

from test_uk_execution_checkpoint import backtest, decisions, fixture
from test_uk_netting import reconcile
import reversal_walk_forward as walk_forward


def selection_fixture():
    schedule, _ = fixture(n=15)
    grid = pd.DataFrame({
        "formation_sessions": [1, 3], "top_frac": [.5, .5],
        "holding_sessions": [3, 1],
    }, index=pd.Index([8, 2], name="configuration_id"))
    returns = pd.DataFrame(.01, index=schedule.index[1:], columns=grid.index)
    rules = {
        "selection_metric": "mean_daily_net_excess",
        "tie_break": "lowest_configuration_id", "minimum_training_mean": 0.,
    }
    folds = walk_forward.build_walk_forward_folds(
        returns, schedule, initial_train_sessions=4, test_sessions=3,
    )
    return schedule, grid, returns, rules, folds


class WalkForwardSelectionTests(unittest.TestCase):
    def test_folds_use_known_close_returns_and_consecutive_test_sessions(self):
        schedule, _, returns, _, folds = selection_fixture()
        dates = schedule.index
        self.assertEqual(folds["train_sessions"].tolist(), [4, 7, 10, 13])
        self.assertEqual(folds["test_sessions"].tolist(), [3, 3, 3, 1])
        self.assertEqual(folds["selection_date"].tolist(), dates[[4, 7, 10, 13]].tolist())
        self.assertEqual(folds["test_start"].tolist(), dates[[5, 8, 11, 14]].tolist())
        self.assertEqual(folds["test_end"].tolist(), dates[[7, 10, 13, 14]].tolist())
        self.assertTrue(folds["train_start"].eq(returns.index[0]).all())
        for row in folds.itertuples():
            self.assertEqual(row.selection_time, schedule.at[row.selection_date, "market_close"])
            self.assertLess(row.selection_time, schedule.at[row.test_start, "market_open"])

    def test_selection_includes_latest_known_close_but_excludes_next_return(self):
        schedule, grid, returns, rules, folds = selection_fixture()
        returns[8] = 0.
        returns[2] = .01
        returns.loc[schedule.index[5]:, 8] = 1.
        selected = walk_forward.select_walk_forward_configurations(returns, folds, grid, rules)
        self.assertEqual(selected.loc[0, "configuration_id"], 2)
        self.assertEqual(selected.loc[1, "configuration_id"], 8)
        self.assertAlmostEqual(selected.loc[0, "best_training_mean_bps"], 100.)
        # This return is known exactly at the first selection close.
        returns.at[schedule.index[4], 8] = .08
        selected = walk_forward.select_walk_forward_configurations(returns, folds, grid, rules)
        self.assertEqual(selected.loc[0, "configuration_id"], 8)
        self.assertAlmostEqual(selected.loc[0, "best_training_mean_bps"], 200.)

    def test_exact_tie_uses_lowest_id_and_zero_or_negative_mean_stays_cash(self):
        _, grid, returns, rules, folds = selection_fixture()
        chosen = walk_forward.select_walk_forward_configurations(returns, folds, grid, rules)
        self.assertTrue(chosen["configuration_id"].eq(2).all())
        for value in [0., -.01]:
            with self.subTest(mean=value):
                returns[:] = value
                chosen = walk_forward.select_walk_forward_configurations(returns, folds, grid, rules)
                self.assertTrue(chosen["configuration_id"].isna().all())
                self.assertFalse(chosen["allow_new_entries"].any())

    def test_future_changes_cannot_reselect_earlier_folds_and_inputs_are_preserved(self):
        schedule, grid, returns, rules, folds = selection_fixture()
        before = returns.copy(deep=True)
        original_grid, original_folds = grid.copy(deep=True), folds.copy(deep=True)
        first = walk_forward.select_walk_forward_configurations(returns, folds, grid, rules)
        pd.testing.assert_frame_equal(returns, before)
        pd.testing.assert_frame_equal(grid, original_grid)
        pd.testing.assert_frame_equal(folds, original_folds)
        altered = returns.copy(deep=True)
        altered.loc[altered.index > schedule.index[7], 8] = 10.
        second = walk_forward.select_walk_forward_configurations(altered, folds, grid, rules)
        pd.testing.assert_frame_equal(first.iloc[:2], second.iloc[:2])
        self.assertEqual(second.loc[2, "configuration_id"], 8)

    def test_missing_sessions_candidates_or_training_values_are_rejected(self):
        schedule, grid, returns, rules, folds = selection_fixture()
        with self.assertRaises(ValueError):
            walk_forward.build_walk_forward_folds(returns.drop(returns.index[2]), schedule, 4, 3)
        with self.assertRaises(ValueError):
            walk_forward.select_walk_forward_configurations(returns[[8]], folds, grid, rules)
        returns.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            walk_forward.select_walk_forward_configurations(returns, folds, grid, rules)


def target_fixture():
    schedule, market = fixture(n=16, tickers=("A.L", "B.L", "C.L"))
    dates, index = schedule.index, market.index
    panels = {}
    for formation, scores in [(1, [.2, -.1, -.2]), (3, [-.2, .3, .1])]:
        panels[formation] = pd.DataFrame({
            "signal_time": index.get_level_values("Date").map(schedule["market_close"]),
            "eligible": True, "raw_reversal_score": scores * len(dates),
        }, index=index)
    choices = pd.DataFrame({
        "selection_date": dates[[2, 6, 10]], "test_end": dates[[6, 10, 15]],
        "formation_sessions": [1, 3, 1], "holding_sessions": [3, 1, 3],
        "top_frac": [.5, .5, .5], "configuration_id": [8, 2, pd.NA],
        "allow_new_entries": [True, True, False],
    }, index=pd.Index([0, 1, 2], name="fold"))
    calendars = {1: dates[:14], 3: dates[[0, 3, 6, 9]]}
    return schedule, market, choices, panels, calendars


class WalkForwardTargetTests(unittest.TestCase):
    def test_global_calendar_boundaries_and_unfilled_slots(self):
        schedule, _, choices, panels, calendars = target_fixture()
        result = walk_forward.build_walk_forward_decisions(choices, panels, calendars, min_eligible=1)
        dates = schedule.index
        self.assertEqual(result.index.get_level_values("Date").unique().tolist(),
                         dates[[2, 3, 6, 7, 8, 9]].tolist())
        initial = result.xs(dates[2], level="Date")
        self.assertTrue(initial["target_weight"].eq(0).all())
        self.assertTrue(initial["configuration_id"].isna().all())
        first = result.xs(dates[3], level="Date")
        np.testing.assert_allclose(first["target_weight"], [.5, 0., 0.])
        self.assertTrue(first["holding_sessions"].eq(3).all())
        boundary = result.xs(dates[6], level="Date")
        np.testing.assert_allclose(boundary["target_weight"], [0., .5, .5])
        self.assertTrue(boundary["configuration_id"].eq(2).all())
        self.assertTrue(boundary["holding_sessions"].eq(1).all())

    def test_all_cash_choices_still_value_from_first_selection_close(self):
        schedule, market, choices, panels, calendars = target_fixture()
        choices["allow_new_entries"] = False
        choices["configuration_id"] = pd.NA
        result = walk_forward.build_walk_forward_decisions(choices, panels, calendars, min_eligible=1)
        self.assertEqual(result.index.get_level_values("Date").unique().tolist(), [schedule.index[2]])
        lengths = result.groupby(level="Date")["holding_sessions"].first()
        book = backtest.run_execution_backtest(
            result, market, schedule, initial_capital_gbp=1000.,
            holding_sessions_by_date=lengths, cost_per_side_bps=10., net_orders=True,
        )
        pd.testing.assert_index_equal(book["daily"].index, schedule.index[2:])
        self.assertTrue(book["daily"]["equity_gbp"].eq(1000.).all())
        self.assertTrue(book["orders"].empty)

    def test_future_scores_do_not_change_past_targets_or_mutate_inputs(self):
        schedule, _, choices, panels, calendars = target_fixture()
        original_choices = choices.copy(deep=True)
        original_panels = {key: value.copy(deep=True) for key, value in panels.items()}
        first = walk_forward.build_walk_forward_decisions(choices, panels, calendars, min_eligible=1)
        pd.testing.assert_frame_equal(choices, original_choices)
        for key in panels:
            pd.testing.assert_frame_equal(panels[key], original_panels[key])
        cutoff = schedule.index[7]
        for panel in panels.values():
            panel.loc[panel.index.get_level_values("Date") > cutoff, "raw_reversal_score"] *= -10.
        second = walk_forward.build_walk_forward_decisions(choices, panels, calendars, min_eligible=1)
        pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])

    def test_missing_requested_feature_date_is_rejected(self):
        schedule, _, choices, panels, calendars = target_fixture()
        panels[3] = panels[3].drop(schedule.index[8], level="Date")
        with self.assertRaisesRegex(ValueError, "Missing feature dates"):
            walk_forward.build_walk_forward_decisions(choices, panels, calendars, min_eligible=1)


class VariableHoldingExecutionTests(unittest.TestCase):
    def run_book(self, schedule, market, targets, lengths, cost=0., capital=1000., net=True):
        return backtest.run_execution_backtest(
            targets, market, schedule, initial_capital_gbp=capital,
            cost_per_side_bps=cost, net_orders=net, holding_sessions_by_date=lengths,
        )

    def test_constant_series_preserves_scalar_execution_with_and_without_netting(self):
        schedule, market = fixture(n=12)
        targets = decisions(schedule, [(0, {"A.L": .5}), (3, {"B.L": .8}), (6, {"A.L": 1.})])
        lengths = pd.Series(3, index=schedule.index[[0, 3, 6]], dtype="int64")
        for net in [False, True]:
            scalar = backtest.run_execution_backtest(
                targets, market, schedule, initial_capital_gbp=1000.,
                holding_sessions=3, cost_per_side_bps=100., net_orders=net,
            )
            varying = self.run_book(schedule, market, targets, lengths, cost=100., net=net)
            for key in ["daily", "orders", "holdings"]:
                pd.testing.assert_frame_equal(scalar[key], varying[key])
            self.assertEqual(scalar["final_positions"], varying["final_positions"])

    def test_shorter_new_holding_does_not_shorten_existing_exit_and_fees_match_by_hand(self):
        schedule, market = fixture(n=12)
        dates = schedule.index
        targets = decisions(schedule, [(0, {"A.L": .5}), (2, {"B.L": .5})])
        lengths = pd.Series([5, 1], index=dates[[0, 2]], dtype="int64")
        result = self.run_book(schedule, market, targets, lengths, cost=100., capital=2020.)
        # A: 1010 allocation buys 1000 notional plus 10 fee; exit costs 10.
        # B: equity at its signal is 2010, giving a 1005 allocation.
        # Both prices stay at 10; each round trip loses exactly its two fees.
        b_one_way_fee = 1005. * .01 / 1.01
        self.assertAlmostEqual(result["daily"]["cost_gbp"].sum(), 20. + 2*b_one_way_fee)
        self.assertAlmostEqual(result["daily"]["equity_gbp"].iloc[-1], 2000. - 2*b_one_way_fee)
        self.assertEqual(result["holdings"].loc[(dates[3], "A.L"), "scheduled_exit"], dates[6])
        self.assertEqual(result["holdings"].loc[(dates[3], "B.L"), "scheduled_exit"], dates[4])
        fills = result["orders"].loc[lambda x: x["status"].eq("assumed_fill")]
        self.assertEqual(list(zip(fills["Date"], fills["ticker"], fills["side"])),
                         [(dates[1], "A.L", "buy"), (dates[3], "B.L", "buy"),
                          (dates[4], "B.L", "sell"), (dates[6], "A.L", "sell")])
        reconcile(self, result, market, 2020., 100.)

    def test_due_inventory_renews_to_new_horizon_without_a_round_trip(self):
        schedule, market = fixture()
        dates = schedule.index
        targets = decisions(schedule, [(0, {"A.L": 1.}), (1, {"A.L": 1.})])
        lengths = pd.Series([1, 3], index=dates[[0, 1]], dtype="int64")
        result = self.run_book(schedule, market, targets, lengths)
        self.assertEqual(result["orders"]["status"].tolist(), ["assumed_fill", "retained", "assumed_fill"])
        self.assertEqual(result["orders"]["Date"].tolist(), dates[[1, 2, 5]].tolist())
        self.assertEqual(result["holdings"].loc[(dates[2], "A.L"), "scheduled_exit"], dates[5])
        self.assertEqual(result["holdings"].loc[(dates[4], "A.L"), "units"], 100.)

    def test_blocked_renewal_keeps_original_overdue_exit(self):
        schedule, market = fixture()
        dates = schedule.index
        market.loc[(dates[2], "A.L"), "known_suspended"] = True
        targets = decisions(schedule, [(0, {"A.L": 1.}), (1, {"A.L": 1.})])
        lengths = pd.Series([1, 3], index=dates[[0, 1]], dtype="int64")
        result = self.run_book(schedule, market, targets, lengths)
        self.assertEqual(result["holdings"].loc[(dates[2], "A.L"), "scheduled_exit"], dates[2])
        sells = result["orders"].loc[lambda x: x["side"].eq("sell")]
        self.assertEqual(sells["status"].tolist(), ["blocked", "assumed_fill"])
        self.assertEqual(sells.iloc[-1]["Date"], dates[3])

    def test_future_horizon_and_price_changes_preserve_past_and_inputs(self):
        schedule, market = fixture(n=12)
        dates = schedule.index
        targets = decisions(schedule, [(0, {"A.L": 1.}), (6, {"B.L": 1.})])
        lengths = pd.Series([3, 1], index=dates[[0, 6]], dtype="int64")
        originals = [value.copy(deep=True) for value in [market, targets, lengths]]
        first = self.run_book(schedule, market, targets, lengths, cost=100.)
        pd.testing.assert_frame_equal(market, originals[0])
        pd.testing.assert_frame_equal(targets, originals[1])
        pd.testing.assert_series_equal(lengths, originals[2])
        lengths.iloc[1] = 3
        market.loc[market.index.get_level_values("Date") > dates[5], ["open_gbp", "close_gbp"]] *= 2.
        second = self.run_book(schedule, market, targets, lengths, cost=100.)
        for key in ["daily", "holdings"]:
            pd.testing.assert_frame_equal(first[key].loc[:dates[5]], second[key].loc[:dates[5]])
        pd.testing.assert_frame_equal(
            first["orders"].loc[lambda x: x["Date"] <= dates[5]],
            second["orders"].loc[lambda x: x["Date"] <= dates[5]],
        )

    def test_invalid_or_misaligned_horizons_are_rejected(self):
        schedule, market = fixture()
        targets = decisions(schedule, [(0, {"A.L": 1.}), (1, {"B.L": 1.})])
        index = schedule.index[:2]
        cases = [
            pd.Series([True, True], index=index), pd.Series([0, 1], index=index),
            pd.Series([1.5, 1.], index=index), pd.Series([pd.NA, 1], index=index, dtype="Int64"),
            pd.Series([1, 1], index=index[::-1]), pd.Series([1, 1], index=index[[0, 0]]),
            pd.Series([1, 1], index=schedule.index[1:3]),
            pd.Series([8, 1], index=index), pd.Series([2**64-1, 1], index=index, dtype="uint64"),
        ]
        for number, lengths in enumerate(cases):
            with self.subTest(case=number), self.assertRaises(ValueError):
                self.run_book(schedule, market, targets, lengths)


if __name__ == "__main__":
    unittest.main()
