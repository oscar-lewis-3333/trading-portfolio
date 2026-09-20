#testing that cost scenarios rereun accounting while keeping every chosen target
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cost_stress
from weekly_sweep import weekly_protocol
from test_intraday_backtest import fixture


class CostStressTests(unittest.TestCase):
    def setUp(self):
        self.p, self.t, self.s, self.r = fixture()
        self.b = self.t.copy()
        self.b.loc[:, :] = [.5, .5, 0.]
        self.protocol = weekly_protocol()
        self.protocol["oos_end"] = "2024-01-09"

    def run_small(self):
        #only the period is shortened. the real ledger and metric calculations run.
        with patch.object(cost_stress, "_check_protocol"):
            return cost_stress.run_cost_stress(self.p, self.t, self.b, self.s, self.r, self.protocol)

    def test_entry_and_two_leg_rotation_match_closed_form_at_every_cost(self):
        summary, runs = self.run_small()
        for round_trip in [25., 40., 60.]:
            c = round_trip / 20000
            final_equity = 1000 / (125 * (1 + c)) * 200 * (1 - c) / (1 + c) * 50 / 40
            ledger, trades, positions = runs[(round_trip, "walk_forward")]
            self.assertAlmostEqual(ledger.iloc[-1].equity_close_gbp, final_equity)
            self.assertAlmostEqual(summary.loc[(round_trip, "walk_forward"), "total_return"], final_equity / 1000 - 1)
            np.testing.assert_allclose(trades.fee_gbp, trades.signed_notional_gbp.abs() * 9 / 10000)
            np.testing.assert_allclose(trades.other_cost_gbp, trades.signed_notional_gbp.abs() * (round_trip / 2 - 9) / 10000)
            self.assertLess(ledger.accounting_error_gbp.abs().max(), 1e-9)

    def test_inputs_unchanged_and_targets_and_execution_identical_across_costs(self):
        before = [x.copy(deep=True) for x in [self.p, self.t, self.b, self.s, self.r]]
        original = cost_stress.run_intraday_ledger
        calls = []
        def record(prices, targets, schedule, refs, **kwargs):
            calls.append((targets.copy(), schedule.copy(), refs.copy(), kwargs))
            return original(prices, targets, schedule, refs, **kwargs)
        
        with patch.object(cost_stress, "run_intraday_ledger", side_effect=record):
            self.run_small()
        self.assertEqual(len(calls), 6)

        for i, (targets, schedule, refs, kwargs) in enumerate(calls):
            pd.testing.assert_frame_equal(targets, self.t if i % 2 == 0 else self.b)
            pd.testing.assert_frame_equal(schedule, self.s)
            pd.testing.assert_frame_equal(refs, self.r)
            self.assertEqual(kwargs["initial_cash"], 1000)

        for actual, expected in zip([self.p, self.t, self.b, self.s, self.r], before):
            pd.testing.assert_frame_equal(actual, expected)

    def test_sharpe_comparison_uses_matching_cost_benchmark(self):
        summary, _ = self.run_small()
        for cost in [25., 40., 60.]:
            a, b = summary.loc[(cost, "walk_forward")], summary.loc[(cost, "benchmark")]
            self.assertAlmostEqual(a.sharpe_difference, a.sharpe_zero_cash_rate - b.sharpe_zero_cash_rate)
            
        returns = summary.xs("walk_forward", level="strategy").total_return
        self.assertTrue((returns.diff().dropna() < 0).all())

    def test_missing_target_day_fails(self):
        self.t = self.t.iloc[:1]
        with self.assertRaisesRegex(ValueError, "Mondays"):
            self.run_small()

    def test_changed_frozen_protocol_fails_before_simulation(self):
        with patch.object(cost_stress, "run_intraday_ledger") as ledger:
            with self.assertRaises(ValueError):
                cost_stress.run_cost_stress(self.p, self.t, self.b, self.s, self.r, self.protocol)
            ledger.assert_not_called()

    def test_incomplete_ledger_or_wrong_valuation_fails(self):
        original = cost_stress.run_intraday_ledger
        for truncate in [True, False]:
            def corrupt(*args, **kwargs):
                ledger, trades, positions = original(*args, **kwargs)
                if truncate:
                    ledger = ledger.iloc[:-1]
                else:
                    ledger["valuation_at"] = ledger.index
                return ledger, trades, positions
            with patch.object(cost_stress, "run_intraday_ledger", side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, "evaluation dates"):
                    self.run_small()


if __name__ == "__main__":
    unittest.main()
