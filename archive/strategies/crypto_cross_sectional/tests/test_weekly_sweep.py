#testing that the weekly sweep works as intended
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CRYPTO_RESEARCH_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
import weekly_sweep as sweep


def metric_ledger(returns, start="2022-01-01"):
    dates = pd.date_range(start, periods=len(returns), freq="D", tz="UTC")
    return pd.DataFrame({
        "valuation_at": dates + pd.Timedelta(days=1), "net_return": returns,
        "traded_notional_gbp": 0., "equity_before_rebalance_gbp": np.nan,
        "crypto_weight_close": 1., "fee_gbp": 0., "other_cost_gbp": 0.,
    }, index=dates)


def training_fixture():
    dates = pd.date_range("2021-12-20", "2024-01-08", freq="D", tz="UTC")
    prices = pd.DataFrame({"timestamp": dates, "close": 100.})
    weeks = pd.date_range(dates.min(), dates.max(), freq="W-MON")
    schedule = pd.DataFrame({"day": weeks, "execution_at": weeks + pd.Timedelta(minutes=5),
                             "required_assets": 2, "status": "ready"})
    refs = pd.DataFrame([{"day": day, "product_id": asset}
                         for day in weeks for asset in ["BTC-USD", "ETH-USD"]])
    return prices.copy(), prices, schedule, refs


class WeeklySweepTests(unittest.TestCase):
    def test_protocol_is_idempotent_and_never_overwrites_a_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            protocol = sweep.freeze_protocol(path)
            before = path.read_bytes()
            self.assertEqual(sweep.freeze_protocol(path), protocol)
            self.assertEqual(before, path.read_bytes())
            protocol["top_fractions"].append(.4)
            path.write_text(json.dumps(protocol))
            altered = path.read_bytes()
            with self.assertRaises(ValueError):
                sweep.freeze_protocol(path)
            self.assertEqual(altered, path.read_bytes())

    def test_drawdown_includes_baseline_and_turnover_counts_both_trade_directions(self):
        ledger = metric_ledger([-.2, .25])
        ledger["traded_notional_gbp"] = [200., 300.]
        ledger["equity_before_rebalance_gbp"] = [1000., 1000.]
        ledger["fee_gbp"] = [.5, .75]
        result = sweep.summarise_training_ledger(ledger, "2022-01-01", "2022-01-03")
        self.assertAlmostEqual(result["total_return"], 0)
        self.assertAlmostEqual(result["cagr"], 0)
        self.assertAlmostEqual(result["max_drawdown"], -.2)
        self.assertAlmostEqual(result["annualised_one_way_turnover"], 91.25)
        self.assertAlmostEqual(result["total_cost_gbp"], 1.25)

    def test_missing_daily_observation_or_future_valuation_fails(self):
        ledger = metric_ledger([.01, -.01, .02])
        with self.assertRaisesRegex(ValueError, "every calendar day"):
            sweep.summarise_training_ledger(ledger.drop(ledger.index[1]), "2022-01-01", "2022-01-04")
        ledger.loc[ledger.index[-1], "valuation_at"] += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "valuation timing"):
            sweep.summarise_training_ledger(ledger, "2022-01-01", "2022-01-04")

    def test_undefined_sharpe_is_not_selected(self):
        result = sweep.summarise_training_ledger(metric_ledger([0., 0.]), "2022-01-01", "2022-01-03")
        self.assertTrue(np.isnan(result["sharpe_zero_cash_rate"]))
        table = pd.DataFrame([{"lookback_days": 7, "top_fraction": .1,
                               "sharpe_difference": np.nan, "annualised_one_way_turnover": 0}])
        with self.assertRaisesRegex(ValueError, "No candidate"):
            sweep.rank_training_candidates(table.set_index(["lookback_days", "top_fraction"]))

    def test_selection_uses_fixed_score_then_turnover_then_parameter_ties(self):
        table = pd.DataFrame([
            [60, .3, 1., 3.], [30, .3, 1., 3.], [30, .1, 1., 3.],
            [7, .1, .99, 0.], [180, .3, 1. + 1e-14, 4.],
        ], columns=["lookback_days", "top_fraction", "sharpe_difference", "annualised_one_way_turnover"])
        ranked = sweep.rank_training_candidates(table.set_index(["lookback_days", "top_fraction"]))
        self.assertEqual(ranked.index[0], (30, .1))
        self.assertEqual(ranked.index[-1], (7, .1))

    def test_future_data_are_removed_before_features_and_simulation(self):
        inputs = training_fixture()
        cutoff = pd.Timestamp("2024-01-01", tz="UTC")
        feature_calls, ledger_calls = [], []

        def features(history, lookback_days):
            self.assertTrue(history.timestamp.lt(cutoff).all())
            self.assertTrue(history.close.gt(0).all())
            feature_calls.append(lookback_days)
            return history

        def universe(history, **spec):
            self.assertEqual(spec, {"max_assets": 30, "min_assets": 20})
            weeks = pd.date_range(pd.Timestamp("2021-12-20", tz="UTC"), history.timestamp.max() + pd.Timedelta(days=1), freq="W-MON")
            members = pd.DataFrame([
                {"decision_at": day, "product_id": asset, "momentum_return": score, "momentum_ready": True}
                for day in weeks for asset, score in [("BTC-USD", .2), ("ETH-USD", .1)]
            ])
            return members, pd.DataFrame({"universe_size": 2}, index=weeks)

        def ledger(prices, targets, schedule, refs, **kwargs):
            self.assertTrue(prices.timestamp.lt(cutoff).all())
            self.assertTrue(prices.close.gt(0).all())
            self.assertTrue((targets.index < cutoff).all())
            self.assertTrue(schedule.day.lt(cutoff).all())
            self.assertTrue(refs.day.lt(cutoff).all())
            self.assertEqual(kwargs, {"initial_cash": 1000., "fee_bps": 9., "other_cost_bps": 3.5})
            ledger_calls.append(1)
            dates = pd.date_range(targets.index.min(), prices.timestamp.max(), freq="D")
            returns = np.resize([.01, -.02, .015], len(dates))
            result = metric_ledger(returns, start=str(dates.min().date()))
            return result, pd.DataFrame(), pd.DataFrame()

        mutated = [frame.copy() for frame in inputs]
        for frame in mutated[:2]:
            frame.loc[frame.timestamp.ge(cutoff), "close"] = -999.
        with patch.object(sweep, "add_momentum_features", side_effect=features), \
             patch.object(sweep, "build_weekly_universe", side_effect=universe), \
             patch.object(sweep, "run_intraday_ledger", side_effect=ledger):
            first, runs, _ = sweep.run_training_sweep(*inputs, sweep.weekly_protocol(), progress=False)
            second, _, _ = sweep.run_training_sweep(*mutated, sweep.weekly_protocol(), progress=False)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(runs), 18)
        self.assertEqual(set(feature_calls), {7, 14, 30, 60, 90, 180})
        self.assertEqual(len(ledger_calls), 38)  # 18 candidates + one benchmark, twice.
        self.assertTrue(first.observations.eq(730).all())

    def test_unfrozen_cutoff_or_changed_grid_fails(self):
        inputs = training_fixture()
        with self.assertRaisesRegex(ValueError, "frozen fold"):
            sweep.run_training_sweep(*inputs, sweep.weekly_protocol(), training_end="2023-06-01")
        changed = sweep.weekly_protocol()
        changed["lookback_days"].append(21)
        with self.assertRaisesRegex(ValueError, "Protocol differs"):
            sweep.run_training_sweep(*inputs, changed)

    def test_missing_weekly_schedule_and_incomplete_training_period_fail(self):
        history, prices, schedule, refs = training_fixture()
        with self.assertRaisesRegex(ValueError, "weekly decision"):
            sweep.run_training_sweep(history, prices, schedule.iloc[1:], refs, sweep.weekly_protocol())
        truncated = prices.loc[prices.timestamp.lt(pd.Timestamp("2023-12-31", tz="UTC"))]
        with self.assertRaisesRegex(ValueError, "complete training cutoff"):
            sweep.run_training_sweep(history, truncated, schedule, refs, sweep.weekly_protocol())


if __name__ == "__main__":
    unittest.main()
