"""Check metric definitions, full-grid coverage and matched benchmark reuse."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reversal_sweep
from rebalance_schedule import build_development_grid


class SweepMetricTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"development_start": "2022-01-01", "development_end_exclusive": "2022-01-04",
                         "holdout_start": "2022-01-04"}
        dates = pd.date_range("2022-01-01", periods=3, tz="UTC")
        self.ledger = pd.DataFrame({
            "valuation_at": dates + pd.Timedelta(days=1), "net_return": [-.1, .2, -.05],
            "traded_notional_gbp": [1000., 200., 0.], "equity_before_rebalance_gbp": [1000., 900., np.nan],
            "fee_gbp": [1., .2, 0.], "other_cost_gbp": [.5, .1, 0.],
            "accounting_error_gbp": 0., "daily_pnl_error_gbp": 0.,
        }, index=dates)

    def test_hand_calculated_growth_drawdown_turnover_and_sharpe(self):
        m = reversal_sweep.summarise_development_ledger(self.ledger, self.protocol)
        returns = np.array([-.1, .2, -.05])
        self.assertAlmostEqual(m["total_return"], .026)
        self.assertAlmostEqual(m["cagr"], 1.026 ** (365 / 3) - 1)
        self.assertAlmostEqual(m["max_drawdown"], -.1)
        self.assertAlmostEqual(m["annualised_one_way_turnover"], (1 + 200 / 900) * 365 / 3)
        self.assertAlmostEqual(m["total_cost_gbp"], 1.8)
        self.assertAlmostEqual(m["sharpe_zero_cash_rate"], returns.mean() / returns.std(ddof=1) * np.sqrt(365))

    def test_zero_volatility_has_undefined_sharpe(self):
        self.ledger["net_return"] = 0.
        m = reversal_sweep.summarise_development_ledger(self.ledger, self.protocol)
        self.assertTrue(np.isnan(m["sharpe_zero_cash_rate"]))
        self.assertEqual(m["total_return"], 0.)

    def test_missing_day_and_wrong_availability_fail(self):
        with self.assertRaisesRegex(ValueError, "every development day"):
            reversal_sweep.summarise_development_ledger(self.ledger.iloc[1:], self.protocol)
        self.ledger["valuation_at"] -= pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "following midnight"):
            reversal_sweep.summarise_development_ledger(self.ledger, self.protocol)

    def test_missing_turnover_denominator_fails(self):
        self.ledger.iloc[0, self.ledger.columns.get_loc("equity_before_rebalance_gbp")] = np.nan
        with self.assertRaisesRegex(ValueError, "Turnover"):
            reversal_sweep.summarise_development_ledger(self.ledger, self.protocol)


class DevelopmentSweepTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"development_start": "2022-01-01", "development_end_exclusive": "2022-01-09",
                         "holdout_start": "2022-01-09"}
        self.grid = build_development_grid()
        self.days = pd.date_range("2022-01-01", periods=8, tz="UTC")
        self.targets = {}
        for config_id, config in self.grid.iterrows():
            dates = self.days[::int(config.rebalance_days)]
            self.targets[config_id] = {
                "strategy": pd.DataFrame({"A": 1., "B": 0., "CASH": 0.}, index=dates),
                "benchmark": pd.DataFrame({"A": .5, "B": .5, "CASH": 0.}, index=dates),
            }
        rows, refs = [], []
        for i, day in enumerate(self.days):
            for asset in ("A", "B"):
                opening = 100 * 1.02 ** i if asset == "A" else 100.
                rows.append({"timestamp": day, "product_id": asset, "open_gbp": opening,
                             "close_gbp": opening * 1.02 if asset == "A" else opening})
                refs.append({"day": day, "product_id": asset, "execution_at": day + pd.Timedelta(minutes=5),
                             "reference_bar_start": day + pd.Timedelta(minutes=4),
                             "assumed_available_at": day + pd.Timedelta(minutes=5), "reference_price_gbp": opening})
        self.prices = pd.DataFrame(rows)
        self.refs = pd.DataFrame(refs)
        self.schedule = pd.DataFrame({"day": self.days, "execution_at": self.days + pd.Timedelta(minutes=5),
                                      "required_assets": 2, "status": "ready"})

    def run_sweep(self):
        return reversal_sweep.run_development_sweep(self.prices, self.targets, self.grid,
            self.schedule, self.refs, self.protocol, fee_bps=0., other_cost_bps=0., progress=False)

    def test_all_sixty_real_ledgers_and_exactly_four_benchmarks(self):
        real = reversal_sweep.run_development_ledger
        with patch.object(reversal_sweep, "run_development_ledger", wraps=real) as calls:
            summary, strategy, benchmark = self.run_sweep()
        self.assertEqual(calls.call_count, 64)
        self.assertEqual(summary.index.tolist(), self.grid.index.tolist())
        self.assertEqual(strategy.shape, (8, 60))
        self.assertTrue(strategy.columns.equals(benchmark.columns))
        self.assertTrue(strategy.index.equals(self.days))
        np.testing.assert_allclose(strategy, .02, atol=1e-14)
        np.testing.assert_allclose(summary.total_return, 1.02 ** 8 - 1)
        np.testing.assert_allclose(summary.mean_daily_net_excess_bps, (strategy - benchmark).mean() * 10000)
        for interval, configs in self.grid.groupby("rebalance_days"):
            paired = benchmark[configs.index]
            self.assertTrue(paired.eq(paired.iloc[:, 0], axis=0).all().all())

    def test_missing_configuration_rejected_before_simulation(self):
        self.targets.pop(self.grid.index[0])
        with patch.object(reversal_sweep, "run_development_ledger", side_effect=AssertionError("Do not simulate")):
            with self.assertRaisesRegex(ValueError, "complete 60"):
                self.run_sweep()

    def test_wrong_benchmark_rejected_before_simulation(self):
        candidates = self.grid.index[self.grid.rebalance_days.eq(1)]
        self.targets[candidates[1]]["benchmark"].iloc[0] = [1., 0., 0.]
        with patch.object(reversal_sweep, "run_development_ledger", side_effect=AssertionError("Do not simulate")):
            with self.assertRaisesRegex(ValueError, "Benchmarks differ"):
                self.run_sweep()

    def test_parameter_id_mismatch_rejected(self):
        self.grid.loc[self.grid.index[0], "lookback_days"] = 999
        with self.assertRaisesRegex(ValueError, "parameters disagree"):
            self.run_sweep()


if __name__ == "__main__":
    unittest.main()
