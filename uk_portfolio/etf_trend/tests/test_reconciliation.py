"""0% cash reconciliation: the edited backtester must reproduce every published etf_trend number."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import etf_research
import etf_risk_weighting
import etf_validation

DATA_START = "2000-01-01"
DATA_END = "2026-09-01"
ANALYSIS_START = "2017-04-10"
HOLDOUT_START = "2023-01-01"
QUOTE_EXCLUSIONS = {"IJPN": ["2025-10-24"], "CMOP": ["2025-10-24"]}
BACKTEST = {"initial_capital_gbp": 10_000.0, "cost_per_side_bps": 10.0, "cash_rate_annual": 0.0}
FINAL_SPEC = {
    "lookback_months": 6, "passive_fraction": 0.5566532251165176,
    "holdout_start": HOLDOUT_START, "holdout_end_exclusive": DATA_END, **BACKTEST,
}
EVALUATION_SETTINGS = {"bootstrap_block_months": 3, "bootstrap_samples": 10_000, "confidence_level": 0.95, "seed": 42, "cost_scenarios_bps": (5.0, 10.0, 20.0)}

#previous results
PUBLISHED_FINAL_GBP = {"trend": 13_325.30, "passive_cash": 12_462.23, "buy_and_hold": 15_077.45}
PUBLISHED_COSTS = {
    5.0: (13_382.19, 12_468.88, 16.93, -1.70, 33.58, 0.0364),
    10.0: (13_325.30, 12_462.23, 16.08, -2.67, 32.84, 0.0448),
    20.0: (13_212.23, 12_448.96, 14.37, -4.64, 31.35, 0.0670)
}
CACHE = PROJECT / "data" / "repaired" / f"IUSA.L_{DATA_START}_{DATA_END}_repaired.csv"


@unittest.skipUnless(CACHE.is_file(), "study price caches are not present")
class ZeroCashReconciliationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = etf_research.prepare_cached_data(
            PROJECT, etf_research.study_universe(), data_start=DATA_START, data_end=DATA_END,
            analysis_start=ANALYSIS_START, holdout_start=HOLDOUT_START, quote_exclusions=QUOTE_EXCLUSIONS
        )
        cls.holdout_inputs = etf_research.period_inputs(cls.data, "holdout")
        cls.holdout = etf_validation.run_frozen_evaluation(**cls.holdout_inputs, spec=FINAL_SPEC)
        cls.buy_and_hold = etf_research.run_buy_and_hold(cls.holdout_inputs, BACKTEST)

    def test_published_holdout_final_values(self):
        finals = {
            "trend": self.holdout["trend"]["daily"]["nav_close"].iloc[-1],
            "passive_cash": self.holdout["passive_cash"]["daily"]["nav_close"].iloc[-1],
            "buy_and_hold": self.buy_and_hold["daily"]["nav_close"].iloc[-1]
        }
        for name, published in PUBLISHED_FINAL_GBP.items():
            self.assertEqual(round(finals[name], 2), published, name)

    def test_zero_rate_credits_no_interest(self):
        for result in (*self.holdout.values(), self.buy_and_hold):
            self.assertTrue(result["daily"]["interest"].eq(0.0).all())

    def test_published_cost_and_bootstrap_table(self):
        summary, _ = etf_validation.run_cost_sensitivity(**self.holdout_inputs, spec=FINAL_SPEC, evaluation_settings=EVALUATION_SETTINGS)
        for cost, (trend, benchmark, mean_bps, lower, upper, p_value) in PUBLISHED_COSTS.items():
            row = summary.loc[cost]
            self.assertEqual(round(row["trend_final_gbp"], 2), trend)
            self.assertEqual(round(row["benchmark_final_gbp"], 2), benchmark)
            self.assertEqual(round(row["mean_monthly_excess_bps"], 2), mean_bps)
            self.assertEqual(round(row["lower_ci_bps"], 2), lower)
            self.assertEqual(round(row["upper_ci_bps"], 2), upper)
            self.assertEqual(round(row["one_sided_p_value"], 4), p_value)

    def test_extension_runner_reproduces_original_portfolios(self):
        #new runner's results must be same as before
        for period in ("development", "holdout"):
            targets = etf_risk_weighting.build_targets(self.data, period, vol_window=126)
            runs = etf_risk_weighting.run_portfolios(targets, BACKTEST)
            if period == "holdout":
                original_trend = self.holdout["trend"]
                original_hold = self.buy_and_hold
            else:
                _, sweep, benchmarks = etf_research.run_development(self.data, (6,), BACKTEST)
                original_trend = sweep[6]
                original_hold = benchmarks["buy_and_hold"]
            columns = ["nav_close", "net_return", "cost", "cash"]
            pd.testing.assert_frame_equal(runs["equal_weight_trend"]["daily"][columns], original_trend["daily"][columns])
            pd.testing.assert_frame_equal(runs["buy_and_hold"]["daily"][columns], original_hold["daily"][columns])


if __name__ == "__main__":
    unittest.main()