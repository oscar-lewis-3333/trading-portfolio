
#final testing and ensuring cash interest works as expected
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from reversal_backtest import run_execution_backtest
from reversal_risk.execution import run_cash_interest_backtest
from reversal_risk.reporting import build_final_report
from test_sweep_inference import scalar_hac_mean, scalar_holm


def fixture():
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"], name="Date")
    schedule = pd.DataFrame({"market_close": dates.tz_localize("UTC") + pd.Timedelta(hours=16)}, index=dates)
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["Date", "ticker"])
    market = pd.DataFrame({"open_gbp": [10., 10., 11., 12.], "close_gbp": [10., 11., 12., 12.],
                           "dividend_gbp": [0., 0., .2, 0.], "known_suspended": False,
                           "open_reference_usable": True, "positive_reported_volume": True}, index=index)
    decisions = pd.DataFrame({"signal_time": schedule.market_close.iloc[:2].to_numpy(),
                              "target_weight": [.5, .5]}, index=index[:2])
    return decisions, market, schedule


class CashInterestTests(unittest.TestCase):
    def test_zero_rate_reproduces_original_ledger_and_inputs_are_unchanged(self):
        decisions, market, schedule = fixture()
        originals = [x.copy(deep=True) for x in (decisions, market, schedule)]
        args = dict(decisions=decisions, market=market, schedule=schedule,
                    holding_sessions=1, cost_per_side_bps=10, net_orders=True)
        base = run_execution_backtest(**args)
        new = run_cash_interest_backtest(**args, cash_aer=0)
        pd.testing.assert_frame_equal(new["daily"].drop(columns="cash_interest_gbp"), base["daily"])
        for key in ("orders", "holdings"):
            pd.testing.assert_frame_equal(new[key], base[key])
        for actual, expected in zip((decisions, market, schedule), originals):
            pd.testing.assert_frame_equal(actual, expected)

    def test_weekend_interest_and_trade_cash_reconcile_without_dividend_interest(self):
        decisions, market, schedule = fixture()
        result = run_cash_interest_backtest(decisions, market, schedule, holding_sessions=1,
                                             cash_aer=.038, cost_per_side_bps=10, net_orders=True)
        daily = result["daily"]
        self.assertEqual(daily.cash_interest_gbp.iloc[0], 0)
        self.assertAlmostEqual(daily.cash_interest_gbp.iloc[1], 10000 * (1.038 ** (3 / 365.25) - 1))
        for i in range(1, len(daily)):
            days = (daily.index[i] - daily.index[i-1]).days
            expected = daily.cash_gbp.iloc[i-1] * (1.038 ** (days / 365.25) - 1)
            self.assertAlmostEqual(daily.cash_interest_gbp.iloc[i], expected)
        fills = result["orders"].loc[lambda x: x.status.eq("assumed_fill")]
        cash_flow = fills.groupby("Date").cash_change_gbp.sum().reindex(daily.index, fill_value=0)
        np.testing.assert_allclose(daily.cash_gbp, 10000 + (cash_flow + daily.cash_interest_gbp).cumsum())
        np.testing.assert_allclose(fills.cost_gbp, fills.notional_gbp * .001)
        np.testing.assert_allclose(daily.equity_gbp, daily.cash_gbp + daily.stock_value_gbp + daily.dividend_reserve_gbp)

    def test_all_cash_compounds_across_actual_calendar_gaps(self):
        decisions, market, schedule = fixture()
        decisions.target_weight = 0.
        result = run_cash_interest_backtest(decisions, market, schedule, holding_sessions=1, cash_aer=.038)
        days = (schedule.index - schedule.index[0]).days
        np.testing.assert_allclose(result["daily"].equity_gbp, 10000 * 1.038 ** (days / 365.25))


class FinalInferenceTests(unittest.TestCase):
    def setUp(self):
        self.rules = dict(hac_lags=5, familywise_alpha=.05, alternative="greater",
                          hac_use_t=False, hac_kernel="bartlett", hac_use_correction=True,
                          multiple_testing="holm")
        dates = pd.bdate_range("2024-01-05", periods=81, name="Date")
        innovations = np.random.default_rng(123).normal(0, .012, 80)
        self.results = {}
        for i, name in enumerate(("Original", "Capped", "Drawdown")):
            for suffix, scale in (("", 1.), (" benchmark", .2)):
                returns = (innovations * (.7 + i/10) + .001) * scale
                equity = np.r_[10000., 10000 * np.cumprod(1 + returns)]
                self.results[name + suffix] = {"daily": pd.DataFrame({
                    "equity_gbp": equity, "cost_gbp": 0., "cash_interest_gbp": 0.,
                    "stale_value_gbp": 0.}, index=dates), "final_positions": {}}

    def test_all_nine_tests_match_scalar_hac_and_single_holm_family(self):
        report = build_final_report(self.results, self.rules)
        self.assertEqual(len(report["inference"]), 9)
        p = []
        for label, values in report["contrast_returns"].items():
            mean, se, z, probability = scalar_hac_mean(values, lags=5)
            row = report["inference"].loc[label]
            np.testing.assert_allclose(row[["mean_bps", "hac_se_bps", "z_stat", "p_one_sided"]].to_numpy(float),
                                       [mean*10000, se*10000, z, probability], rtol=1e-11, atol=1e-11)
            p.append(probability)
        np.testing.assert_allclose(report["inference"].p_holm_family, scalar_holm(p), atol=1e-12)
        self.assertEqual(report["inference"].role.eq("primary").sum(), 3)

    def test_cash_uses_calendar_days_and_each_variant_uses_its_own_benchmark(self):
        report = build_final_report(self.results, self.rules)
        self.assertAlmostEqual(report["cash_returns"].iloc[0], 1.038 ** (3/365.25) - 1)
        for name in ("Original", "Capped", "Drawdown"):
            np.testing.assert_allclose(report["contrast_returns"][f"{name} vs matched benchmark"],
                                       report["returns"][name] - report["returns"][f"{name} benchmark"])
        self.assertAlmostEqual(report["equity"]["Cash only"].iloc[0], 10000)

    def test_missing_variant_and_misaligned_dates_are_rejected(self):
        incomplete = dict(self.results)
        incomplete.pop("Drawdown")
        with self.assertRaises(ValueError): build_final_report(incomplete, self.rules)
        shifted = copy.deepcopy(self.results)
        shifted["Capped"]["daily"].index += pd.Timedelta(days=1)
        with self.assertRaises(ValueError): build_final_report(shifted, self.rules)


if __name__ == "__main__":
    unittest.main()
