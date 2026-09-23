"""Check that notebook consolidation preserves timing and ledger boundaries."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import momentum_backtest
import momentum_research as research
import momentum_validation


def study_fixture():
    schedule = pd.DataFrame(index=pd.bdate_range("2015-12-31", "2020-12-31", name="Date"))
    dates = schedule.index[1:]
    tickers = pd.Index(["A.L", "B.L"], name="ticker")
    calendar = momentum_backtest.build_monthly_rebalance_calendar(schedule)
    calendar = calendar.loc[calendar["execution_date"].isin(dates)]
    executions = pd.DatetimeIndex(calendar["execution_date"], name="Date")
    selection = pd.DataFrame({"signal_date": calendar.index}, index=executions)
    grid = pd.DataFrame({"formation_sessions": [126, 252], "skip_sessions": [0, 21],
                         "top_frac": [.5, .5]}, index=pd.Index([0, 1], name="configuration_id"))
    noise = np.where(np.arange(len(dates)) % 2, .001, -.001)
    # Candidate A wins the first training window, B wins the next one.
    returns = [np.where(dates.year <= 2018, .01, -.05) + noise,
               np.where(dates.year <= 2018, -.01, .1) + noise]
    sweep = {"parameter_grid": grid, "backtests": {}}
    for key, weights in enumerate([[1., 0.], [0., 1.]]):
        sweep["backtests"][key] = {
            "daily": pd.DataFrame({"net_return": returns[key]}, index=dates),
            "targets": pd.DataFrame([weights] * len(executions), index=executions, columns=tickers),
            "selection": selection.copy(),
        }
    sweep["benchmark"] = {
        "daily": pd.DataFrame({"net_return": 0.}, index=dates),
        "targets": pd.DataFrame(.5, index=executions, columns=tickers),
    }
    index = pd.MultiIndex.from_product([dates, tickers], names=["Date", "ticker"])
    market = pd.DataFrame({"open_gbp": 1., "close_gbp": 1., "dividend_gbp": 0.,
                           "open_reference_usable": True, "close_reference_usable": True}, index=index)
    protocol = {
        "parameter_grid": grid.reset_index().to_dict("records"), "initial_capital_gbp": 1000.,
        "execution": {"cost_per_side_bps": 100., "max_missing_valuation_sessions": 0},
        "walk_forward": {"training_years": 3, "selection_metric": "sharpe_zero_rf",
                         "refit_frequency": "annual", "tie_break": "lowest_configuration_id"},
    }
    return sweep, market, schedule, protocol


class ResearchReplayTests(unittest.TestCase):
    def test_continuous_years_charge_actual_switch_and_separate_period_restarts_cash(self):
        sweep, market, schedule, protocol = study_fixture()
        result = research.run_walk_forward_period(sweep, market, schedule, protocol,
                                                  {"start": "2019-01-01", "end_exclusive": "2021-01-01"})
        self.assertEqual(result["choices"]["configuration_id"].tolist(), [0, 1])
        daily = result["backtests"]["Walk-forward"]["daily"]
        # One entry fee, then a sale and purchase when the selected stock changes.
        entered = 1000 / 1.01
        self.assertAlmostEqual(daily.loc["2019", "equity_gbp"].iloc[-1], entered)
        self.assertAlmostEqual(daily["equity_gbp"].iloc[-1], entered * .99 / 1.01)
        self.assertAlmostEqual(result["equity"]["Benchmark"].iloc[-1], entered)
        separate = research.run_walk_forward_period(sweep, market, schedule, protocol,
                                                    {"start": "2020-01-01", "end_exclusive": "2021-01-01"})
        self.assertAlmostEqual(separate["equity"]["Walk-forward"].iloc[-1], entered)

    def test_future_candidate_returns_cannot_change_annual_choices(self):
        sweep, market, schedule, protocol = study_fixture()
        bounds = {"start": "2019-01-01", "end_exclusive": "2021-01-01"}
        expected = research.run_walk_forward_period(sweep, market, schedule, protocol, bounds)
        changed = deepcopy(sweep)
        changed["backtests"][0]["daily"].loc["2020", "net_return"] = 100.
        changed["backtests"][1]["daily"].loc["2020", "net_return"] = -.99
        actual = research.run_walk_forward_period(changed, market, schedule, protocol, bounds)
        pd.testing.assert_frame_equal(actual["choices"], expected["choices"])
        pd.testing.assert_frame_equal(actual["equity"], expected["equity"])

    def test_midyear_start_is_not_silently_treated_as_complete_annual_fold(self):
        sweep, market, schedule, protocol = study_fixture()
        with self.assertRaisesRegex(ValueError, "complete annual folds"):
            research.run_walk_forward_period(sweep, market, schedule, protocol,
                                              {"start": "2019-06-01", "end_exclusive": "2021-01-01"})

    def test_reporting_rejects_an_equity_slice_with_the_wrong_starting_capital(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        daily = pd.DataFrame({"equity_gbp": [1100., 1210., 1331.], "net_return": .1,
                              "traded_value_gbp": 0., "total_cost_gbp": 0., "cash_weight": 0.}, index=dates)
        self.assertAlmostEqual(research.summarise_portfolios({"A": daily}, 1000.).loc["A", "total_return_pct"], 33.1)
        with self.assertRaisesRegex(ValueError, "reconcile"):
            research.summarise_portfolios({"A": daily.iloc[1:]}, 1000.)

    def test_inference_keeps_paired_dates_and_the_full_test_family(self):
        dates = pd.bdate_range("2020-01-01", periods=20)
        returns = pd.DataFrame({0: np.sin(np.arange(20)) / 100,
                                1: np.cos(np.arange(20)) / 100}, index=dates)
        benchmark = pd.Series(.001, index=dates)
        rules = dict(primary_block_length=3, sensitivity_block_lengths=[2, 5],
                     n_bootstrap=100, alpha=.05, seed=42)
        result = research.test_net_excess(returns, benchmark, rules)
        for block in [2, 3, 5]:
            expected = momentum_validation.bootstrap_sweep_excess(
                returns.sub(benchmark, axis=0), block_length=block, n_bootstrap=100)
            pd.testing.assert_frame_equal(result.xs(block), expected)
        with self.assertRaisesRegex(ValueError, "match exactly"):
            research.test_net_excess(returns, benchmark.iloc[1:], rules)


if __name__ == "__main__":
    unittest.main()
