
#check the sweep across features works as intended

from contextlib import redirect_stdout
import hashlib
import io
import json
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from test_uk_data_preparation import CACHE, NOTEBOOK, PROJECT, defines, run_cells

sys.path.insert(0, str(PROJECT / "src"))
import reversal_features
import reversal_validation


def saved_cell(name, namespace):
    for number, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if defines(source, name):
                with redirect_stdout(io.StringIO()):
                    exec(compile(source, f"{NOTEBOOK}:cell-{number}", "exec"), namespace)
                return
    raise AssertionError(f"No saved cell defining {name}")


def schedule_fixture(periods=80):
    dates = pd.bdate_range("2020-01-02", periods=periods + 2).delete([3, 9])
    dates.name = "Date"
    return pd.DataFrame({
        "market_open": dates.tz_localize("UTC") + pd.Timedelta(hours=8),
        "market_close": dates.tz_localize("UTC") + pd.Timedelta(hours=16),
    }, index=dates)


class UKSweepPreparationTests(unittest.TestCase):
    def fixture(self):
        schedule = schedule_fixture()
        frames = []
        for ticker in ["A.L", "B.L", "C.L"]:
            values = 200. - np.arange(len(schedule))
            if ticker == "B.L":
                values[65] = np.nan
            frames.append(pd.DataFrame({
                "ticker": ticker, "adj_close": values, "Volume": 1000.,
            }, index=schedule.index))
        prices = pd.concat(frames).sort_index()
        index = prices.set_index("ticker", append=True).sort_index().index
        positions = schedule.index.get_indexer(index.get_level_values("Date"))
        # Hand-declared baseline: 60-session warmup; B's gap affects its
        # six closing prices on positions 65..70, but no longer at 71.
        eligible = (positions >= 60) & ~(
            (index.get_level_values("ticker") == "B.L")
            & (positions >= 65) & (positions <= 70)
        )
        ns = {
            "pd": pd, "np": np, "display": lambda *a, **k: None,
            "reversal_features": reversal_features,
            "reversal_validation": reversal_validation,
            "uk_prices": prices, "uk_schedule": schedule,
            "uk_features": pd.DataFrame({"eligible": eligible}, index=index),
            "UK_BASELINE": {"formation_sessions": 5, "top_frac": .2,
                            "holding_sessions": 5, "rebalance_sessions": 5},
            "UK_ELIGIBILITY": {"liquidity_lookback_sessions": 60,
                               "require_complete_formation_window": True},
            "UK_PERIODS": {"holdout_start": "2022-01-01"},
            "uk_decision_dates": schedule.index[[60, 65, 70]],
        }
        saved_cell("uk_sweep_grid", ns)
        saved_cell("uk_sweep_features", ns)
        saved_cell("uk_sweep_decision_dates", ns)
        return ns

    def test_declared_grid_is_complete_unique_and_contains_baseline(self):
        ns = self.fixture()
        grid = ns["uk_sweep_grid"]
        self.assertEqual(len(grid), 36)
        self.assertFalse(grid.duplicated(list(ns["UK_SWEEP_SPEC"])).any())
        self.assertEqual(grid.index[grid["is_baseline"]].tolist(), [23])
        self.assertTrue(grid["holding_sessions"].eq(grid["rebalance_sessions"]).all())

    def test_interior_gap_excludes_all_windows_until_longest_history_recovers(self):
        ns = self.fixture()
        dates = ns["uk_schedule"].index
        self.assertTrue(ns["uk_features"].loc[(dates[74], "B.L"), "eligible"])
        for lookback, panel in ns["uk_sweep_features"].items():
            self.assertFalse(panel.loc[(dates[74], "B.L"), "eligible"])
            self.assertTrue(panel.loc[(dates[76], "B.L"), "eligible"])
            pd.testing.assert_series_equal(panel["eligible"], ns["uk_sweep_common_eligible"])
            # A has complete prices: today's 126 compared with 126 + lookback.
            self.assertAlmostEqual(
                panel.loc[(dates[74], "A.L"), "raw_reversal_score"],
                1. - 126. / (126. + lookback),
            )

    def test_future_price_changes_leave_earlier_features_and_eligibility_unchanged(self):
        ns = self.fixture()
        before = {key: frame.copy(deep=True) for key, frame in ns["uk_sweep_features"].items()}
        cutoff = ns["uk_schedule"].index[72]
        ns["uk_prices"].loc[ns["uk_prices"].index > cutoff, "adj_close"] *= 9.
        saved_cell("uk_sweep_features", ns)
        for key in before:
            pd.testing.assert_frame_equal(before[key].loc[:cutoff], ns["uk_sweep_features"][key].loc[:cutoff])

    def test_calendars_use_session_offsets_and_share_start_and_valuation_dates(self):
        ns = self.fixture()
        sessions = ns["uk_schedule"].index
        pd.testing.assert_index_equal(ns["uk_sweep_valuation_dates"], sessions[60:])
        for holding, dates in ns["uk_sweep_decision_dates"].items():
            positions = sessions.get_indexer(dates)
            self.assertEqual(positions[0], 60)
            self.assertTrue(np.all(np.diff(positions) == holding))
            self.assertTrue((positions + 1 + holding < len(sessions)).all())
            self.assertGreaterEqual(positions[-1] + holding + 1 + holding, len(sessions))


class UKExecutionConfigurationTests(unittest.TestCase):
    def fixture(self):
        schedule = schedule_fixture(12)
        index = pd.MultiIndex.from_product(
            [schedule.index, ["A.L", "B.L", "C.L"]], names=["Date", "ticker"]
        )
        features = pd.DataFrame({
            "signal_time": index.get_level_values("Date").map(schedule["market_close"]),
            "raw_reversal_score": [.1, -.1, -.2] * len(schedule),
            "eligible": True,
        }, index=index)
        market = pd.DataFrame({
            "open_gbp": 100., "close_gbp": 100., "dividend_gbp": 0.,
            "known_suspended": False, "open_reference_usable": True,
            "positive_reported_volume": True,
        }, index=index)
        return schedule, features, market

    def run_case(self, schedule, features, market, dates=None, holding=3, minimum=1, cost=100.):
        return reversal_validation.evaluate_execution_configuration(
            features, schedule.index[[2]] if dates is None else dates, market, schedule,
            top_frac=2 / 3, holding_sessions=holding, min_eligible=minimum,
            initial_capital_gbp=12120., cost_per_side_bps=cost,
        )

    def test_partial_investment_benchmark_weights_and_fees_match_hand_calculation(self):
        schedule, features, market = self.fixture()
        result = self.run_case(schedule, features, market)
        np.testing.assert_allclose(result["decisions"]["target_weight"], [.5, 0., 0.])
        np.testing.assert_allclose(result["benchmark_decisions"]["target_weight"], [1 / 6] * 3)
        for name in ["strategy", "benchmark"]:
            daily = result[name]["daily"]
            orders = result[name]["orders"]
            pd.testing.assert_index_equal(daily.index, schedule.index[2:])
            # Invest 6060 including entry fees: 6000 notional + 60 fee.
            # Flat-price exit delivers 5940. Uninvested 6060 -> final 12000.
            self.assertAlmostEqual(daily["equity_gbp"].iloc[-1], 12000.)
            self.assertAlmostEqual(daily["cost_gbp"].sum(), 120.)
            self.assertTrue(orders.loc[orders["side"].eq("buy"), "Date"].eq(schedule.index[3]).all())
            self.assertTrue(orders.loc[orders["side"].eq("sell"), "Date"].eq(schedule.index[6]).all())

    def test_no_declines_or_insufficient_eligibility_keeps_both_portfolios_in_cash(self):
        for minimum, falling in [(4, True), (1, False)]:
            schedule, features, market = self.fixture()
            if not falling:
                features["raw_reversal_score"] = -.1
            result = self.run_case(schedule, features, market, minimum=minimum)
            for name in ["strategy", "benchmark"]:
                self.assertTrue(result[name]["daily"]["equity_gbp"].eq(12120.).all())
                self.assertTrue(result[name]["orders"].empty)

    def test_same_selection_is_retained_at_successive_rebalances(self):
        schedule, features, market = self.fixture()
        result = self.run_case(schedule, features, market, dates=schedule.index[[2, 3, 4]], holding=1, cost=0.)
        for name, stocks in [("strategy", 1), ("benchmark", 3)]:
            orders = result[name]["orders"]
            fills = orders.loc[orders["status"].eq("assumed_fill")]
            self.assertEqual(len(fills), 2 * stocks)
            self.assertEqual(int(orders["status"].eq("retained").sum()), 2 * stocks)

    def test_future_inputs_do_not_change_earlier_execution_and_inputs_are_preserved(self):
        schedule, features, market = self.fixture()
        original = (features.copy(deep=True), market.copy(deep=True), schedule.copy(deep=True))
        before = self.run_case(schedule, features, market)
        for actual, expected in zip((features, market, schedule), original):
            pd.testing.assert_frame_equal(actual, expected)
        cutoff = schedule.index[4]
        market.loc[market.index.get_level_values("Date") > cutoff, ["open_gbp", "close_gbp"]] *= 2.
        features.loc[features.index.get_level_values("Date") > cutoff, "raw_reversal_score"] *= -100.
        after = self.run_case(schedule, features, market)
        for name in ["strategy", "benchmark"]:
            pd.testing.assert_frame_equal(before[name]["daily"].loc[:cutoff], after[name]["daily"].loc[:cutoff])

    def test_invalid_or_missing_decision_dates_are_rejected(self):
        schedule, features, market = self.fixture()
        for dates in [schedule.index[:0], schedule.index[[3, 2]], schedule.index[[2, 2]]]:
            with self.assertRaises(ValueError):
                self.run_case(schedule, features, market, dates=dates)
        incomplete = features.drop(schedule.index[2], level="Date")
        with self.assertRaisesRegex(ValueError, "every requested decision date"):
            self.run_case(schedule, incomplete, market, dates=schedule.index[[2, 3]])


class UKSweepNotebookReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not NOTEBOOK.is_file() or not CACHE.is_file():
            raise unittest.SkipTest("The UK notebook and frozen cache are required")
        protected = [NOTEBOOK, CACHE, *(PROJECT / "src").glob("*.py")]
        cls.hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        cells = []
        for number, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            cells.append((number, source))
            if defines(source, "uk_sweep_baseline"):
                break
        else:
            raise AssertionError("No saved baseline-equivalence cell found")
        cls.ns = {"__name__": "__main__", "display": lambda *a, **k: None}
        run_cells(cells, cls.ns)
        plt.close("all")

    def test_baseline_reproduces_both_daily_records_from_a_fresh_namespace(self):
        ns = self.ns
        self.assertEqual(ns["uk_baseline_id"], 23)
        self.assertEqual(ns["UK_PRIMARY_COST_BPS"], 10)
        self.assertEqual(len(ns["uk_sweep_valuation_dates"]), 1711)
        for name, original in [("strategy", ns["uk_cost_backtests"][10]), ("benchmark", ns["uk_benchmark_backtest"])]:
            daily = ns["uk_sweep_baseline"][name]["daily"]
            pd.testing.assert_frame_equal(daily, original["daily"], check_exact=False, rtol=1e-10, atol=1e-6)
            self.assertTrue((daily.index < pd.Timestamp(ns["UK_PERIODS"]["holdout_start"])).all())

    def test_real_feature_masks_and_calendars_preserve_the_declared_comparison(self):
        ns = self.ns
        self.assertEqual(int(ns["uk_sweep_common_eligible"].sum()), 154937)
        pd.testing.assert_series_equal(ns["uk_sweep_common_eligible"], ns["uk_features"]["eligible"])
        for panel in ns["uk_sweep_features"].values():
            pd.testing.assert_series_equal(panel["eligible"], ns["uk_sweep_common_eligible"])
        self.assertEqual({h: len(d) for h, d in ns["uk_sweep_decision_dates"].items()}, {1: 1709, 3: 569, 5: 341})

    def test_replay_preserves_notebook_source_and_frozen_cache(self):
        for path, digest in self.hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
