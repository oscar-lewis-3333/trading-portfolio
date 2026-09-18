

#ensuring the features are implemented as intended
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "notebooks/reversal_signals_uk.ipynb"
sys.path.insert(0, str(PROJECT / "src"))
import reversal_namespace  # widens the trading_portfolio package path
import reversal_features
import trading_portfolio.reversal_signals.src.reversal_preparation as reversal_preparation


def feature_cells():
    targets = {"uk_extension_liquidity", "uk_extension_features", "uk_extension_eligible_counts"}
    found, cells = set(), []
    for number, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        names = {node.id for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        if names & targets:
            found |= names & targets
            cells.append(compile(source, f"{NOTEBOOK}:cell-{number}", "exec"))
    if found != targets or len(cells) != 3:
        raise AssertionError("Expected the three saved extension feature cells")
    return cells


def run_stage(prices, schedule, spec):
    namespace = dict(pd=pd, np=np, reversal_features=reversal_features,
                     display=lambda *a, **k: None, uk_extension_prices=prices,
                     uk_extension_schedule=schedule, uk_extension_frozen_spec=spec)
    with redirect_stdout(io.StringIO()), patch.object(
            socket.socket, "connect", side_effect=AssertionError("Offline feature checkpoint")):
        for cell in feature_cells():
            exec(cell, namespace)
    return namespace


def frozen_spec():
    return json.loads((PROJECT / "data/uk_extension_2023_2025_v1/extension_spec.json").read_text())["frozen_strategy"]


def fixture():
    # Session offsets must follow the supplied exchange index, not calendar days.
    dates = pd.bdate_range("2020-01-02", periods=97).delete([3, 9]).rename("Date")
    schedule = pd.DataFrame({
        "market_open": dates.tz_localize("UTC") + pd.Timedelta(hours=8),
        "market_close": dates.tz_localize("UTC") + pd.Timedelta(hours=16, minutes=30),
    }, index=dates)
    prices = []
    for ticker, close in [("A.L", 200. - np.arange(95)),
                          ("B.L", 100. + 2 * np.arange(95)),
                          ("C.L", np.full(95, 100.))]:
        prices.append(pd.DataFrame({"ticker": ticker, "Close": close,
                                    "adj_close": close, "Volume": 2000.}, index=dates))
    return pd.concat(prices).sort_index(), schedule


class SyntheticFeatureTests(unittest.TestCase):
    def test_gbp_activity_warmup_and_reversal_sign_match_hand_calculation(self):
        prices, schedule = fixture()
        ns = run_stage(prices, schedule, frozen_spec())
        date = schedule.index[60]
        key = (date, "A.L")
        self.assertEqual(ns["uk_extension_liquidity"].loc[key, "median_traded_value_gbp"], 341000.)
        self.assertEqual(ns["uk_extension_liquidity"].loc[key, "zero_volume_fraction"], 0.)
        self.assertFalse(ns["uk_extension_liquidity"].loc[(schedule.index[59], "A.L"), "history_ready"])
        for lookback, panel in ns["uk_extension_features"].items():
            self.assertAlmostEqual(panel.loc[key, "raw_reversal_score"], lookback / (140. + lookback))
            self.assertTrue(panel.loc[key, "eligible"])
            self.assertLess(panel.loc[(date, "B.L"), "raw_reversal_score"], 0.)
            self.assertEqual(panel.loc[(date, "C.L"), "raw_reversal_score"], 0.)
            self.assertEqual(panel.loc[key, "signal_time"], schedule.loc[date, "market_close"])

    def test_future_changes_do_not_change_earlier_scores_or_eligibility(self):
        prices, schedule = fixture()
        before = run_stage(prices, schedule, frozen_spec())
        changed = prices.copy()
        cutoff = schedule.index[70]
        changed.loc[changed.index > cutoff, ["Close", "adj_close"]] *= 9.
        changed.loc[changed.index > cutoff, "Volume"] = 0.
        after = run_stage(changed, schedule, frozen_spec())
        pd.testing.assert_frame_equal(before["uk_extension_liquidity"].loc[:cutoff], after["uk_extension_liquidity"].loc[:cutoff])
        for lookback, panel in before["uk_extension_features"].items():
            pd.testing.assert_frame_equal(panel.loc[:cutoff], after["uk_extension_features"][lookback].loc[:cutoff])
        self.assertNotEqual(before["uk_extension_features"][1].loc[(schedule.index[71], "A.L"), "raw_reversal_score"],
                            after["uk_extension_features"][1].loc[(schedule.index[71], "A.L"), "raw_reversal_score"])

    def test_todays_volume_changes_eligibility_but_not_background_liquidity(self):
        prices, schedule = fixture()
        before = run_stage(prices, schedule, frozen_spec())
        date = schedule.index[70]
        prices.loc[(prices.index == date) & prices.ticker.eq("A.L"), "Volume"] = 0.
        after = run_stage(prices, schedule, frozen_spec())
        pd.testing.assert_series_equal(before["uk_extension_liquidity"].loc[(date, "A.L")], after["uk_extension_liquidity"].loc[(date, "A.L")])
        self.assertFalse(after["uk_extension_eligible"].loc[(date, "A.L")])
        self.assertAlmostEqual(after["uk_extension_liquidity"].loc[(schedule.index[71], "A.L"), "zero_volume_fraction"], 1 / 60)

    def test_unknown_volume_is_not_zero_and_prevents_ready_history(self):
        prices, schedule = fixture()
        date = schedule.index[64]
        prices.loc[(prices.index == date) & prices.ticker.eq("A.L"), "Volume"] = np.nan
        ns = run_stage(prices, schedule, frozen_spec())
        activity = ns["uk_extension_activity"]
        self.assertTrue(activity.loc[(activity.index == date) & activity.ticker.eq("A.L"), "reported_zero_volume"].isna().all())
        self.assertFalse(ns["uk_extension_eligible"].loc[(date, "A.L")])
        self.assertFalse(ns["uk_extension_liquidity"].loc[(schedule.index[65], "A.L"), "history_ready"])

    def test_interior_gap_excludes_every_window_until_eleven_closes_recover(self):
        prices, schedule = fixture()
        prices.loc[(prices.index == schedule.index[65]) & prices.ticker.eq("A.L"), "adj_close"] = np.nan
        ns = run_stage(prices, schedule, frozen_spec())
        for panel in ns["uk_extension_features"].values():
            self.assertTrue(panel.loc[(schedule.index[64], "A.L"), "eligible"])
            self.assertFalse(panel.xs("A.L", level="ticker").loc[schedule.index[65:76], "eligible"].any())
            self.assertTrue(panel.loc[(schedule.index[76], "A.L"), "eligible"])
            pd.testing.assert_series_equal(panel.eligible, ns["uk_extension_eligible"], check_names=False)
        self.assertTrue(np.isfinite(ns["uk_extension_features"][10].loc[(schedule.index[70], "A.L"), "raw_reversal_score"]))

    def test_liquidity_thresholds_are_inclusive_and_apply_per_stock(self):
        prices, schedule = fixture()
        # C has exactly GBP 100,000 traded value on normal days.
        prices.loc[prices.ticker.eq("C.L"), "Volume"] = 1000.
        zero_days = schedule.index[[1, 2, 4]]
        prices.loc[prices.ticker.eq("C.L") & prices.index.isin(zero_days), "Volume"] = 0.
        before = run_stage(prices, schedule, frozen_spec())
        key = (schedule.index[60], "C.L")
        self.assertEqual(before["uk_extension_liquidity"].loc[key, "zero_volume_fraction"], .05)
        self.assertTrue(before["uk_extension_eligible"].loc[key])
        prices.loc[prices.ticker.eq("C.L") & (prices.index == schedule.index[5]), "Volume"] = 0.
        after = run_stage(prices, schedule, frozen_spec())
        self.assertFalse(after["uk_extension_eligible"].loc[key])
        self.assertTrue(after["uk_extension_eligible"].loc[(schedule.index[60], "A.L")])


class CachedSnapshotFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook_hash = reversal_preparation.sha256(NOTEBOOK)
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline checkpoint")):
            cls.prepared = reversal_preparation.prepare_uk_extension(PROJECT)
        cls.ns = run_stage(cls.prepared["prices"], cls.prepared["schedule"], cls.prepared["manifest"]["frozen_strategy"])

    def test_saved_cells_replay_with_one_index_and_common_boolean_eligibility(self):
        ns, prepared = self.ns, self.prepared
        self.assertEqual(ns["uk_extension_formation_windows"], [1, 3, 5, 10])
        self.assertEqual(len(ns["uk_extension_grid"]), 36)
        for panel in ns["uk_extension_features"].values():
            pd.testing.assert_index_equal(panel.index, prepared["panel"].index)
            self.assertTrue(pd.api.types.is_bool_dtype(panel.eligible))
            self.assertFalse(panel.eligible.isna().any())
            pd.testing.assert_series_equal(panel.eligible, ns["uk_extension_eligible"], check_names=False)
            self.assertTrue(np.isfinite(panel.loc[panel.eligible, "raw_reversal_score"]).all())

    def test_suspensions_quarantines_and_missing_rows_are_ineligible(self):
        panel = self.prepared["panel"]
        unavailable = panel.close_suspended | panel.unverified_quote | ~panel.observed_row
        self.assertGreater(int(unavailable.sum()), 0)
        self.assertFalse(self.ns["uk_extension_eligible"].loc[unavailable].any())
        dates = self.prepared["schedule"].index
        self.assertFalse(self.ns["uk_extension_eligible_counts"].loc[dates[:60]].any())

    def test_selected_stock_histories_match_direct_past_only_calculations(self):
        # Audit representative corrected histories independently, using date
        # slices rather than the implementation's rolling/shift expressions.
        for ticker, date in [("HSM.L", "2023-11-15"), ("TIN.L", "2025-11-14"), ("RFG.L", "2021-11-23")]:
            with self.subTest(ticker=ticker, date=date):
                prices = self.ns["uk_extension_keyed_prices"].xs(ticker, level="ticker")
                position = prices.index.get_loc(date)
                previous = prices.iloc[position - 60:position]
                actual = self.ns["uk_extension_liquidity"].loc[(pd.Timestamp(date), ticker)]
                self.assertEqual(len(previous), 60)
                self.assertTrue(previous[["Close", "Volume"]].notna().all().all())
                values = previous.Close.to_numpy() * previous.Volume.to_numpy()
                self.assertAlmostEqual(actual.median_traded_value_gbp, float(np.median(values)), places=6)
                self.assertAlmostEqual(actual.zero_volume_fraction, float((previous.Volume.to_numpy() == 0).mean()))
                for lookback, features in self.ns["uk_extension_features"].items():
                    expected = 1. - prices.adj_close.iloc[position] / prices.adj_close.iloc[position - lookback]
                    self.assertAlmostEqual(features.loc[(pd.Timestamp(date), ticker), "raw_reversal_score"], expected)

    def test_frozen_sources_inputs_and_notebook_are_unchanged(self):
        self.assertEqual(reversal_preparation.sha256(NOTEBOOK), self.notebook_hash)
        protected = json.loads((PROJECT / "data/uk_extension_2023_2025_v1/preparation_backup_v1/protected_sha256.json").read_text())
        for path, digest in protected.items():
            reversal_preparation.verify_frozen_source(PROJECT / path, digest)
        self.assertFalse(any(name in self.ns for name in ["uk_extension_equity", "uk_extension_net_excess", "uk_extension_choices"]))

    @classmethod
    def tearDownClass(cls):
        counts = cls.ns["uk_extension_eligible_counts"]
        print("Verified eligible-stock counts by year:\n" + counts.groupby(counts.index.year).agg(["min", "median", "max"]).to_string())


if __name__ == "__main__":
    unittest.main()
