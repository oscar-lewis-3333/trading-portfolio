#regress extracted stages against archived code and guard replay-cache provenance
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import reversal_research as research
from test_uk_extension_features import fixture, frozen_spec, run_stage
import test_uk_extension_execution as archived_execution


class ExtractedStageTests(unittest.TestCase):
    def test_features_match_archived_calculation_with_gaps_and_volume_shocks(self):
        prices, schedule = fixture()
        cases = [prices]
        missing = prices.copy()
        missing.loc[(missing.ticker == "B.L") & (missing.index == schedule.index[65]), "adj_close"] = np.nan
        cases.append(missing)
        volumes = prices.copy()
        volumes.loc[(volumes.ticker == "A.L") & (volumes.index >= schedule.index[71]), "Volume"] = 0.
        volumes.loc[(volumes.ticker == "C.L") & (volumes.index == schedule.index[62]), "Volume"] = np.nan
        cases.append(volumes)
        for number, sample in enumerate(cases):
            with self.subTest(case=number):
                original = run_stage(sample, schedule, frozen_spec())["uk_extension_features"]
                actual = research.build_features(sample, schedule, frozen_spec())
                for lookback in original:
                    pd.testing.assert_frame_equal(actual[lookback], original[lookback])

    def test_future_observations_do_not_change_past_feature_rows(self):
        prices, schedule = fixture()
        before = research.build_features(prices, schedule, frozen_spec())
        cutoff = schedule.index[70]
        prices.loc[prices.index > cutoff, ["Close", "adj_close"]] *= 7.
        prices.loc[prices.index > cutoff, "Volume"] = 0.
        after = research.build_features(prices, schedule, frozen_spec())
        for lookback in before:
            pd.testing.assert_frame_equal(before[lookback].loc[:cutoff], after[lookback].loc[:cutoff])

    def test_pre_year_selection_and_cash_markers_match_archive(self):
        oracle = archived_execution.EvaluationStartTests()
        for on_cash_date in (True, False):
            with self.subTest(on_cash_date=on_cash_date):
                original = oracle.run_targets(on_cash_date)
                inputs = {"spec": original["uk_extension_frozen_spec"],
                          "schedule": pd.DataFrame(index=original["uk_extension_sessions"]),
                          "features": original["uk_extension_features"],
                          "decision_dates": original["uk_extension_decision_dates"]}
                actual = research.selected_decisions(inputs, original["uk_extension_choices"], "2024-01-01")
                pd.testing.assert_frame_equal(actual, original["uk_extension_decisions"])
                pd.testing.assert_frame_equal(research.matched_benchmark(actual), original["uk_extension_benchmark_decisions"])

    def test_notebook_has_executable_stages_in_place_of_static_results(self):
        notebook = json.loads((PROJECT / "notebooks/reversal_signals_uk.ipynb").read_text())
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        for name in ("run_baseline", "run_sweep", "run_walk_forward", "run_cost_sensitivity",
                     "prepare_validation", "run_holdout", "run_risk_comparison"):
            self.assertIn("research." + name + "(", code)
        self.assertFalse(any(c.get("attachments") for c in notebook["cells"]))
        self.assertLess(len(notebook["cells"]), 44)
        self.assertNotIn("reversal_signals_ml", code)


class ReplayCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / "src").mkdir()
        self.source = root / "src/calculation.py"
        self.source.write_text("x = 1\n")
        self.inputs = {"project": root, "fingerprint": "original inputs"}
        self.calls = 0

    def calculate(self):
        self.calls += 1
        return pd.DataFrame({"result": [self.calls]})

    def test_unchanged_inputs_hit_cache_and_changes_invalidate_it(self):
        a = research.cached_stage(self.inputs, "stage", self.calculate, parameters={"cost": 10})
        b = research.cached_stage(self.inputs, "stage", self.calculate, parameters={"cost": 10})
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(self.calls, 1)
        research.cached_stage(self.inputs, "stage", self.calculate, parameters={"cost": 20})
        self.source.write_text("x = 2\n")
        research.cached_stage(self.inputs, "stage", self.calculate, parameters={"cost": 10})
        self.inputs["fingerprint"] = "changed inputs"
        research.cached_stage(self.inputs, "stage", self.calculate, parameters={"cost": 10})
        self.assertEqual(self.calls, 4)

    def test_corrupt_payload_is_rejected_and_explicit_rebuild_recomputes(self):
        research.cached_stage(self.inputs, "stage", self.calculate)
        path = next((self.inputs["project"] / "data").rglob("stage.pkl"))
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Replay cache changed"):
            research.cached_stage(self.inputs, "stage", self.calculate)
        result = research.cached_stage(self.inputs, "stage", self.calculate, use_cache=False)
        self.assertEqual(result.iloc[0, 0], 2)


class FrozenReplayTests(unittest.TestCase):
    #local data fixtures are required
    @classmethod
    def setUpClass(cls):
        development = research.prepare_development(PROJECT)
        cls.baseline = research.run_baseline(development)
        validation_inputs = research.prepare_validation(PROJECT, development)
        cls.validation = research.run_holdout(validation_inputs, "2022-01-01")
        extension_inputs = research.prepare_extension(PROJECT)
        extension = research.run_holdout(extension_inputs, "2024-01-01")
        cls.risk = research.run_risk_comparison(extension_inputs, extension)

    def test_baseline_matches_all_three_archived_terminal_values(self):
        expected = [46020.96, 22403.60, 15747.45]
        actual = self.baseline["equity"].iloc[-1].reindex(
            ["Reversal", "Rebalanced universe", "Buy and hold"])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=.005)

    def test_original_validation_reproduces_every_equity_date_and_fold_choice(self):
        fixtures = PROJECT / "data/uk_holdout_2021_2023_v1"
        for key, file in [("equity", "original_holdout_equity.pkl"),
                          ("choices", "original_holdout_choices.pkl")]:
            pd.testing.assert_frame_equal(self.validation[key], pd.read_pickle(fixtures / file),
                                          check_exact=False, rtol=1e-10, atol=1e-8)

    def test_risk_outputs_reconcile_to_all_six_frozen_tables(self):
        fixtures = PROJECT / "data/uk_final_review_2026_09_15"
        for name in ("equity", "summary", "inference", "returns", "contrast_returns", "cash_returns"):
            with self.subTest(table=name):
                actual = self.risk[name]
                expected = pd.read_csv(fixtures / f"{name}.csv", index_col=0)
                if name in ("equity", "returns", "contrast_returns", "cash_returns"):
                    expected.index = pd.to_datetime(expected.index)
                if isinstance(actual, pd.Series):
                    pd.testing.assert_series_equal(actual, expected.iloc[:, 0], check_freq=False,
                                                    check_names=False, rtol=1e-10, atol=1e-8)
                else:
                    pd.testing.assert_frame_equal(actual, expected, check_freq=False,
                                                   check_names=False, rtol=1e-10, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
