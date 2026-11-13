#analytical invariants and independent sclar bootstrap testing
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CRYPTO_RESEARCH_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
import inference
from weekly_sweep import weekly_protocol


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-01", periods=120, tz="UTC")
        rng = np.random.default_rng(7)
        self.b = pd.Series(rng.normal(.0002, .02, 120), index=self.dates)
        self.a = self.b + pd.Series(rng.normal(.0003, .01, 120), index=self.dates)

    def run_bootstrap(self, a=None, b=None, **kwargs):
        options = dict(block_days=7, repetitions=101, seed=12)
        options.update(kwargs)
        return inference.paired_sharpe_bootstrap(
            self.a if a is None else a, self.b if b is None else b, **options)

    def test_identical_returns_have_exactly_zero_difference_everywhere(self):
        summary, draws = self.run_bootstrap(a=self.b)
        np.testing.assert_array_equal(draws, np.zeros(101))
        self.assertEqual(summary.ci_lower, 0)
        self.assertEqual(summary.ci_upper, 0)

    def test_sharpe_is_invariant_to_positive_scaling(self):
        summary, draws = self.run_bootstrap(a=2 * self.b)
        np.testing.assert_allclose(draws, 0, atol=1e-12)
        self.assertAlmostEqual(summary.sharpe_difference, 0)

    def test_added_constant_return_has_strictly_positive_advantage(self):
        summary, draws = self.run_bootstrap(a=self.b + .001)
        self.assertGreater(summary.ci_lower, 0)
        self.assertTrue((draws > 0).all())
        self.assertAlmostEqual(summary.sharpe_difference, .001 / self.b.std() * np.sqrt(365))

    def test_scalar_circular_block_oracle(self):
        summary, draws = self.run_bootstrap(repetitions=31)
        rng = np.random.default_rng(12)
        expected = []
        for _ in range(31):
            dates = []
            while len(dates) < len(self.a):
                start = int(rng.integers(len(self.a)))
                dates.extend((start + offset) % len(self.a) for offset in range(7))
            a = self.a.iloc[dates[:len(self.a)]]
            b = self.b.iloc[dates[:len(self.b)]]
            expected.append((a.mean() / a.std() - b.mean() / b.std()) * np.sqrt(365))
        np.testing.assert_allclose(draws, expected, atol=1e-12)
        np.testing.assert_allclose([summary.ci_lower, summary.ci_upper], np.quantile(expected, [.025, .975]), atol=1e-12)

    def test_batch_size_does_not_change_seeded_result(self):
        one, draws_one = self.run_bootstrap(batch_size=1)
        many, draws_many = self.run_bootstrap(batch_size=37)
        np.testing.assert_array_equal(draws_one, draws_many)
        pd.testing.assert_series_equal(one, many)

    def test_invalid_dates_are_rejected_without_silent_alignment(self):
        cases = [self.b.iloc[:-1], self.b.iloc[::-1], self.b.drop(self.dates[3])]
        naive = self.b.copy()
        naive.index = naive.index.tz_localize(None)
        cases.append(naive)
        duplicate = self.b.copy()
        duplicate.index = self.dates[:1].append(self.dates[:-1])
        cases.append(duplicate)
        for bad in cases:
            with self.subTest(index=bad.index), self.assertRaises(ValueError):
                self.run_bootstrap(b=bad)
        for index in [self.dates + pd.Timedelta(minutes=1),
                      self.dates.delete(3)]:
            bad = pd.Series(np.arange(len(index)) / 1000, index=index)
            with self.assertRaises(ValueError):
                self.run_bootstrap(a=bad, b=bad)

    def test_invalid_returns_and_settings_fail(self):
        for value in [np.nan, np.inf, -1, -2]:
            bad = self.a.copy()
            bad.iloc[3] = value
            with self.assertRaises(ValueError):
                self.run_bootstrap(a=bad)
        with self.assertRaises(ValueError):
            self.run_bootstrap(a=self.a * 0 + .01)
        for options in [dict(block_days=61), dict(block_days=True), dict(repetitions=1), dict(batch_size=0),
                        dict(confidence_level=1), dict(annualisation_days=0)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_bootstrap(**options)

    def test_wrapper_enforces_period_valuation_and_frozen_inference(self):
        protocol = weekly_protocol()
        dates = pd.date_range(protocol["oos_start"], pd.Timestamp(protocol["oos_end"]) - pd.Timedelta(days=1), tz="UTC")
        ledger = pd.DataFrame({"valuation_at": dates + pd.Timedelta(days=1), "net_return": np.sin(np.arange(len(dates))) / 100}, index=dates)
        runs = {name: (ledger.copy(), None, None) for name in ["walk_forward", "benchmark"]}
        def fake(a, b, **kwargs):
            pd.testing.assert_series_equal(a, ledger.net_return)
            pd.testing.assert_series_equal(b, ledger.net_return)
            return pd.Series({"block_days": kwargs["block_days"], "sharpe_difference": 0.}), np.zeros(1)
        with patch.object(inference, "paired_sharpe_bootstrap", side_effect=fake) as mocked:
            summary, _ = inference.test_walk_forward_edge(runs, protocol)
        self.assertEqual(list(summary.index), [28, 14, 56])
        for call in mocked.call_args_list:
            self.assertEqual(call.kwargs["repetitions"], 10000)
            self.assertEqual(call.kwargs["confidence_level"], .95)
            self.assertEqual(call.kwargs["seed"], protocol["inference"]["seed"])
        for bad in [ledger.iloc[1:], ledger.assign(valuation_at=dates)]:
            with self.assertRaises(ValueError):
                inference.test_walk_forward_edge({**runs, "benchmark": (bad, None, None)}, protocol)


if __name__ == "__main__":
    unittest.main()
