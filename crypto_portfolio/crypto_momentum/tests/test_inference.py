"""Synthetic-only bootstrap and protocol tests. No market data are loaded."""
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from _support import OfflineTestCase
import crypto_inference
import crypto_validation


def example_pair():
    rng = np.random.default_rng(71)
    dates = pd.date_range("2010-01-01", periods=64, tz="UTC")
    first = rng.normal(0.001, 0.02, len(dates))
    second = first * 0.6 + rng.normal(0, 0.003, len(dates))
    return pd.Series(first, index=dates), pd.Series(second, index=dates)


class BootstrapTests(OfflineTestCase):
    def test_blocks_wrap_and_truncate_to_exact_sample_length(self):
        rng = Mock()
        rng.integers.return_value = np.array([6, 1, 4])
        actual = crypto_inference._circular_block_indices(8, 3, rng)
        np.testing.assert_array_equal(actual, [6, 7, 0, 1, 2, 3, 4, 5])
        rng.integers.assert_called_once_with(0, 8, size=3)

    def test_draws_and_percentiles_match_independent_scalar_calculation(self):
        first, second = example_pair()
        summary, actual = crypto_inference.paired_sharpe_bootstrap(
            first, second, block_days=7, repetitions=250, seed=910)
        rng = np.random.default_rng(910)
        expected = []
        for _ in range(250):
            starts = rng.integers(0, 64, size=10)
            indices = [(int(start) + offset) % 64 for start in starts for offset in range(7)][:64]
            a = [float(first.iloc[i]) for i in indices]
            b = [float(second.iloc[i]) for i in indices]
            expected.append((statistics.mean(a) / statistics.stdev(a)
                             - statistics.mean(b) / statistics.stdev(b)) * math.sqrt(365.25))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        ordered = sorted(expected)
        def percentile(probability):
            position = probability * (len(ordered) - 1)
            lower = int(math.floor(position))
            fraction = position - lower
            return ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction
        self.assertAlmostEqual(summary.ci_lower, percentile(0.025), places=12)
        self.assertAlmostEqual(summary.ci_upper, percentile(0.975), places=12)

    def test_identical_series_have_exactly_zero_difference(self):
        first, _ = example_pair()
        summary, draws = crypto_inference.paired_sharpe_bootstrap(first, first, block_days=7, repetitions=100)
        self.assertTrue(draws.eq(0).all())
        self.assertEqual(summary.sharpe_difference, 0)
        self.assertEqual(summary.ci_lower, 0)
        self.assertEqual(summary.ci_upper, 0)

    def test_reproducibility_input_preservation_and_swap_symmetry(self):
        first, second = example_pair()
        originals = first.copy(), second.copy()
        kwargs = dict(block_days=7, repetitions=200, seed=123)
        summary, draws = crypto_inference.paired_sharpe_bootstrap(first, second, **kwargs)
        repeated, repeated_draws = crypto_inference.paired_sharpe_bootstrap(first, second, **kwargs)
        reverse, reversed_draws = crypto_inference.paired_sharpe_bootstrap(second, first, **kwargs)
        pd.testing.assert_series_equal(summary, repeated)
        pd.testing.assert_series_equal(draws, repeated_draws)
        np.testing.assert_allclose(reversed_draws, -draws, atol=1e-12)
        self.assertAlmostEqual(reverse.ci_lower, -summary.ci_upper, places=12)
        self.assertAlmostEqual(reverse.ci_upper, -summary.ci_lower, places=12)
        pd.testing.assert_series_equal(first, originals[0])
        pd.testing.assert_series_equal(second, originals[1])

    def test_full_length_blocks_preserve_all_observations(self):
        first, second = example_pair()
        summary, draws = crypto_inference.paired_sharpe_bootstrap(first, second, block_days=64, repetitions=50)
        np.testing.assert_allclose(draws, summary.sharpe_difference, atol=1e-12)

    def test_bad_dates_and_values_are_rejected(self):
        first, second = example_pair()
        with self.assertRaises(ValueError):
            crypto_inference.paired_sharpe_bootstrap(first, second.iloc[1:])
        invalids = [first.iloc[::-1], first.drop(first.index[4]),
                    pd.concat([first, first.iloc[-1:]]), first.tz_localize(None)]
        shifted = first.copy()
        shifted.index += pd.Timedelta(hours=1)
        invalids.append(shifted)
        for value in (np.nan, np.inf, -1.0):
            changed = first.copy()
            changed.iloc[0] = value
            invalids.append(changed)
        invalids.append(pd.Series(0.001, index=first.index))
        for number, invalid in enumerate(invalids):
            with self.subTest(case=number), self.assertRaises(ValueError):
                crypto_inference.paired_sharpe_bootstrap(invalid, invalid)

    def test_bad_bootstrap_parameters_are_rejected(self):
        first, second = example_pair()
        cases = [{"block_days": value} for value in (0, 65, True, 2.5)]
        cases += [{"repetitions": value} for value in (1, True, 2.5)]
        cases += [{"confidence_level": value} for value in (0, 1, np.nan)]
        cases += [{"annualisation_days": value} for value in (0, np.inf)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                crypto_inference.paired_sharpe_bootstrap(first, second, **kwargs)

    def test_undefined_resample_is_not_silently_discarded(self):
        first, second = example_pair()
        with patch.object(crypto_inference, "_circular_block_indices", return_value=np.zeros(64, dtype=int)):
            with self.assertRaises(ValueError):
                crypto_inference.paired_sharpe_bootstrap(first, second, repetitions=2)

    def test_published_synthetic_positive_shift_result(self):
        rng = np.random.default_rng(12345)
        returns = pd.Series(rng.normal(0.0003, 0.02, 731),
                            index=pd.date_range("2010-01-01", periods=731, tz="UTC"))
        summary, draws = crypto_inference.paired_sharpe_bootstrap(returns + 0.0005, returns)
        self.assertTrue(draws.gt(0).all())
        self.assertEqual(round(summary.sharpe_difference, 6), 0.471311)
        self.assertEqual(round(summary.ci_lower, 6), 0.453457)
        self.assertEqual(round(summary.ci_upper, 6), 0.492115)


class ProtocolTests(OfflineTestCase):
    def test_freeze_is_idempotent_and_hash_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "protocol.json"
            protocol = {"lookback": 30, "block_days": 28}
            first = crypto_validation.freeze_research_protocol(protocol, path)
            contents = path.read_bytes()
            second = crypto_validation.freeze_research_protocol(protocol, path)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), contents)
            canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
            self.assertEqual(first["protocol_sha256"], hashlib.sha256(canonical.encode()).hexdigest())

    def test_changed_protocol_cannot_overwrite_existing_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            crypto_validation.freeze_research_protocol({"lookback": 30}, path)
            contents = path.read_bytes()
            with self.assertRaises(ValueError):
                crypto_validation.freeze_research_protocol({"lookback": 14}, path)
            self.assertEqual(path.read_bytes(), contents)
