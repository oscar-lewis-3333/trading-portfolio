
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from reversal_validation import bootstrap_block_means


class BlockBootstrapTests(unittest.TestCase):
    def frame(self, values):
        return pd.DataFrame({"excess": values}, index=pd.date_range("2020-01-01", periods=len(values)))

    def test_clustered_series_matches_exact_circular_block_distribution(self):
        data = self.frame([0., 0., 0., 6., 6., 6.])
        probabilities = np.array([1, 4, 8, 10, 8, 4, 1]) / 36.
        draws = bootstrap_block_means(data, block_length=3, n_bootstrap=20_000)["excess"]
        self.assertTrue(draws.isin(range(7)).all())
        observed = draws.value_counts(normalize=True).reindex(range(7), fill_value=0.)
        np.testing.assert_allclose(observed, probabilities, rtol=0, atol=.015)
        self.assertAlmostEqual(draws.mean(), 3., delta=.05)
        self.assertAlmostEqual(draws.var(ddof=0), 11. / 6., delta=.08)

    def test_single_day_blocks_match_known_iid_mean_and_variance(self):
        data = self.frame([0., 0., 0., 6., 6., 6.])
        draws = bootstrap_block_means(data, block_length=1, n_bootstrap=20_000)["excess"]
        self.assertAlmostEqual(draws.mean(), 3., delta=.05)
        self.assertAlmostEqual(draws.var(ddof=0), 1.5, delta=.08)

    def test_columns_remain_paired_and_units_are_preserved(self):
        data = self.frame([.01, -.03, .02, .04, -.02, .00, .05])
        data["bps"] = 10_000 * data["excess"]
        data["shifted"] = data["excess"] + .125
        before = data.copy(deep=True)
        draws = bootstrap_block_means(data, block_length=3, n_bootstrap=200, seed=17)
        np.testing.assert_allclose(draws["bps"], 10_000 * draws["excess"], atol=1e-12)
        np.testing.assert_allclose(draws["shifted"], draws["excess"] + .125, atol=1e-12)
        np.testing.assert_allclose(
            draws["bps"].quantile([.025, .975]),
            10_000 * draws["excess"].quantile([.025, .975]),
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(data, before)
        pd.testing.assert_frame_equal(
            draws, bootstrap_block_means(data, block_length=3, n_bootstrap=200, seed=17)
        )

    def test_unknown_or_disordered_observations_are_rejected(self):
        data = self.frame([.01, .02, .03, .04])
        missing = data.copy()
        missing.iloc[1, 0] = np.nan
        infinite = data.copy()
        infinite.iloc[1, 0] = np.inf
        duplicated = data.copy()
        duplicated.index = [data.index[0], data.index[0], *data.index[2:]]
        for invalid in [missing, infinite, duplicated, data.iloc[::-1]]:
            with self.subTest(index=list(invalid.index)):
                with self.assertRaises(ValueError):
                    bootstrap_block_means(invalid, block_length=2, n_bootstrap=10)


if __name__ == "__main__":
    unittest.main()
