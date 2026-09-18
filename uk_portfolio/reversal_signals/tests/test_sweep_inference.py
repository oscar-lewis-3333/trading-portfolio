
#file dedicated to ensuring the HAC tests, HOLM adjustments are as planned

import math
import sys
import unittest
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import reversal_validation as validation


def scalar_hac_mean(values, lags=5, correction=True):
    values = [float(value) for value in values]
    n = len(values)
    mean = math.fsum(values) / n
    residuals = [value - mean for value in values]
    terms = [math.fsum(value * value for value in residuals)]
    for lag in range(1, lags + 1):
        cross_products = math.fsum(residuals[t] * residuals[t - lag] for t in range(lag, n))
        terms.append(2. * (1. - lag / (lags + 1.)) * cross_products)
    variance = math.fsum(terms) / (n * n)
    if correction:
        variance *= n / (n - 1.)
    standard_error = math.sqrt(variance)
    z_stat = mean / standard_error
    p_value = .5 * math.erfc(z_stat / math.sqrt(2.))
    return mean, standard_error, z_stat, p_value


def scalar_holm(p_values):

    p_values = list(p_values)
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.] * len(order)
    previous = 0.
    for rank, position in enumerate(order):
        previous = min(1., max(previous, (len(order) - rank) * p_values[position]))
        adjusted[position] = previous
    return np.array(adjusted)


class SweepInferenceTests(unittest.TestCase):
    def setUp(self):
        self.rules = {
            "hac_lags": 5, "hac_kernel": "bartlett", "hac_use_correction": True,
            "hac_use_t": False, "alternative": "greater", "multiple_testing": "holm",
            "familywise_alpha": .05
        }
        innovations = np.random.default_rng(62).normal(0., .01, 80)
        base = np.zeros(80)
        for i in range(1, len(base)):
            base[i] = .6 * base[i - 1] + innovations[i]
        base -= math.fsum(base) / len(base)
        self.base = base
        self.returns = pd.DataFrame(
            {7: base + .004, 42: base - .004, 13: base},
            index=pd.date_range("2018-01-03", periods=80, freq="7D", name="Date")).rename_axis(columns="configuration_id")
        self.grid = pd.DataFrame({"variant": "synthetic"}, index=self.returns.columns)

    def infer(self, returns=None, grid=None, **changes):
        return validation.infer_sweep_means(self.returns if returns is None else returns,
            self.grid if grid is None else grid, dict(self.rules, **changes))

    def test_mean_hac_standard_error_and_tail_match_scalar_oracle(self):
        for lags in (0, 1, 5, 10):
            for correction in (False, True):
                actual = self.infer(hac_lags=lags, hac_use_correction=correction)
                for configuration_id in self.returns:
                    with self.subTest(lags=lags, correction=correction, configuration_id=configuration_id):
                        mean, se, z_stat, p_value = scalar_hac_mean(self.returns[configuration_id], lags, correction)
                        row = actual.loc[configuration_id]
                        np.testing.assert_allclose(
                            row[["mean_bps", "hac_se_bps", "z_stat", "p_one_sided"]].to_numpy(dtype=float),
                            [10_000 * mean, 10_000 * se, z_stat, p_value], rtol=1e-12, atol=1e-12)

    def test_confidence_limits_convert_the_entire_interval_to_basis_points(self):
        actual = self.infer()
        critical = NormalDist().inv_cdf(.975)
        expected = []
        for configuration_id in self.returns:
            mean, se, _, _ = scalar_hac_mean(self.returns[configuration_id])
            expected.append([10_000 * (mean - critical * se),
                             10_000 * (mean + critical * se)])
        np.testing.assert_allclose(actual[["ci_lower_bps", "ci_upper_bps"]], expected,rtol=1e-12, atol=1e-12)

    def test_one_sided_test_preserves_the_direction_of_the_mean(self):
        actual = self.infer()
        self.assertLess(actual.loc[7, "p_one_sided"], .5)
        self.assertGreater(actual.loc[42, "p_one_sided"], .5)
        self.assertAlmostEqual(actual.loc[13, "p_one_sided"], .5)
        self.assertAlmostEqual(actual.loc[7, "p_one_sided"] + actual.loc[42, "p_one_sided"], 1.)

    def returns_for_p_values(self, p_values):
        _, se, _, _ = scalar_hac_mean(self.base)
        data = {
            key: self.base + NormalDist().inv_cdf(1. - probability) * se
            for key, probability in p_values.items()
        }
        return pd.DataFrame(data, index=self.returns.index).rename_axis(columns="configuration_id")

    def test_holm_matches_manual_step_down_with_mixed_decisions(self):
        targets = {51: .04, 2: .0001, 7: .006, 19: .9, 4: .2}
        returns = self.returns_for_p_values(targets)
        grid = pd.DataFrame({"variant": "synthetic"}, index=returns.columns)
        actual = self.infer(returns, grid)
        np.testing.assert_allclose(actual["p_one_sided"], list(targets.values()), atol=1e-14)
        expected = scalar_holm(targets.values())
        np.testing.assert_allclose(actual["p_holm"], [.12, .0005, .024, .9, .4], atol=1e-14)
        np.testing.assert_allclose(actual["p_holm"], expected, atol=1e-14)
        np.testing.assert_array_equal(actual["reject_holm"], [False, True, True, False, False])

    def test_all_holm_values_can_equal_one_without_an_error(self):
        targets = dict(enumerate(np.linspace(.02, .9, 72)))
        returns = self.returns_for_p_values(targets)
        grid = pd.DataFrame({"variant": "synthetic"}, index=returns.columns)
        actual = self.infer(returns, grid)
        self.assertGreater(actual["p_one_sided"].min() * 72, 1.)
        self.assertTrue(actual["p_holm"].eq(1.).all())
        self.assertFalse(actual["reject_holm"].any())

    def test_column_order_is_preserved_and_inputs_are_not_mutated(self):
        before = self.returns.copy(deep=True)
        grid_before, rules_before = self.grid.copy(deep=True), dict(self.rules)
        expected = self.infer()
        order = [13, 7, 42]
        actual = self.infer(self.returns[order], self.grid.loc[order])
        pd.testing.assert_frame_equal(actual.loc[expected.index], expected)
        pd.testing.assert_frame_equal(self.returns, before)
        pd.testing.assert_frame_equal(self.grid, grid_before)
        self.assertEqual(self.rules, rules_before)

    def test_positive_rescaling_preserves_p_values_and_decisions(self):
        expected, actual = self.infer(), self.infer(self.returns * 7.)
        for column in ("mean_bps", "hac_se_bps", "ci_lower_bps", "ci_upper_bps"):
            np.testing.assert_allclose(actual[column], expected[column] * 7., atol=1e-12)
        for column in ("z_stat", "p_one_sided", "p_holm"):
            np.testing.assert_allclose(actual[column], expected[column], atol=1e-12)
        np.testing.assert_array_equal(actual["reject_holm"], expected["reject_holm"])

    def test_invalid_dates_columns_and_unknown_values_are_rejected(self):
        bad_matrices = [self.returns.iloc[:0], self.returns[7], self.returns.iloc[::-1],
                        self.returns.iloc[[0, 0, 1, 2, 3, 4, 5, 6]],
                        self.returns[[7, 42]], self.returns[[7, 7, 42]],
                        self.returns[[13, 7, 42]], self.returns.reset_index(drop=True)]
        missing_date = self.returns.copy()
        missing_date.index = pd.DatetimeIndex([pd.NaT, *self.returns.index[1:]])
        bad_matrices.append(missing_date)
        for value in (np.nan, np.inf, -np.inf):
            altered = self.returns.copy()
            altered.iloc[0, 0] = value
            bad_matrices.append(altered)
        for number, matrix in enumerate(bad_matrices):
            with self.subTest(case=number), self.assertRaises(ValueError):
                self.infer(matrix)
        with self.assertRaises(ValueError):
            self.infer(grid=self.grid.iloc[[0, 0, 2]])

    def test_invalid_inference_settings_are_rejected(self):
        cases = [
            {"hac_lags": value} for value in (True, -1, 1.5, len(self.returns) - 1)
        ] + [
            {"familywise_alpha": value} for value in (True, 0., 1., np.nan, np.inf, "0.05")
        ] + [
            {"alternative": "less"}, {"hac_use_t": True}, {"multiple_testing": "fdr_bh"},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.infer(**changes)

    def test_constant_series_requires_review_instead_of_a_p_value(self):
        for value in (0., .01):
            altered = self.returns.copy()
            altered[7] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.infer(altered)


if __name__ == "__main__":
    unittest.main()
