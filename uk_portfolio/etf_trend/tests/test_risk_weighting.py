"""Inverse-volatility weights, look-ahead, accounting, window selection, Gate A rule and the Sharpe bootstrap."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import etf_backtest
import etf_risk_weighting as rw


def random_prices(n_sessions=400, funds=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2020-01-01", periods=n_sessions)
    scales = np.linspace(0.005, 0.02, len(funds))
    returns = rng.normal(0.0002, scales, size=(n_sessions, len(funds)))
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=sessions, columns=list(funds))


class VolatilityTests(unittest.TestCase):
    def setUp(self):
        self.prices = random_prices()
        self.decision = self.prices.index[300]

    def test_minimum_observations_match_agreed_counts(self):
        self.assertEqual(rw.minimum_observations(63), 57)
        self.assertEqual(rw.minimum_observations(126), 114)

    def test_matches_direct_calculation(self):
        vol, _ = rw.trailing_volatility(self.prices, [self.decision], window=63)
        returns = self.prices.pct_change().iloc[300 - 62:301]
        expected = returns.std(ddof=1) * np.sqrt(252)
        np.testing.assert_allclose(vol.iloc[0].to_numpy(), expected.to_numpy(), rtol=1e-12)

    def test_no_look_ahead_but_decision_close_is_used(self):
        base, _ = rw.trailing_volatility(self.prices, [self.decision], window=126)
        later = self.prices.copy()
        later.iloc[301:] *= 1.5
        later.iloc[305, 0] *= 3
        after, _ = rw.trailing_volatility(later, [self.decision], window=126)
        pd.testing.assert_frame_equal(base, after)
        same_day = self.prices.copy()
        same_day.loc[self.decision, "AAA"] *= 1.05
        changed, _ = rw.trailing_volatility(same_day, [self.decision], window=126)
        self.assertNotEqual(changed.iloc[0]["AAA"], base.iloc[0]["AAA"])

    def test_missing_quotes_are_skipped_and_the_90pct_rule_applies(self):
        for gaps, available in ((3, True), (4, False)):
            prices = self.prices.copy()
            for k in range(gaps):
                prices.iloc[300 - 5 - 10 * k, 0] = np.nan
            vol, observed = rw.trailing_volatility(prices, [self.decision], window=63)
            self.assertEqual(observed.iloc[0]["AAA"], 63 - 2 * gaps)
            self.assertEqual(bool(np.isfinite(vol.iloc[0]["AAA"])), available)
            self.assertTrue(np.isfinite(vol.iloc[0]["BBB"]))

    def test_window_before_history_is_unavailable(self):
        vol, _ = rw.trailing_volatility(self.prices, [self.prices.index[62], self.prices.index[63]], window=63)
        self.assertTrue(vol.iloc[0].isna().all())
        self.assertTrue(vol.iloc[1].notna().all())


class WeightTests(unittest.TestCase):
    def setUp(self):
        dates = pd.DatetimeIndex(["2021-01-29", "2021-02-26"])
        self.vol = pd.DataFrame({"AAA": [0.10, 0.10], "BBB": [0.20, 0.20], "CCC": [0.40, 0.40]}, index=dates)
        self.signal = pd.DataFrame({"AAA": [1.0, 1.0], "BBB": [1.0, 0.0], "CCC": [1.0, 1.0]}, index=dates)

    def test_all_on_is_fully_invested_in_inverse_vol_proportions(self):
        weights, _ = rw.inverse_vol_weights(self.signal.iloc[:1], self.vol.iloc[:1])
        np.testing.assert_allclose(weights.iloc[0][["AAA", "BBB", "CCC"]], [4 / 7, 2 / 7, 1 / 7], rtol=1e-12)
        self.assertAlmostEqual(weights.iloc[0]["CASH"], 0.0, places=12)

    def test_off_fund_slot_goes_to_cash_without_rescaling(self):
        weights, _ = rw.inverse_vol_weights(self.signal, self.vol)
        row = weights.iloc[1]
        np.testing.assert_allclose(row[["AAA", "BBB", "CCC"]], [4 / 7, 0.0, 1 / 7], rtol=1e-12)
        self.assertAlmostEqual(row["CASH"], 2 / 7, places=12)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)

    def test_unavailable_fund_keeps_equal_slot_in_cash(self):
        vol = self.vol.copy()
        vol.iloc[0, 2] = np.nan
        weights, unavailable = rw.inverse_vol_weights(self.signal, vol)
        row = weights.iloc[0]
        #two available funds share 2/3 by inverse vol
        np.testing.assert_allclose(row[["AAA", "BBB", "CCC"]], [4 / 9, 2 / 9, 0.0], rtol=1e-12)
        self.assertAlmostEqual(row["CASH"], 1 / 3, places=12)
        self.assertTrue(unavailable.iloc[0]["CCC"])
        self.assertEqual(len(rw.unavailable_events(unavailable)), 1)

    def test_constant_mix_ignores_signal(self):
        weights, _ = rw.constant_mix_weights(self.vol)
        np.testing.assert_allclose(weights[["AAA", "BBB", "CCC"]].to_numpy(), [[4 / 7, 2 / 7, 1 / 7]] * 2, rtol=1e-12)

    def test_equal_vols_reduce_to_the_original_flat_slots(self):
        vol = pd.DataFrame(0.15, index=self.vol.index, columns=self.vol.columns)
        weights, _ = rw.inverse_vol_weights(self.signal, vol)
        pd.testing.assert_frame_equal(weights, etf_backtest.build_target_weights(self.signal), rtol=0, atol=1e-15)

    def test_execution_dates_shift_to_the_next_session(self):
        schedule = pd.DataFrame({"execution_date": pd.DatetimeIndex(["2021-02-01", "2021-03-01"])}, index=self.vol.index)
        weights, _ = rw.inverse_vol_weights(self.signal, self.vol)
        shifted = rw.to_execution_weights(weights, schedule)
        self.assertEqual(list(shifted.index), list(schedule["execution_date"]))
        np.testing.assert_allclose(shifted.to_numpy(), weights.to_numpy())


class AccountingTests(unittest.TestCase):
    def test_costs_fall_on_net_fund_trades_only(self):
        values = pd.Series({"AAA": 6_000.0, "BBB": 2_000.0, "CASH": 2_000.0})
        target = pd.Series({"AAA": 0.3, "BBB": 0.5, "CASH": 0.2})
        result = etf_backtest.rebalance_portfolio(values, target, cost_per_side_bps=10.0)
        self.assertAlmostEqual(result["cost"], 0.001 * result["traded_notional"], places=10)
        self.assertAlmostEqual(result["traded_notional"], result["trades"].abs().sum(), places=10)
        self.assertNotIn("CASH", result["trades"].index)
        self.assertAlmostEqual(result["values"].sum() + result["cost"], values.sum(), places=8)
        self.assertAlmostEqual(result["values"].sum(), result["nav_after"], places=8)

    def test_traded_fraction_excludes_initial_funding(self):
        daily = pd.DataFrame({
            "rebalanced": [True, False, True],
            "traded_notional": [10_000.0, 0.0, 2_000.0],
            "nav_open_before_cost": [10_000.0, 10_000.0, 10_000.0],
        }, index=pd.DatetimeIndex(["2021-01-04", "2021-06-01", "2022-01-03"]))
        years = (daily.index[-1] - daily.index[0]).days / 365.25 + 1 / 365.25
        self.assertAlmostEqual(rw.annual_traded_fraction(daily), 0.2 / years, places=12)


class SelectionAndGateTests(unittest.TestCase):
    def frame(self, sharpe, turnover):
        return pd.DataFrame({"sharpe_excess": sharpe, "traded_pct_nav_pa": turnover}, index=[63, 126])

    def test_clear_sharpe_winner(self):
        self.assertEqual(rw.select_vol_window(self.frame([0.60, 0.50], [90, 60]))[0], 63)

    def test_close_sharpe_goes_to_lower_turnover(self):
        self.assertEqual(rw.select_vol_window(self.frame([0.54, 0.50], [60, 90]))[0], 63)
        self.assertEqual(rw.select_vol_window(self.frame([0.55, 0.50], [60, 90]))[0], 63)
        self.assertEqual(rw.select_vol_window(self.frame([0.50, 0.53], [60, 90]))[0], 63)

    def test_full_tie_defaults_to_126(self):
        self.assertEqual(rw.select_vol_window(self.frame([0.50, 0.52], [70, 70]))[0], 126)

    def test_gate_a_rule(self):
        self.assertEqual(rw.gate_a_verdict(0.15, 0.11), "helped")
        self.assertEqual(rw.gate_a_verdict(-0.2, -0.3), "hurt")
        self.assertEqual(rw.gate_a_verdict(0.3, 0.05), "no difference")
        self.assertEqual(rw.gate_a_verdict(0.3, -0.3), "no difference")
        self.assertEqual(rw.gate_a_verdict(0.10, 0.5), "no difference")


class SharpeBootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        months = pd.period_range("2019-01", periods=48, freq="M")
        self.benchmark = pd.Series(rng.normal(0.004, 0.02, 48), index=months)

    def test_identical_series_give_zero_difference(self):
        result = rw.paired_sharpe_bootstrap(self.benchmark, self.benchmark, n_bootstrap=500)
        self.assertEqual(result["sharpe_difference"], 0.0)
        self.assertEqual(result["lower_ci"], 0.0)
        self.assertEqual(result["upper_ci"], 0.0)

    def test_observed_values_and_reproducibility(self):
        strategy = self.benchmark * 0.5 + 0.003
        first = rw.paired_sharpe_bootstrap(strategy, self.benchmark, n_bootstrap=2_000)
        second = rw.paired_sharpe_bootstrap(strategy, self.benchmark, n_bootstrap=2_000)
        pd.testing.assert_series_equal(first, second)
        expected = rw.sharpe_ratio(strategy) - rw.sharpe_ratio(self.benchmark)
        self.assertAlmostEqual(first["sharpe_difference"], expected, places=12)
        self.assertLess(first["lower_ci"], first["sharpe_difference"])
        self.assertGreater(first["upper_ci"], first["sharpe_difference"])
        self.assertLess(first["one_sided_p_value"], 0.05)

    def test_sharpe_is_scale_invariant_so_leverage_changes_nothing(self):
        result = rw.paired_sharpe_bootstrap(self.benchmark * 2, self.benchmark, n_bootstrap=500)
        self.assertAlmostEqual(result["sharpe_difference"], 0.0, places=12)

    def test_rejects_gaps_and_mismatches(self):
        with self.assertRaises(ValueError):
            rw.paired_sharpe_bootstrap(self.benchmark.drop(self.benchmark.index[5]), self.benchmark.drop(self.benchmark.index[5]))
        with self.assertRaises(ValueError):
            rw.paired_sharpe_bootstrap(self.benchmark.iloc[1:], self.benchmark.iloc[:-1])


class RiskShareTests(unittest.TestCase):
    def test_shares_sum_to_one_and_cash_only_is_blank(self):
        prices = random_prices(n_sessions=300)
        dates = prices.index[[200, 250]]
        weights = pd.DataFrame({"AAA": [0.3, 0.0], "BBB": [0.3, 0.0], "CCC": [0.2, 0.0], "CASH": [0.2, 1.0]}, index=dates)
        groups = {"AAA": "equity", "BBB": "equity", "CCC": "bond"}
        shares = rw.class_risk_shares(weights, prices, groups, window=63)
        self.assertAlmostEqual(shares.iloc[0][["bond", "equity"]].sum(), 1.0, places=12)
        self.assertAlmostEqual(shares.iloc[0]["invested_pct"], 80.0, places=12)
        self.assertTrue(shares.iloc[1][["bond", "equity"]].isna().all())


class ComparePairsTests(unittest.TestCase):
    def test_table_has_both_comparisons_and_all_blocks(self):
        rng = np.random.default_rng(5)
        months = pd.period_range("2019-01", periods=40, freq="M")
        excess = pd.DataFrame(rng.normal(0.003, 0.02, size=(40, 3)), index=months, columns=["inverse_vol_trend", "equal_weight_trend", "inverse_vol_constant_mix"])
        table = rw.compare_pairs(excess, n_bootstrap=300)
        self.assertEqual(len(table), 6)
        self.assertEqual(int(table["primary"].sum()), 2)
        self.assertEqual(sorted(table.index.get_level_values("block_months").unique()), [2, 3, 6])


if __name__ == "__main__":
    unittest.main()