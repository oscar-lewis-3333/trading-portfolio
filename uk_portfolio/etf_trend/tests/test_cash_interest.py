"""Cash accrual, Bank Rate parsing and the cash-return series used as the Sharpe risk-free rate."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import etf_backtest
import etf_cash
import etf_risk_weighting

CAPITAL = 10_000.0


def price_panel(sessions, values=None):
    if values is None:
        values = {"AAA": np.linspace(100, 110, len(sessions)), "BBB": np.linspace(50, 45, len(sessions))}
    close = pd.DataFrame(values, index=sessions)
    return close * 0.999, close


class CashAccrualTests(unittest.TestCase):
    def setUp(self):
        #02/01/2025-31/01/2025
        self.sessions = pd.bdate_range("2025-01-02", "2025-01-31")

    def test_zero_scalar_gives_unit_factors(self):
        factors = etf_backtest.cash_accrual_factors(0.0, self.sessions)
        self.assertTrue(factors.eq(1.0).all())

    def test_scalar_counts_calendar_days_including_weekends(self):
        factors = etf_backtest.cash_accrual_factors(0.0365, self.sessions)
        self.assertEqual(factors.iloc[0], 1.0)
        #Fri 3 Jan -> Mon 6 Jan earns three days
        self.assertAlmostEqual(factors.loc["2025-01-06"], 1 + 0.0365 * 3 / 365, places=15)
        self.assertAlmostEqual(factors.loc["2025-01-07"], 1 + 0.0365 * 1 / 365, places=15)

    def test_constant_series_matches_scalar(self):
        rates = pd.Series(0.05, index=pd.bdate_range("2024-12-01", "2025-02-28"))
        pd.testing.assert_series_equal(
            etf_backtest.cash_accrual_factors(rates, self.sessions),
            etf_backtest.cash_accrual_factors(0.05, self.sessions),
            check_names=False, rtol=0, atol=1e-15)

    def test_rate_change_applies_from_its_effective_day(self):
        #rate rises from 4% to 5% effective Monday 13 Jan; Fri 10 -> Mon 13 earns three days at 4%
        rates = pd.Series(0.04, index=pd.bdate_range("2024-12-01", "2025-02-28"))
        rates.loc["2025-01-13":] = 0.05
        factors = etf_backtest.cash_accrual_factors(rates, self.sessions)
        self.assertAlmostEqual(factors.loc["2025-01-13"], 1 + 0.04 * 3 / 365, places=15)
        self.assertAlmostEqual(factors.loc["2025-01-14"], 1 + 0.05 / 365, places=15)

    def test_rates_must_cover_the_sessions(self):
        late_start = pd.Series(0.05, index=pd.bdate_range("2025-01-10", "2025-02-28"))
        with self.assertRaises(ValueError):
            etf_backtest.cash_accrual_factors(late_start, self.sessions)
        stale = pd.Series(0.05, index=pd.bdate_range("2024-12-01", "2025-01-15"))
        with self.assertRaises(ValueError):
            etf_backtest.cash_accrual_factors(stale, self.sessions)

    def test_all_cash_portfolio_compounds_the_factors(self):
        open_prices, close_prices = price_panel(self.sessions)
        weights = pd.DataFrame({"AAA": [0.0], "BBB": [0.0], "CASH": [1.0]}, index=self.sessions[:1])
        result = etf_backtest.run_backtest(open_prices, close_prices, weights, initial_capital_gbp=CAPITAL, cost_per_side_bps=10.0, cash_rate_annual=0.05)
        expected = CAPITAL * etf_backtest.cash_accrual_factors(0.05, self.sessions).prod()
        self.assertAlmostEqual(result["daily"]["nav_close"].iloc[-1], expected, places=8)
        self.assertAlmostEqual(result["daily"]["interest"].sum(), expected - CAPITAL, places=8)
        self.assertEqual(result["daily"]["cost"].sum(), 0.0)

    def test_all_cash_portfolio_has_zero_excess_return(self):
        open_prices, close_prices = price_panel(self.sessions)
        rates = pd.Series(0.04, index=pd.bdate_range("2024-12-01", "2025-02-28"))
        rates.loc["2025-01-16":] = 0.045
        weights = pd.DataFrame({"AAA": [0.0], "BBB": [0.0], "CASH": [1.0]}, index=self.sessions[:1])
        result = etf_backtest.run_backtest(open_prices, close_prices, weights, initial_capital_gbp=CAPITAL, cost_per_side_bps=10.0, cash_rate_annual=rates)
        cash_monthly = etf_cash.monthly_cash_returns(rates, self.sessions)
        excess = etf_risk_weighting.monthly_excess_returns({"cash_only": result}, cash_monthly)
        self.assertLess(excess.abs().to_numpy().max(), 1e-12)

    def test_interest_is_credited_before_the_rebalance(self):
        #half cash for one night at 36.5% (0.1% a day), then fully invest: costs fall on the accrued cash
        sessions = pd.DatetimeIndex(["2025-01-06", "2025-01-07"])
        open_prices = pd.DataFrame({"AAA": [100.0, 100.0]}, index=sessions)
        close_prices = open_prices.copy()
        weights = pd.DataFrame({"AAA": [0.5, 1.0], "CASH": [0.5, 0.0]}, index=sessions)
        result = etf_backtest.run_backtest(open_prices, close_prices, weights, initial_capital_gbp=CAPITAL, cost_per_side_bps=0.0, cash_rate_annual=0.365)
        day_two = result["daily"].iloc[1]
        self.assertAlmostEqual(day_two["interest"], 5_000.0 * 0.001, places=9)
        self.assertAlmostEqual(day_two["traded_notional"], 5_005.0, places=9)
        self.assertAlmostEqual(day_two["cash"], 0.0, places=9)

    def test_zero_scalar_and_zero_series_are_identical(self):
        open_prices, close_prices = price_panel(self.sessions)
        weights = pd.DataFrame({"AAA": [0.3, 0.6], "BBB": [0.3, 0.1], "CASH": [0.4, 0.3]}, index=self.sessions[[0, 10]])
        zero_series = pd.Series(0.0, index=pd.bdate_range("2024-12-01", "2025-02-28"))
        runs = [etf_backtest.run_backtest(open_prices, close_prices, weights, initial_capital_gbp=CAPITAL, cost_per_side_bps=10.0, cash_rate_annual=rate) for rate in (0.0, zero_series)]
        pd.testing.assert_frame_equal(runs[0]["daily"], runs[1]["daily"], rtol=0, atol=0)


class BankRateParsingTests(unittest.TestCase):
    CSV = "DATE,IUDBEDR\n02 Jan 2025,4.75\n03 Jan 2025,4.75\n06 Feb 2025,4.5\n"

    def test_parse_returns_decimal_rates(self):
        rates = etf_cash.parse_boe_csv(self.CSV)
        self.assertEqual(list(rates), [0.0475, 0.0475, 0.045])
        self.assertEqual(rates.index[2], pd.Timestamp("2025-02-06"))

    def test_html_error_page_is_rejected(self):
        with self.assertRaises(ValueError):
            etf_cash.parse_boe_csv("<html><body>Access denied</body></html>")

    def test_manual_file_is_cached_with_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "download.csv"
            source.write_text(self.CSV)
            cache = Path(directory) / "cash"
            rates, metadata = etf_cash.load_bank_rate(cache, start="2025-01-01", end="2025-03-01", source_file=source)
            self.assertEqual(metadata["observations"], 3)
            self.assertIn("manual download", metadata["source"])
            again, _ = etf_cash.load_bank_rate(cache, start="2025-01-01", end="2025-03-01")
            pd.testing.assert_series_equal(rates, again)

    def test_scenarios_floor_at_zero(self):
        rates = pd.Series([0.001, 0.0525], index=pd.DatetimeIndex(["2020-03-19", "2023-08-03"]))
        scenarios = etf_cash.build_cash_rate_scenarios(rates)
        self.assertEqual(scenarios["zero"], 0.0)
        np.testing.assert_allclose(scenarios["bank_rate_minus_50bp"], [0.0, 0.0475], rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()