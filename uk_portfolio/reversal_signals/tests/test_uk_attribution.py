

#ensure profit/loss accumulate properly
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from reversal_validation import attribute_stock_pnl


class StockAttributionTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        self.dates.name = "Date"
        d0, d1, d2, d3 = self.dates
        # A: buy 10, sell 4, sell 6. B: buy 5, sell 5. Costs are 1%.
        orders = pd.DataFrame([
            (d1, "A.L", "buy", "assumed_fill", 100., 1., -101.),
            (d1, "B.L", "buy", "assumed_fill", 100., 1., -101.),
            (d2, "A.L", "sell", "assumed_fill", 36., .36, 35.64),
            (d3, "A.L", "sell", "assumed_fill", 72., .72, 71.28),
            (d3, "B.L", "sell", "assumed_fill", 90., .90, 89.10),
        ], columns=["Date", "ticker", "side", "status", "notional_gbp", "cost_gbp", "cash_change_gbp"])
        holdings = pd.DataFrame([
            (d1, "A.L", 10., 100.), (d1, "B.L", 5., 100.),
            (d2, "A.L", 6., 54.), (d2, "B.L", 5., 105.),
        ], columns=["Date", "ticker", "units", "value_gbp"]).set_index(["Date", "ticker"])
        self.result = {
            "orders": orders,
            "holdings": holdings,
            "daily": pd.DataFrame({"equity_gbp": [1000., 998., 1012.64, 1018.02]}, index=self.dates),
        }
        index = pd.MultiIndex.from_product([self.dates, ["A.L", "B.L"]], names=["Date", "ticker"])
        # Large entry-date dividends must not be received by new purchasers.
        self.market = pd.DataFrame({"dividend_gbp": [0., 0., 777., 888., 2., 0., .5, .2]}, index=index)

    def partial_result(self):
        end = self.dates[2]
        return {
            "daily": self.result["daily"].loc[:end].copy(),
            "holdings": self.result["holdings"].loc[:end].copy(),
            "orders": self.result["orders"].loc[lambda frame: frame["Date"] <= end].copy(),
        }

    def test_closed_stock_profits_and_costs_match_hand_calculation(self):
        table = attribute_stock_pnl(self.result, self.market, 1000.)
        # A: -101 + 35.64 + 71.28 + 10*2 + 6*.5 = 28.92.
        # B: -101 + 89.10 + 5*.2 = -10.90.
        np.testing.assert_allclose(table.loc[["A.L", "B.L"], "net_pnl_gbp"], [28.92, -10.90])
        np.testing.assert_allclose(table.loc[["A.L", "B.L"], "dividends_gbp"], [23., 1.])
        np.testing.assert_allclose(table.loc[["A.L", "B.L"], "cost_gbp"], [2.08, 1.90])
        np.testing.assert_allclose(table.loc[["A.L", "B.L"], "traded_notional_gbp"], [208., 190.])
        self.assertTrue(table["ending_value_gbp"].eq(0.).all())

    def test_unsold_positions_include_their_ending_value(self):
        table = attribute_stock_pnl(self.partial_result(), self.market, 1000.)
        # A: -101 + 35.64 + 20 + 54 = 8.64. B: -101 + 105 = 4.
        np.testing.assert_allclose(table.loc[["A.L", "B.L"], "net_pnl_gbp"], [8.64, 4.])
        np.testing.assert_allclose(table.loc[["A.L", "B.L"], "ending_value_gbp"], [54., 105.])

    def test_missing_dividend_fails_only_when_shares_were_owned_overnight(self):
        missing = self.market.copy()
        missing.loc[(self.dates[1], "A.L"), "dividend_gbp"] = np.nan
        expected = attribute_stock_pnl(self.result, self.market, 1000.)
        pd.testing.assert_frame_equal(attribute_stock_pnl(self.result, missing, 1000.), expected)
        missing.loc[(self.dates[2], "A.L"), "dividend_gbp"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing dividend"):
            attribute_stock_pnl(self.result, missing, 1000.)

    def test_future_dividends_do_not_change_an_earlier_endpoint(self):
        partial = self.partial_result()
        expected = attribute_stock_pnl(partial, self.market, 1000.)
        altered = self.market.copy()
        altered.loc[(self.dates[3], slice(None)), "dividend_gbp"] = 999.
        pd.testing.assert_frame_equal(attribute_stock_pnl(partial, altered, 1000.), expected)

    def test_inconsistent_cash_flow_is_detected(self):
        altered = copy.deepcopy(self.result)
        altered["orders"].loc[0, "cash_change_gbp"] += 1.
        with self.assertRaisesRegex(ValueError, "do not reconcile"):
            attribute_stock_pnl(altered, self.market, 1000.)

    def test_cash_only_backtest_has_empty_attribution(self):
        flat = {
            "daily": pd.DataFrame({"equity_gbp": 1000.}, index=self.dates),
            "holdings": self.result["holdings"].iloc[:0].copy(),
            "orders": self.result["orders"].iloc[:0].copy(),
        }
        self.assertTrue(attribute_stock_pnl(flat, self.market, 1000.).empty)

    def test_inputs_are_preserved(self):
        before = copy.deepcopy(self.result)
        market_before = self.market.copy(deep=True)
        attribute_stock_pnl(self.result, self.market, 1000.)
        for key in self.result:
            pd.testing.assert_frame_equal(self.result[key], before[key])
        pd.testing.assert_frame_equal(self.market, market_before)


if __name__ == "__main__":
    unittest.main()
