import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import reversal_features


def make_schedule(dates):
    sessions = pd.DatetimeIndex(dates, name="Date")
    local = sessions.tz_localize("America/New_York")
    return pd.DataFrame({
        "market_open": (local + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC"),
        "market_close": (local + pd.Timedelta(hours=16)).tz_convert("UTC"),
    }, index=sessions)



class ForwardOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.schedule = make_schedule([
            "2020-07-01", "2020-07-02", "2020-07-06", "2020-07-07",
            "2020-07-08", "2020-07-09", "2020-07-10", "2020-07-13",
        ])
        self.panel = pd.DataFrame({
            "ticker": "AAA",
            "Open": [100., 100., 110., 105., 108., 112., 115., 120.],
            "Dividends": [0., 9., 2., 0., 1., 0., 3., 0.],
        }, index=self.schedule.index)

    def test_session_timing_and_dividend_entitlement(self):
        # The entry-day dividend of 9 is excluded; exit-day dividends count.
        for horizon, dividends, total_return in [(1, 2., .12), (3, 3., .11), (5, 6., .21)]:
            with self.subTest(horizon=horizon):
                row = reversal_features.build_forward_outcomes(
                    self.panel, self.schedule, horizon=horizon
                ).iloc[0]
                self.assertEqual(row["signal_time"], self.schedule["market_close"].iloc[0])
                self.assertEqual(row["entry_time"], self.schedule["market_open"].iloc[1])
                self.assertEqual(row["exit_time"], self.schedule["market_open"].iloc[horizon + 1])
                self.assertEqual(row["dividends_earned"], dividends)
                self.assertAlmostEqual(row["forward_return"], total_return)

    def test_unfinished_horizons_remain_unknown(self):
        for horizon in (1, 3, 5):
            with self.subTest(horizon=horizon):
                result = reversal_features.build_forward_outcomes(
                    self.panel, self.schedule, horizon=horizon
                )
                self.assertTrue(result.iloc[:-(horizon + 1)]["outcome_observed"].all())
                tail = result.iloc[-(horizon + 1):]
                self.assertFalse(tail["outcome_observed"].any())
                self.assertTrue(tail["forward_return"].isna().all())

    def test_missing_endpoint_or_intervening_dividend_is_unknown(self):
        for column, position in [("Open", 1), ("Open", 4), ("Dividends", 2)]:
            with self.subTest(column=column, position=position):
                damaged = self.panel.copy()
                damaged.loc[damaged.index[position], column] = np.nan
                row = reversal_features.build_forward_outcomes(
                    damaged, self.schedule, horizon=3
                ).iloc[0]
                self.assertFalse(row["outcome_observed"])
                self.assertTrue(pd.isna(row["forward_return"]))

    def test_unused_interior_open_does_not_erase_known_return(self):
        damaged = self.panel.copy()
        damaged.loc[damaged.index[2], "Open"] = np.nan
        row = reversal_features.build_forward_outcomes(
            damaged, self.schedule, horizon=3
        ).iloc[0]
        self.assertTrue(row["outcome_observed"])
        self.assertAlmostEqual(row["forward_return"], .11)

    def test_ticker_isolation_and_price_unit_invariance(self):
        scaled = self.panel.copy()
        scaled["ticker"] = "BBB"
        scaled[["Open", "Dividends"]] *= 7
        mixed = pd.concat([self.panel, scaled]).sample(frac=1, random_state=12)
        before = mixed.copy(deep=True)
        for horizon in (1, 3, 5):
            result = reversal_features.build_forward_outcomes(
                mixed, self.schedule, horizon=horizon
            )
            expected = reversal_features.build_forward_outcomes(
                self.panel, self.schedule, horizon=horizon
            )["forward_return"]
            for ticker in ("AAA", "BBB"):
                pd.testing.assert_series_equal(
                    result.loc[result["ticker"].eq(ticker), "forward_return"], expected
                )
        pd.testing.assert_frame_equal(mixed, before)



if __name__ == "__main__":
    unittest.main()
