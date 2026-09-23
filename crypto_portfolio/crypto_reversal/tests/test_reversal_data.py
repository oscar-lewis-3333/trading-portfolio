"""Independent membership checks; no network or market-return assumptions."""
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reversal_data


class DailyUniverseTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "development_start": "2022-01-03",
            "development_end_exclusive": "2022-01-08",
            "eligibility": {"history_days": 2, "liquidity_days": 2,
                            "min_median_dollar_volume": 50.0},
            "universe": {"min_assets": 2, "max_assets": 2},
            "excluded_products": ["MEME-USD"],
        }
        self.prices = pd.DataFrame([
            {"product_id": product, "timestamp": day,
             "open": 10., "high": 10., "low": 10., "close": 10.,
             "volume": volume}
            for product, volume in [("B-USD", 10.), ("A-USD", 10.),
                                    ("C-USD", 8.), ("MEME-USD", 100.)]
            for day in pd.date_range("2022-01-01", "2022-01-07", tz="UTC")
        ])

    def test_completed_candles_warmup_ties_and_exclusions(self):
        _, members, coverage = reversal_data.build_daily_universe(self.prices, self.protocol)
        self.assertFalse(coverage.loc["2022-01-03", "active"])
        first = members.loc[members["decision_at"].eq(pd.Timestamp("2022-01-04", tz="UTC"))]
        self.assertEqual(first["product_id"].tolist(), ["A-USD", "B-USD"])
        self.assertEqual(first["liquidity_rank"].tolist(), [1, 2])
        self.assertTrue((first["timestamp"] == pd.Timestamp("2022-01-03", tz="UTC")).all())
        self.assertEqual(coverage.loc["2022-01-04", "after_meme_exclusions"], 3)
        self.assertEqual(coverage.loc["2022-01-04", "universe_size"], 2)

    def test_gap_restarts_history_and_is_not_forward_filled(self):
        prices = self.prices.loc[~(
            self.prices["product_id"].eq("A-USD")
            & self.prices["timestamp"].eq(pd.Timestamp("2022-01-03", tz="UTC"))
        )]
        _, members, _ = reversal_data.build_daily_universe(prices, self.protocol)
        a_dates = members.loc[members["product_id"].eq("A-USD"), "decision_at"]
        self.assertEqual(a_dates.tolist(), [pd.Timestamp("2022-01-07", tz="UTC")])

    def test_missing_calendar_day_is_inactive(self):
        prices = self.prices.loc[self.prices["timestamp"].ne(pd.Timestamp("2022-01-04", tz="UTC"))]
        _, members, coverage = reversal_data.build_daily_universe(prices, self.protocol)
        self.assertEqual(len(coverage), 5)
        self.assertFalse(coverage.loc["2022-01-05", "active"])
        self.assertEqual(coverage.loc["2022-01-05", "universe_size"], 0)
        self.assertFalse(members["decision_at"].eq(pd.Timestamp("2022-01-05", tz="UTC")).any())

    def test_future_mutation_cannot_change_earlier_members(self):
        _, before, _ = reversal_data.build_daily_universe(self.prices, self.protocol)
        prices = self.prices.copy()
        cutoff = pd.Timestamp("2022-01-05", tz="UTC")
        prices.loc[prices["timestamp"].ge(cutoff) & prices["product_id"].eq("C-USD"), "volume"] = 10000.
        _, after, _ = reversal_data.build_daily_universe(prices, self.protocol)
        pd.testing.assert_frame_equal(
            before.loc[before["decision_at"].le(cutoff)].reset_index(drop=True),
            after.loc[after["decision_at"].le(cutoff)].reset_index(drop=True),
        )

    def test_inputs_unchanged_and_invalid_breadth_rejected(self):
        original = self.prices.copy(deep=True)
        original_protocol = deepcopy(self.protocol)
        reversal_data.build_daily_universe(self.prices, self.protocol)
        pd.testing.assert_frame_equal(original, self.prices)
        self.assertEqual(original_protocol, self.protocol)
        self.protocol["universe"]["min_assets"] = 3
        with self.assertRaisesRegex(ValueError, "min_assets"):
            reversal_data.build_daily_universe(self.prices, self.protocol)

    def test_loader_rejects_bad_snapshot_and_invalid_prices(self):
        protocol = dict(self.protocol, history_snapshot="unused", history_start="2022-01-01",
                        history_end_exclusive="2022-01-08", quote_currency="USD")
        spec = dict(source="coinbase_exchange", quote_currency="USD", start="2022-01-01",
                    end_exclusive="2022-01-08")
        with patch.object(reversal_data, "load_saved_daily_prices", return_value=(self.prices, dict(spec, quote_currency="GBP"))):
            with self.assertRaisesRegex(ValueError, "declared protocol"):
                reversal_data.load_reversal_history(".", protocol)
        bad = self.prices.copy()
        bad.loc[0, "close"] = 11.  # Outside this candle's high-low range.
        with patch.object(reversal_data, "load_saved_daily_prices", return_value=(bad, spec)):
            with self.assertRaisesRegex(ValueError, "invalid daily candles"):
                reversal_data.load_reversal_history(".", protocol)


if __name__ == "__main__":
    unittest.main()
