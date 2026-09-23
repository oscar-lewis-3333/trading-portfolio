"""GBP conversion must divide by dated, available FX without touching targets."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversal_valuation import build_development_gbp_inputs


class ReversalValuationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"development_start": "2022-01-01",
                         "development_end_exclusive": "2022-01-04",
                         "holdout_start": "2022-01-04"}
        self.days = pd.date_range("2022-01-01", periods=4, tz="UTC")
        self.prices = pd.DataFrame({"product_id": "BTC-USD", "timestamp": self.days,
                                    "open": [100., 110., 120., 999.],
                                    "close": [110., 120., 130., 999.]})
        self.references = pd.DataFrame({"product_id": "BTC-USD", "day": self.days[:3],
            "execution_at": self.days[:3] + pd.Timedelta(minutes=5),
            "reference_price_usd": [102., 112., 122.]})
        dates = pd.to_datetime(["2021-12-31", "2022-01-03"], utc=True)
        self.fx = pd.DataFrame({"reference_date": dates,
            "assumed_available_at": dates + pd.Timedelta(days=1), "usd_per_gbp": [2., 4.]})

    def build(self):
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Offline only")):
            return build_development_gbp_inputs(self.prices, self.references, self.fx, self.protocol)

    def test_division_weekend_carry_and_close_time_alignment(self):
        prices, refs, audit = self.build()
        self.assertEqual(prices.open_gbp.tolist(), [50., 55., 60.])
        # Monday's rate is first used at Tuesday midnight, Monday candle's close.
        self.assertEqual(prices.close_gbp.tolist(), [55., 60., 32.5])
        self.assertEqual(refs.reference_price_gbp.tolist(), [51., 56., 61.])
        self.assertEqual(audit.reference_age_days.tolist(), [1., 2., 3., 1.])
        self.assertEqual(len(prices), 3)  # Holdout candle was not valued.

    def test_future_fx_cannot_change_earlier_execution_marks(self):
        original = self.build()[1]
        self.fx.loc[1, "usd_per_gbp"] = 99.
        updated = self.build()[1]
        pd.testing.assert_frame_equal(original, updated)

    def test_stale_and_missing_rates_fail(self):
        self.fx["reference_date"] -= pd.Timedelta(days=10)
        self.fx["assumed_available_at"] -= pd.Timedelta(days=10)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.build()
        self.setUp()
        self.fx = self.fx.iloc[1:]
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.build()

    def test_invalid_rate_and_availability_fail(self):
        self.fx.loc[0, "usd_per_gbp"] = 0.
        with self.assertRaisesRegex(ValueError, "positive"):
            self.build()
        self.setUp()
        self.fx["assumed_available_at"] = self.fx["reference_date"]
        with self.assertRaisesRegex(ValueError, "next-midnight"):
            self.build()

    def test_holdout_and_duplicate_references_fail(self):
        self.references.loc[2, "execution_at"] = self.days[3]
        with self.assertRaisesRegex(ValueError, "inside development"):
            self.build()
        self.setUp()
        self.references = pd.concat([self.references, self.references.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.build()

    def test_inputs_unchanged(self):
        before = [f.copy(deep=True) for f in (self.prices, self.references, self.fx)]
        self.build()
        for old, current in zip(before, (self.prices, self.references, self.fx)):
            pd.testing.assert_frame_equal(old, current)


if __name__ == "__main__":
    unittest.main()
