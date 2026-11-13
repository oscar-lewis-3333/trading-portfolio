"""Independent timing, missing-data, provenance and offline-replay checks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from execution_resolution import build_execution_data, save_execution_data, load_execution_data


class ExecutionResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / "old"
        self.new = self.root / "new"
        self.old.mkdir()
        self.new.mkdir()
        self.day = pd.Timestamp("2023-07-03", tz="UTC")
        self.protocol = {"development_start": "2022-01-01",
                         "development_end_exclusive": "2025-01-01", "holdout_start": "2025-01-01"}

    def path(self, product, minutes=15, directory=None):
        suffix = "0015" if minutes == 15 else "0100"
        return (directory or self.old) / f"{product}_20230703_0000_{suffix}_60s.json"

    def save(self, product, bars, minutes=15, directory=None):
        payload = [[int(self.day.timestamp()) + m * 60, price, price, price, price, volume]
                   for m, price, volume in bars]
        record = {"source": "coinbase_exchange", "http_status": 200, "error": None,
                  "request": {"product_id": product,
                              "url": f"https://api.exchange.coinbase.com/products/{product}/candles",
                              "params": {"granularity": 60, "start": self.day.isoformat(),
                                         "end": (self.day + pd.Timedelta(minutes=minutes)).isoformat()}},
                  "response_text": json.dumps(payload)}
        path = self.path(product, minutes, directory)
        path.write_text(json.dumps(record))
        return path

    def plan(self, products):
        return pd.DataFrame([{"product_id": p, "day": self.day,
                              "earliest_execution_at": self.day + pd.Timedelta(minutes=5),
                              "cache_path": str(self.path(p)),
                              "download_path": str(self.path(p, directory=self.new))}
                             for p in products])

    def build(self, products):
        with patch("intraday.requests.get", side_effect=AssertionError("No network")):
            return build_execution_data(self.plan(products), self.protocol, self.new, [self.old])

    def test_0005_uses_only_completed_positive_volume_candles(self):
        self.save("BTC-USD", [(2, 101, 1), (4, 999, 0), (5, 888, 1)])
        b = self.build(["BTC-USD"])
        r = b["references"].iloc[0]
        self.assertEqual(r.reference_price_usd, 101)
        self.assertEqual(r.age_upper_bound_minutes, 3)
        self.assertEqual(r.execution_at, self.day + pd.Timedelta(minutes=5))

    def test_common_time_refreshes_other_assets_and_keeps_exits(self):
        # A is available at 00:06; B's 00:00 bar is then too old.
        # Wait until B's 00:07 bar is complete at 00:08, refreshing both marks.
        self.save("APT-USD", [(5, 10, 1)])
        self.save("BTC-USD", [(0, 100, 1), (7, 120, 1)])
        b = self.build(["APT-USD", "BTC-USD"])
        self.assertTrue(b["references"].execution_at.eq(self.day + pd.Timedelta(minutes=8)).all())
        self.assertEqual(set(b["references"].product_id), {"APT-USD", "BTC-USD"})
        self.assertEqual(b["references"].set_index("product_id").loc["BTC-USD", "reference_price_usd"], 120)

    def test_extended_cache_reuse_and_exact_overlap(self):
        self.save("APT-USD", [(16, 999, 1)])  # Excluded extra response row.
        self.save("APT-USD", [(16, 90, 1)], 60)
        b = self.build(["APT-USD"])
        self.assertEqual(b["references"].iloc[0].execution_at, self.day + pd.Timedelta(minutes=17))
        self.assertEqual(b["summary"].new_hour_requests, 0)
        self.assertTrue(b["schedule"].iloc[0].used_hour_window)

    def test_changed_overlap_rejected_without_rewriting_original(self):
        path = self.save("APT-USD", [(10, 10, 0)])
        original = path.read_bytes()
        self.save("APT-USD", [(10, 11, 0), (16, 12, 1)], 60)
        with self.assertRaisesRegex(ValueError, "Overlapping captures differ"):
            self.build(["APT-USD"])
        self.assertEqual(path.read_bytes(), original)

    def test_absent_or_unusable_hour_fails_without_fabricating_prices(self):
        self.save("APT-USD", [])
        with self.assertRaises(FileNotFoundError):
            self.build(["APT-USD"])
        self.save("APT-USD", [], 60)
        with self.assertRaisesRegex(ValueError, "No common fresh reference"):
            self.build(["APT-USD"])

    def test_duplicate_candles_and_bad_metadata_are_rejected(self):
        self.save("APT-USD", [(4, 10, 1), (4, 10, 1)])
        with self.assertRaisesRegex(ValueError, "Invalid or duplicate candle"):
            self.build(["APT-USD"])
        path = self.save("APT-USD", [(4, 10, 1)])
        record = json.loads(path.read_text())
        record["request"]["product_id"] = "BTC-USD"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.build(["APT-USD"])

    def test_holdout_is_rejected_before_accessing_files(self):
        plan = self.plan(["APT-USD"])
        plan["day"] = pd.Timestamp("2025-01-01", tz="UTC")
        plan["earliest_execution_at"] = plan["day"] + pd.Timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError, "development"):
            build_execution_data(plan, self.protocol, self.new)

    def test_saved_bundle_replays_offline_and_detects_raw_change(self):
        path = self.save("BTC-USD", [(4, 100, 1)])
        b = self.build(["BTC-USD"])
        destination = self.root / "bundle"
        save_execution_data(b, self.plan(["BTC-USD"]), self.protocol, destination)
        with patch("intraday.requests.get", side_effect=AssertionError("No network")):
            replay = load_execution_data(self.plan(["BTC-USD"]), self.protocol, destination)
        pd.testing.assert_frame_equal(replay["references"], b["references"], check_dtype=False)
        with self.assertRaises(FileExistsError):
            save_execution_data(b, self.plan(["BTC-USD"]), self.protocol, destination)
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "raw capture changed"):
            load_execution_data(self.plan(["BTC-USD"]), self.protocol, destination)

    def test_export_changes_detected(self):
        self.save("BTC-USD", [(4, 100, 1)])
        b = self.build(["BTC-USD"])
        destination = self.root / "bundle"
        save_execution_data(b, self.plan(["BTC-USD"]), self.protocol, destination)
        path = destination / "references.csv"
        path.write_text(path.read_text().replace("100.0", "999.0"))
        with self.assertRaisesRegex(ValueError, "output changed"):
            load_execution_data(self.plan(["BTC-USD"]), self.protocol, destination)

    def test_new_request_budget_is_enforced(self):
        self.save("APT-USD", [])
        with patch("intraday.requests.get", side_effect=AssertionError("No network")):
            with self.assertRaises(FileNotFoundError):
                build_execution_data(self.plan(["APT-USD"]), self.protocol, self.new, [self.old],
                                     allow_download=True, max_new_requests=0)


if __name__ == "__main__":
    unittest.main()
