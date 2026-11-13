"""Offline execution-reference, cache, request-budget and boundary checks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from execution_probe import probe_execution_sample


class ExecutionProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.day = pd.Timestamp("2023-07-03", tz="UTC")
        self.protocol = {"development_start": "2022-01-01",
                         "development_end_exclusive": "2025-01-01",
                         "holdout_start": "2025-01-01"}

    def path(self, product, directory="old"):
        return self.root / directory / f"{product}_20230703_0000_0015_60s.json"

    def save(self, product, bars, directory="old"):
        path = self.path(product, directory)
        path.parent.mkdir(exist_ok=True)
        payload = [[int((self.day + pd.Timedelta(minutes=m)).timestamp()),
                    price, price, price, price, volume]
                   for m, price, volume in bars]
        record = {
            "source": "coinbase_exchange",
            "request": {"product_id": product,
                        "url": f"https://api.exchange.coinbase.com/products/{product}/candles",
                        "params": {"granularity": 60, "start": self.day.isoformat(),
                                   "end": (self.day + pd.Timedelta(minutes=15)).isoformat()}},
            "http_status": 200, "error": None, "response_text": json.dumps(payload),
        }
        path.write_text(json.dumps(record))
        return path

    def plan(self, products):
        return pd.DataFrame([{
            "product_id": p, "day": self.day,
            "earliest_execution_at": self.day + pd.Timedelta(minutes=5),
            "cache_path": str(self.path(p)), "download_path": str(self.path(p, "new")),
            "cache_exists": False,  # Deliberately stale: inspect filesystem instead.
        } for p in products])

    def run_sample(self, products, **kwargs):
        return probe_execution_sample(self.plan(products), [self.day], self.protocol, **kwargs)

    def test_completed_positive_volume_only_and_immutable_cache(self):
        path = self.save("BTC-USD", [(2, 100., 1), (4, 200., 0), (5, 999., 1)])
        before = path.read_bytes(), path.stat().st_mtime_ns
        with patch("intraday.requests.get", side_effect=AssertionError("network forbidden")):
            report, summary = self.run_sample(["BTC-USD"])
        self.assertEqual(report.iloc[0].reference_price_usd, 100.)
        self.assertEqual(report.iloc[0].age_upper_bound_minutes, 3.)
        self.assertEqual(summary.new_requests, 0)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        self.assertFalse((self.root / "new").exists())

    def test_later_trade_cannot_fill_0005_and_sparse_window_is_retained(self):
        self.save("APT-USD", [(5, 100., 1), (9, 101., 1)])
        report, summary = self.run_sample(["APT-USD"])
        self.assertEqual(report.iloc[0].status, "no_completed_trade_bar")
        self.assertFalse(report.iloc[0].reference_available)
        self.assertEqual(summary.flagged_captured_windows, 1)

    def test_missing_cache_does_not_download_by_default(self):
        with patch("intraday.requests.get", side_effect=AssertionError("network forbidden")):
            report, summary = self.run_sample(["BTC-USD"])
        self.assertEqual(report.iloc[0].status, "missing_cache")
        self.assertEqual(summary.missing_cached_windows, 1)

    def test_request_budget_and_restart_reuse(self):
        response = type("Response", (), {"status_code": 200,
            "text": json.dumps([[int(self.day.timestamp()) + 240, 100, 100, 100, 100, 1]]),
            "url": "fixture"})()
        with patch("intraday.requests.get", return_value=response) as get, patch("intraday.time.sleep"):
            _, first = self.run_sample(["BTC-USD", "ETH-USD"], allow_download=True, max_new_requests=1)
            self.assertEqual(get.call_count, 1)
            _, second = self.run_sample(["BTC-USD", "ETH-USD"], allow_download=True, max_new_requests=1)
            self.assertEqual(get.call_count, 2)
        self.assertEqual(first.usable_references, 1)
        self.assertEqual(second.usable_references, 2)

    def test_bad_candles_stop_before_next_request(self):
        self.save("APT-USD", [(4, 100, 1), (4, 100, 1)])
        with patch("intraday.requests.get", side_effect=AssertionError("network forbidden")):
            report, summary = self.run_sample(["APT-USD", "BTC-USD"], allow_download=True)
        self.assertEqual(report.iloc[0].outcome, "invalid_candles")
        self.assertEqual(summary.unprocessed_windows, 1)

    def test_wrong_metadata_is_rejected(self):
        path = self.save("BTC-USD", [(4, 100, 1)])
        record = json.loads(path.read_text())
        record["request"]["product_id"] = "ETH-USD"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            self.run_sample(["BTC-USD"])

    def test_holdout_duplicate_and_unplanned_dates_are_rejected(self):
        for days in (["2025-01-01"], [self.day, self.day], ["2023-07-04"]):
            with self.subTest(days=days), self.assertRaises(ValueError):
                probe_execution_sample(self.plan(["BTC-USD"]), days, self.protocol)


if __name__ == "__main__":
    unittest.main()
