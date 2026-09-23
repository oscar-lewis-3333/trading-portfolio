"""Request planning must include exits, share data, and stay in development."""
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from execution_plan import build_execution_request_plan


class ExecutionPlanTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"development_start": "2022-01-01",
                         "development_end_exclusive": "2022-01-04",
                         "holdout_start": "2022-01-04"}
        self.grid = pd.DataFrame({"rebalance_days": [1, 2]}, index=["daily", "two_day"])
        assets = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", "CASH"]
        daily = pd.DataFrame(0., index=pd.date_range("2022-01-01", periods=3, tz="UTC"), columns=assets)
        slow = pd.DataFrame(0., index=pd.date_range("2022-01-01", periods=2, freq="2D", tz="UTC"), columns=assets)
        for i, asset in enumerate(["BTC-USD", "ETH-USD", "SOL-USD"]):
            daily.loc[daily.index[i], asset] = 1.
        slow.loc[slow.index[0], "ADA-USD"] = 1.
        slow.loc[slow.index[1], "DOT-USD"] = 1.
        self.targets = {
            "daily": {"strategy": daily.copy(), "benchmark": daily.copy()},
            "two_day": {"strategy": slow.copy(), "benchmark": slow.copy()},
        }

    def build(self, destination, reuse=()):
        return build_execution_request_plan(self.targets, self.grid, self.protocol, destination, reuse)

    def test_union_deduplicates_and_retains_previous_scheduled_holdings(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, summary = self.build(Path(directory) / "not_created")
            expected = {
                ("2022-01-01", "BTC-USD"), ("2022-01-01", "ADA-USD"),
                ("2022-01-02", "BTC-USD"), ("2022-01-02", "ETH-USD"),
                ("2022-01-03", "ETH-USD"), ("2022-01-03", "SOL-USD"),
                ("2022-01-03", "ADA-USD"), ("2022-01-03", "DOT-USD"),
            }
            actual = set(zip(plan.day.dt.strftime("%Y-%m-%d"), plan.product_id))
            self.assertEqual(actual, expected)
            self.assertEqual(summary.unique_product_day_windows, 8)
            self.assertEqual(summary.exit_only_windows, 3)
            self.assertTrue((plan.earliest_execution_at - plan.day).eq(pd.Timedelta(minutes=5)).all())
            self.assertFalse((Path(directory) / "not_created").exists())

    def test_benchmark_only_assets_and_their_exits_are_included(self):
        benchmark = self.targets["daily"]["benchmark"]
        benchmark["XLM-USD"] = 0.
        benchmark.loc[benchmark.index[0], ["BTC-USD", "XLM-USD"]] = [.5, .5]
        with tempfile.TemporaryDirectory() as directory:
            plan, _ = self.build(directory)
        xlm = plan.loc[plan.product_id.eq("XLM-USD")]
        self.assertEqual(xlm.day.dt.strftime("%Y-%m-%d").tolist(), ["2022-01-01", "2022-01-02"])
        self.assertEqual(xlm.exit_only.tolist(), [False, True])

    def test_cash_transition_keeps_exit_price_then_stops_requests(self):
        for role in ("strategy", "benchmark"):
            weights = self.targets["daily"][role]
            weights.iloc[1:] = 0.
            weights.loc[weights.index[1:], "CASH"] = 1.
        with tempfile.TemporaryDirectory() as directory:
            plan, _ = self.build(directory)
        btc = plan.loc[plan.product_id.eq("BTC-USD")]
        self.assertEqual(btc.day.dt.strftime("%Y-%m-%d").tolist(), ["2022-01-01", "2022-01-02"])
        self.assertTrue(btc.iloc[-1].exit_only)

    def test_cache_presence_reuses_files_without_reads_writes_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old"
            new = Path(directory) / "new"
            old.mkdir()
            cached = old / "BTC-USD_20220101_0000_0015_60s.json"
            cached.write_text("presence-only fixture")
            before = cached.stat().st_mtime_ns
            with patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
                plan, summary = self.build(new, [old])
            self.assertEqual(summary.cached_windows, 1)
            self.assertEqual(summary.missing_windows, 7)
            self.assertEqual(plan.loc[plan.cache_exists, "cache_path"].iloc[0], str(cached.resolve()))
            self.assertEqual(cached.stat().st_mtime_ns, before)
            self.assertFalse(new.exists())

    def test_all_cash_needs_no_price_windows(self):
        for pair in self.targets.values():
            for role, weights in pair.items():
                pair[role] = pd.DataFrame({"CASH": 1.}, index=weights.index)
        with tempfile.TemporaryDirectory() as directory:
            plan, summary = self.build(directory)
        self.assertTrue(plan.empty)
        self.assertEqual(summary.decision_dates, 3)
        self.assertEqual(summary.unique_product_day_windows, 0)

    def test_missing_configuration_or_truncated_calendar_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            removed = self.targets.pop("two_day")
            with self.assertRaisesRegex(ValueError, "every uniquely"):
                self.build(directory)
            self.targets["two_day"] = removed
            self.targets["daily"]["strategy"] = self.targets["daily"]["strategy"].iloc[1:]
            with self.assertRaisesRegex(ValueError, "target calendar"):
                self.build(directory)

    def test_holdout_dates_are_rejected(self):
        weights = self.targets["daily"]["strategy"]
        extra = weights.iloc[[-1]].copy()
        extra.index = pd.DatetimeIndex([pd.Timestamp("2022-01-04", tz="UTC")])
        self.targets["daily"]["strategy"] = pd.concat([weights, extra])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "target calendar"):
                self.build(directory)


if __name__ == "__main__":
    unittest.main()
