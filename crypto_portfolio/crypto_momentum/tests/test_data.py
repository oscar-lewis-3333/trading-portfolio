import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from _support import (OfflineTestCase, SYMBOLS, START, DATA_END, RUN,
                      crypto_data, crypto_preparation, saved_prices, saved_responses)


class DataTests(OfflineTestCase):
    def test_saved_date_grid_and_report(self):
        prices = saved_prices()
        before = prices.copy(deep=True)
        _, audit = crypto_preparation.audit_daily_prices(prices, SYMBOLS, START, DATA_END)
        pd.testing.assert_frame_equal(prices, before)
        self.assertTrue(audit.passed.all())
        self.assertEqual(audit.rows.to_dict(), dict.fromkeys(SYMBOLS, 2557))
        pd.testing.assert_frame_equal(audit, pd.read_csv(RUN / "price_audit.csv", index_col="symbol"))

    def test_downloader_replays_raw_responses_offline(self):
        class Response:
            def __init__(self, url, params, payload):
                self.url = crypto_data.requests.Request("GET", url, params=params).prepare().url
                self.payload = payload
            def raise_for_status(self):
                pass
            def json(self):
                return self.payload

        def fake_get(url, params, timeout):
            self.assertEqual(timeout, 30)
            self.assertEqual(params["granularity"], 86400)
            key = (url.split("/")[-2], params["start"], params["end"])
            return Response(url, params, saved_responses()[key])

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(crypto_data.requests, "get", side_effect=fake_get) as get, \
                 patch.object(crypto_data.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                runs = [crypto_data.fetch_coinbase_daily_history(s, START, DATA_END, tmp) for s in SYMBOLS]
            self.assertEqual(get.call_count, 22)
            self.assertEqual(len(list(Path(tmp).glob("*.json"))), 22)
            for path in Path(tmp).glob("*.json"):
                self.assertIn("request_url", json.loads(path.read_text()))
        actual = pd.concat([x[0] for x in runs], ignore_index=True)
        actual = actual.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual.sort_index(axis=1), saved_prices().sort_index(axis=1))
        report = pd.concat([x[1] for x in runs], ignore_index=True)
        expected = pd.read_csv(RUN / "download_report.csv")
        for name in ("requested_start", "requested_end"):
            expected[name] = pd.to_datetime(expected[name], utc=True)
        pd.testing.assert_frame_equal(report, expected)

    def test_missing_day_and_asset_detected(self):
        original = saved_prices()
        for changed in (original.drop(index=100), original.loc[original.symbol != SYMBOLS[0]]):
            with self.subTest(rows=len(changed)):
                _, report = crypto_preparation.audit_daily_prices(changed, SYMBOLS, START, DATA_END)
                self.assertFalse(report.loc[SYMBOLS[0], "passed"])
                self.assertGreater(report.loc[SYMBOLS[0], "missing_days"], 0)

    def test_duplicate_detected(self):
        frame = saved_prices()
        frame = pd.concat([frame, frame.iloc[[100]]], ignore_index=True)
        _, report = crypto_preparation.audit_daily_prices(frame, SYMBOLS, START, DATA_END)
        self.assertEqual(report.loc[SYMBOLS[0], "duplicate_rows"], 2)
        self.assertFalse(report.loc[SYMBOLS[0], "passed"])

    def test_corrupt_values_detected(self):
        original = saved_prices()
        cases = [
            ("timestamp", pd.NaT, "invalid_timestamp_rows"),
            ("timestamp", START + pd.Timedelta(hours=1), "off_midnight_rows"),
            ("timestamp", DATA_END, "outside_range_rows"),
            ("close", np.nan, "nonfinite_numeric_rows"),
            ("close", np.inf, "nonfinite_numeric_rows"),
            ("close", "bad", "nonfinite_numeric_rows"),
            ("low", 0.0, "nonpositive_price_rows"),
            ("close", original.loc[100, "high"] * 2, "invalid_ohlc_rows"),
            ("volume", -1.0, "negative_volume_rows"),
        ]
        for column, value, flag in cases:
            with self.subTest(column=column, value=value):
                frame = original.copy(deep=True)
                if isinstance(value, str):
                    frame[column] = frame[column].astype(object)
                frame.loc[100, column] = value
                _, report = crypto_preparation.audit_daily_prices(frame, SYMBOLS, START, DATA_END)
                self.assertFalse(report.loc[SYMBOLS[0], "passed"])
                self.assertGreater(report.loc[SYMBOLS[0], flag], 0)

    def test_zero_volume_reported(self):
        frame = saved_prices()
        frame.loc[100, "volume"] = 0
        _, report = crypto_preparation.audit_daily_prices(frame, SYMBOLS, START, DATA_END)
        self.assertTrue(report.passed.all())
        self.assertEqual(report.loc[SYMBOLS[0], "zero_volume_rows"], 1)

    def test_invalid_schema_and_symbols_rejected(self):
        frame = saved_prices()
        for changed in (frame.drop(columns="close"), frame.assign(symbol="UNKNOWN")):
            with self.subTest(columns=list(changed.columns)), self.assertRaises(ValueError):
                crypto_preparation.audit_daily_prices(changed, SYMBOLS, START, DATA_END)
