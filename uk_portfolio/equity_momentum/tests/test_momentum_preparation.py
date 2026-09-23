"""Offline checks for data units, gaps, dated blocks and reference-price timing."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import momentum_data as data
import momentum_preparation as preparation


def panel():
    dates = pd.bdate_range("2024-01-02", periods=7, name="Date")
    index = pd.MultiIndex.from_product([dates, ["TEST.L"]], names=["Date", "ticker"])
    frame = pd.DataFrame(10., index=index, columns=preparation.PRICE_COLUMNS)
    for raw, adjusted in zip(["Open", "High", "Low", "Close"], preparation.ADJUSTED_COLUMNS):
        frame[adjusted] = frame[raw]
    frame["Volume"] = [0., 0., 20., 0., 0., 0., 3.]
    frame["Dividends"] = 0.
    frame["Stock Splits"] = 0.
    frame["adjustment_factor"] = 1.
    frame["observed_row"] = True
    frame["source"] = "fixture"
    frame.attrs = {"quote_currency": "GBP"}
    return frame, pd.DataFrame(index=dates)


def review():
    return dict(unit_quarantines=[], suspensions=[], unsupported_distributions=[])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def test_conversion_preserves_volume_actions_and_calendar_gaps(self):
        for currency, divisor in [("GBp", 100), ("GBX", 100), ("GBP", 1)]:
            with self.subTest(currency=currency):
                self.check_conversion(currency, divisor)

    def missing_quote_fixture(self):
        frame, schedule = panel()
        key = (schedule.index[3], "TEST.L")
        frame.loc[key, preparation.PRICE_COLUMNS + preparation.ADJUSTED_COLUMNS] = np.nan
        frame.loc[key, "adjustment_factor"] = np.nan
        spec = dict(ticker="TEST.L", date=str(key[0].date()), quote_currency="GBX",
                    quotes=dict(Open=990., High=1010., Low=970., Close=980., Volume=50),
                    adjustment_factor=1.)
        return frame, schedule, key, spec

    def test_verified_repair_converts_pence_and_only_changes_the_missing_row(self):
        frame, schedule, key, spec = self.missing_quote_fixture()
        original = frame.copy(deep=True)
        out = preparation.apply_verified_price_repairs(frame, [spec])
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(out.drop(index=key)[frame.columns], frame.drop(index=key))
        self.assertEqual(out.loc[key, "Open"], 9.9)
        self.assertEqual(out.loc[key, "Close"], 9.8)
        self.assertEqual(out.loc[key, "adj_close"], 9.8)
        self.assertEqual(out.loc[key, "Volume"], 50)
        self.assertEqual(out.verified_price_repair.sum(), 1)
        prepared, _ = preparation.apply_data_review(out, review())
        self.assertTrue(prepared.loc[key, "open_reference_usable"])
        self.assertTrue(prepared.loc[key, "positive_reported_volume"])
        self.assertAlmostEqual(prepared.loc[key, "traded_value_gbp"], 490.)
        preparation.validate_price_panel(prepared, schedule)

    def test_verified_repair_preserves_gbp_units(self):
        frame, _, key, spec = self.missing_quote_fixture()
        spec["quote_currency"] = "GBP"
        for field in ["Open", "High", "Low", "Close"]:
            spec["quotes"][field] /= 100
        out = preparation.apply_verified_price_repairs(frame, [spec])
        self.assertEqual(out.loc[key, "Open"], 9.9)

    def test_verified_repair_refuses_existing_quotes_and_duplicate_repairs(self):
        frame, _, key, spec = self.missing_quote_fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            preparation.apply_verified_price_repairs(frame, [spec, spec])
        frame.loc[key, "Open"] = 10.
        with self.assertRaisesRegex(ValueError, "overwrite"):
            preparation.apply_verified_price_repairs(frame, [spec])

    def test_verified_repair_cannot_override_suspensions_or_quarantines(self):
        for flag in ["known_suspended", "close_suspended", "unverified_quote"]:
            with self.subTest(flag=flag):
                frame, _, key, spec = self.missing_quote_fixture()
                frame[flag] = False
                frame.loc[key, flag] = True
                with self.assertRaisesRegex(ValueError, "blocked quote"):
                    preparation.apply_verified_price_repairs(frame, [spec])

    def test_verified_repair_rejects_unestablished_adjustments_or_actions(self):
        frame, schedule, key, spec = self.missing_quote_fixture()
        frame.loc[(schedule.index[4], "TEST.L"), "adjustment_factor"] = .5
        with self.assertRaisesRegex(ValueError, "adjustment basis"):
            preparation.apply_verified_price_repairs(frame, [spec])
        for value in [1., np.nan]:
            frame, _, key, spec = self.missing_quote_fixture()
            frame.loc[key, "Dividends"] = value
            with self.assertRaisesRegex(ValueError, "activity or actions"):
                preparation.apply_verified_price_repairs(frame, [spec])

    def test_verified_repair_rejects_malformed_quotes_and_currency(self):
        for field, value in [("Open", np.nan), ("Low", 1020.), ("Volume", 0.)]:
            frame, _, _, spec = self.missing_quote_fixture()
            spec["quotes"][field] = value
            with self.assertRaisesRegex(ValueError, "Invalid verified"):
                preparation.apply_verified_price_repairs(frame, [spec])
        frame, _, _, spec = self.missing_quote_fixture()
        spec["quote_currency"] = "USD"
        with self.assertRaisesRegex(ValueError, "GBP or pence"):
            preparation.apply_verified_price_repairs(frame, [spec])

    def test_logic_fingerprint_tolerates_documentation_and_formatting(self):
        source = self.tmp_path / "source.py"
        source.write_text('"""Original documentation."""\ndef f(x):\n    """Original function text."""\n    return x * 100\n')
        expected = preparation.python_logic_sha256(source)
        source.write_text('# New module comment\ndef f( x ):\n    # New explanation\n    return (x*100)\n')
        self.assertEqual(preparation.python_logic_sha256(source), expected)
        source.write_text('def f(x):\n    return x * 10\n')
        self.assertNotEqual(preparation.python_logic_sha256(source), expected)

    def test_cache_allows_cosmetic_edits_but_rejects_logic_and_data_changes(self):
        frame, _ = panel()
        prepared, _ = preparation.apply_data_review(frame, review())
        folder = self.tmp_path / "data/prepared_2015_2025_v1"
        folder.mkdir(parents=True)
        source = self.tmp_path / "source.py"
        source.write_text('def f(x):\n    return x * 100\n')
        policy = self.tmp_path / "policy.json"
        policy.write_text('{"currency": "GBP"}')
        snapshot = folder / "dataset.pkl"
        pd.to_pickle({"prices": prepared}, snapshot)
        manifest = dict(
            local_sha256={"source.py": preparation.sha256(source),
                          "policy.json": preparation.sha256(policy)},
            python_logic_sha256={"source.py": preparation.python_logic_sha256(source)},
            dataset_sha256=preparation.sha256(snapshot),
        )
        (folder / "manifest.json").write_text(json.dumps(manifest))
        source.write_text('# Updated explanation\ndef f(x):\n    return x*100\n')
        preparation.load_prepared_momentum_data(self.tmp_path)
        source.write_text('def f(x):\n    return x * 10\n')
        with self.assertRaisesRegex(ValueError, "Preparation logic changed"):
            preparation.load_prepared_momentum_data(self.tmp_path)
        source.write_text('def f(x):\n    return x * 100\n')
        policy.write_text('{"currency": "GBX"}')
        with self.assertRaisesRegex(ValueError, "Preparation inputs changed"):
            preparation.load_prepared_momentum_data(self.tmp_path)

    def check_conversion(self, currency, divisor):
        tmp_path = self.tmp_path
        schedule = pd.DataFrame(index=pd.bdate_range("2024-01-02", periods=3, name="Date"))
        raw = pd.DataFrame({"Open": [100., 110.], "High": [105., 115.],
                            "Low": [95., 100.], "Close": [100., 110.],
                            "Adj Close": [90., 99.], "Volume": [30., 50.],
                            "Dividends": [2., 0.], "Stock Splits": [.1, 0.]},
                           index=schedule.index[[0, 2]].tz_localize("Europe/London"))
        path = tmp_path / "TEST.L.pkl"
        pd.to_pickle(dict(symbol="TEST.L", status="downloaded", prices=raw,
                          quote_currency=currency, retrieved_at_utc="fixture"), path)
        before = path.read_bytes()
        out = data.prepare_downloaded_prices(["TEST.L"], tmp_path, schedule).xs("TEST.L", level="ticker")
        assert out.Close.iloc[0] == 100 / divisor
        assert np.isclose(out.adj_open.iloc[0], 90 / divisor)
        assert out.Dividends.iloc[0] == 2 / divisor
        assert out.Volume.iloc[0] == 30
        assert out["Stock Splits"].iloc[0] == .1
        assert not out.observed_row.iloc[1]
        assert out.iloc[1][preparation.PRICE_COLUMNS].isna().all()
        assert path.read_bytes() == before


    def test_unit_quarantine_preserves_earlier_data_and_is_prefix_invariant(self):
        frame, schedule = panel()
        start = schedule.index[3]
        cols = preparation.PRICE_COLUMNS + preparation.ADJUSTED_COLUMNS
        frame.loc[(start, "TEST.L"), cols] = .1
        spec = review()
        spec["unit_quarantines"] = [dict(ticker="TEST.L", start=str(start.date()), end_exclusive=None)]
        original = frame.copy(deep=True)
        out, saved = preparation.apply_data_review(frame, spec)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(out.loc[:schedule.index[2], cols], original.loc[:schedule.index[2], cols])
        assert out.loc[start:, cols].isna().all().all()
        assert len(saved) == 4
        assert out.unit_quarantine.sum() == 4
        for cutoff in [schedule.index[1], schedule.index[4]]:
            prefix, _ = preparation.apply_data_review(frame.loc[:cutoff], spec)
            pd.testing.assert_frame_equal(prefix, out.loc[:cutoff])


    def test_bad_review_evidence_fails_instead_of_changing_quotes(self):
        frame, schedule = panel()
        spec = review()
        spec["unit_quarantines"] = [dict(ticker="TEST.L", start=str(schedule.index[3].date()))]
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            preparation.apply_data_review(frame, spec)


    def test_intraday_suspension_keeps_open_and_reopening_loss(self):
        frame, schedule = panel()
        # Suspend after the day-2 open; day 3 is the first suspended opening.
        spec = review()
        spec["suspensions"] = [dict(ticker="TEST.L", start=str(schedule.index[2].date()),
                                   first_suspended_open=str(schedule.index[3].date()),
                                   resume=str(schedule.index[5].date()))]
        frame.loc[(schedule.index[5], "TEST.L"), preparation.PRICE_COLUMNS + preparation.ADJUSTED_COLUMNS] = 2.
        out, _ = preparation.apply_data_review(frame, spec)
        assert out.Open.iloc[2] == 10
        assert pd.isna(out.Close.iloc[2])
        assert out.Open.iloc[3:5].isna().all()
        assert out.Close.iloc[5] == 2
        assert out.open_reference_usable.iloc[2]
        assert not out.open_reference_usable.iloc[4]
        assert out.open_reference_usable.iloc[5]


    def test_zero_volume_is_not_automatically_a_suspension_or_future_fill_filter(self):
        frame, _ = panel()
        out, _ = preparation.apply_data_review(frame, review())
        assert out.zero_volume_run.tolist() == [1, 2, 0, 1, 2, 3, 0]
        assert out.open_reference_usable.all()
        assert not out.known_suspended.any()
        assert out.positive_reported_volume.tolist() == [False, False, True, False, False, False, True]
        altered = frame.copy()
        altered.loc[altered.index[-1], "Volume"] = 0.
        changed, _ = preparation.apply_data_review(altered, review())
        pd.testing.assert_frame_equal(changed.iloc[:-1], out.iloc[:-1])


    def test_missing_rows_and_real_large_losses_survive_preparation(self):
        frame, schedule = panel()
        frame.loc[(schedule.index[0], "TEST.L"), "observed_row"] = False
        frame.loc[(schedule.index[0], "TEST.L"), preparation.PRICE_COLUMNS + preparation.ADJUSTED_COLUMNS] = np.nan
        frame.loc[(schedule.index[3], "TEST.L"), preparation.PRICE_COLUMNS + preparation.ADJUSTED_COLUMNS] = 1.
        out, _ = preparation.apply_data_review(frame, review())
        assert pd.isna(out.adj_close.iloc[0])
        assert out.adj_close.iloc[3] == 1.
        preparation.validate_price_panel(out, schedule)
        broken = out.copy()
        broken.loc[broken.index[0], "Open"] = 10.
        with self.assertRaisesRegex(ValueError, "unobserved"):
            preparation.validate_price_panel(broken, schedule)
        with self.assertRaisesRegex(ValueError, "calendar"):
            preparation.validate_price_panel(out.iloc[1:], schedule)


    def test_unsupported_distribution_is_blocked_without_inventing_cash(self):
        frame, schedule = panel()
        spec = review()
        spec["unsupported_distributions"] = [dict(ticker="TEST.L", start=str(schedule.index[4].date()))]
        out, _ = preparation.apply_data_review(frame, spec)
        assert out.Close.iloc[4:].isna().all()
        assert out.Dividends.eq(0).all()
        assert not out.unverified_quote.iloc[:4].any()


    def test_coverage_separates_short_history_internal_gaps_and_unavailable(self):
        frame, schedule = panel()
        for position in [0, 3, 6]:
            frame.loc[frame.index[position], "observed_row"] = False
            frame.loc[frame.index[position], preparation.PRICE_COLUMNS + preparation.ADJUSTED_COLUMNS] = np.nan
        out, _ = preparation.apply_data_review(frame, review())
        universe = pd.DataFrame(dict(yf_symbol=["TEST.L", "MISSING.L"], isin=["ONE", "TWO"], name=["Test", "Missing"]))
        coverage = preparation.build_coverage(out, universe, {"MISSING.L": "Download failed"}).set_index("yf_symbol")
        assert coverage.loc["TEST.L", "sessions_before_history"] == 1
        assert coverage.loc["TEST.L", "sessions_after_history"] == 1
        assert coverage.loc["TEST.L", "internal_missing_sessions"] == 1
        assert coverage.loc["TEST.L", "longest_internal_gap"] == 1
        assert not coverage.loc["MISSING.L", "has_prices"]
        assert coverage.loc["MISSING.L", "unavailable_reason"] == "Download failed"


    def test_frozen_snapshot_loads_offline_and_rejects_tampering(self):
        tmp_path = self.tmp_path
        frame, _ = panel()
        prepared, _ = preparation.apply_data_review(frame, review())
        folder = tmp_path / "data/prepared_2015_2025_v1"
        folder.mkdir(parents=True)
        path = folder / "dataset.pkl"
        pd.to_pickle({"prices": prepared}, path)
        (folder / "manifest.json").write_text(json.dumps(dict(local_sha256={}, dataset_sha256=preparation.sha256(path))))
        def deny(*args, **kwargs):
            raise AssertionError("Network must not be used during replay")
        with patch.object(data.yf, "Ticker", deny):
            result = preparation.load_prepared_momentum_data(tmp_path)
        pd.testing.assert_frame_equal(result["prices"], prepared)
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            preparation.load_prepared_momentum_data(tmp_path)

if __name__ == "__main__":
    unittest.main()
