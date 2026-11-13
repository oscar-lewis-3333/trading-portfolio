"""Offline checkpoint for the UK notebook through its cleaning cell.

Replay saved source in an empty namespace, using the existing local snapshot.
These checks validate transformations, not the vendor's historical accuracy.
"""

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
# The original development workflow is preserved verbatim in the research archive.
NOTEBOOK = PROJECT / "notebooks/archive/reversal_signals_uk_research_2026_09_15.ipynb"
CACHE = PROJECT / "data" / "uk_2015_2021_v1" / "raw_panel_v1.pkl"
PRICES = ["Open", "High", "Low", "Close", "Adj Close"]


def defines(source, name):
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == name
        for node in ast.walk(ast.parse(source))
    )


def cleaning_cells():
    cells = []
    for number, cell in enumerate(json.loads(NOTEBOOK.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        cells.append((number, source))
        if defines(source, "uk_clean_panel"):
            return cells
    raise AssertionError("The saved notebook has no uk_clean_panel cell.")


def run_cells(cells, namespace):
    previous_directory = Path.cwd()
    try:
        os.chdir(NOTEBOOK.parent)
        with redirect_stdout(io.StringIO()), \
                patch("yfinance.download", side_effect=AssertionError("Offline checkpoint")), \
                patch.object(socket.socket, "connect", side_effect=AssertionError("Offline checkpoint")):
            for number, source in cells:
                exec(compile(source, f"{NOTEBOOK}:cell-{number}", "exec"), namespace)
    finally:
        os.chdir(previous_directory)


def keyed(frame):
    return frame.set_index("ticker", append=True).sort_index()


def row(frame, ticker, date):
    selected = frame.loc[frame["ticker"].eq(ticker) & (frame.index == pd.Timestamp(date))]
    if len(selected) != 1:
        raise AssertionError(f"Expected exactly one {ticker} observation on {date}.")
    return selected.iloc[0]


class UKDataPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not NOTEBOOK.is_file() or not CACHE.is_file():
            raise unittest.SkipTest("UK notebook and local raw snapshot are required")
        cls.cache_digest = hashlib.sha256(CACHE.read_bytes()).hexdigest()
        cls.frozen_raw = pd.read_pickle(CACHE)
        cls.cells = cleaning_cells()
        cls.ns = {"__name__": "__main__", "display": lambda *args, **kwargs: None}
        run_cells(cls.cells, cls.ns)

    def test_raw_snapshot_and_observed_values_are_preserved(self):
        self.assertEqual(hashlib.sha256(CACHE.read_bytes()).hexdigest(), self.cache_digest)
        pd.testing.assert_frame_equal(self.ns["uk_raw_panel"], self.frozen_raw)
        aligned = self.ns["uk_panel"]
        observed = aligned.loc[aligned["observed_row"], self.frozen_raw.columns]
        # Reindexing introduces NaNs, legitimately converting integer volume to float.
        pd.testing.assert_frame_equal(keyed(observed), keyed(self.frozen_raw), check_dtype=False)

    def test_complete_calendar_and_missing_observations(self):
        ns = self.ns
        panel = ns["uk_panel"]
        sessions = ns["uk_schedule"].index
        self.assertEqual((len(ns["uk_tickers"]), len(ns["cached_tickers"]), len(ns["missing_tickers"])),
                         (646, 528, 118))
        self.assertEqual(len(sessions), 1771)
        self.assertEqual(len(panel), 935088)
        self.assertEqual(int(panel["observed_row"].sum()), 764391)
        self.assertTrue(keyed(panel).index.is_unique)
        for ticker, group in panel.groupby("ticker", sort=False):
            with self.subTest(ticker=ticker):
                pd.testing.assert_index_equal(group.index, sessions, check_names=False)
        inserted = ~panel["observed_row"]
        self.assertEqual(int(inserted.sum()), 170697)
        self.assertTrue(panel.loc[inserted, PRICES + ["Volume"]].isna().all().all())
        incomplete = panel["observed_row"] & panel[PRICES].isna().any(axis=1)
        self.assertEqual(int(incomplete.sum()), 8)
        self.assertTrue(panel.loc[incomplete, "Volume"].eq(0).all())
        self.assertTrue((panel.index < pd.Timestamp(ns["UK_PERIODS"]["holdout_start"])).all())

    def test_six_unit_repairs_preserve_volume_and_adjustment_factors(self):
        before, after = self.ns["uk_panel"], self.ns["uk_unit_panel"]
        self.assertEqual(int(after["unit_correction_provisional"].sum()), 6)
        expected = [
            ("AOM.L", "2021-03-29", 186.5), ("FIN.L", "2021-04-06", 61.5),
            ("COIL.L", "2021-03-23", 82.5), ("SUP.L", "2021-02-03", 155.),
            ("SUP.L", "2021-02-05", 178.), ("SUP.L", "2021-02-08", 184.),
        ]
        for ticker, date, close in expected:
            with self.subTest(ticker=ticker, date=date):
                old, new = row(before, ticker, date), row(after, ticker, date)
                self.assertAlmostEqual(new["Close"], close)
                np.testing.assert_allclose(new[PRICES].astype(float), old[PRICES].astype(float) / 100)
                self.assertEqual(new["Volume"], old["Volume"])
                self.assertAlmostEqual(new["Adj Close"] / new["Close"], old["Adj Close"] / old["Close"])
        untouched = ~after["unit_correction_provisional"]
        pd.testing.assert_frame_equal(after.loc[untouched, before.columns], before.loc[untouched])

    def test_consolidations_preserve_traded_value_on_retained_history(self):
        before, after = self.ns["uk_unit_panel"], self.ns["uk_clean_panel"]
        for ticker, cutoff in [("PXEN.L", "2020-07-01"), ("ROAD.L", "2018-01-22")]:
            with self.subTest(ticker=ticker):
                mask = before["ticker"].eq(ticker) & (before.index < pd.Timestamp(cutoff))
                old_value = before.loc[mask, "Close"] * before.loc[mask, "Volume"]
                new_value = after.loc[mask, "Close"] * after.loc[mask, "Volume"]
                np.testing.assert_allclose(new_value, old_value, rtol=1e-12, atol=1e-8, equal_nan=True)
                old_factor = before.loc[mask, "Adj Close"] / before.loc[mask, "Close"]
                new_factor = after.loc[mask, "Adj Close"] / after.loc[mask, "Close"]
                np.testing.assert_allclose(new_factor, old_factor, equal_nan=True)
        # Cached Yahoo prices retain float32 quantisation before rescaling.
        self.assertAlmostEqual(row(after, "PXEN.L", "2020-06-30")["Close"], 1.625, delta=1e-6)
        self.assertAlmostEqual(row(after, "ROAD.L", "2018-01-19")["Close"],
                               row(before, "ROAD.L", "2018-01-19")["Close"] * 33)

    def test_event_rows_do_not_invent_volume_or_double_adjust_existing_splits(self):
        clean = self.ns["uk_clean_panel"]
        for ticker, date, close, ratio in [
            ("BCE.L", "2018-08-10", 10.5, .02),
            ("PXEN.L", "2020-07-01", 2., .04),
        ]:
            with self.subTest(ticker=ticker):
                event = row(clean, ticker, date)
                self.assertAlmostEqual(event["Close"], close, delta=1e-6)
                self.assertEqual(event["Volume"], 0.)
                self.assertAlmostEqual(event["Stock Splits"], ratio)
        self.assertAlmostEqual(row(clean, "ROAD.L", "2018-06-25")["Stock Splits"], 1 / 33)
        original = row(self.ns["uk_panel"], "ROAD.L", "2020-01-07")
        pd.testing.assert_series_equal(row(clean, "ROAD.L", "2020-01-07")[original.index], original)

    def test_road_exclusions_have_exact_boundaries_and_preserve_provenance(self):
        clean = self.ns["uk_clean_panel"]
        self.assertEqual(int(clean["known_unlisted"].sum()), 107)
        self.assertEqual(int(clean["known_nex_listing"].sum()), 388)
        excluded = clean["known_unlisted"] | clean["known_nex_listing"]
        self.assertFalse((clean["known_unlisted"] & clean["known_nex_listing"]).any())
        self.assertTrue(clean.loc[excluded, "ticker"].eq("ROAD.L").all())
        self.assertTrue(clean.loc[excluded, PRICES + ["Volume"]].isna().all().all())
        for date, unlisted, nex in [
            ("2018-01-19", False, False), ("2018-01-22", True, False),
            ("2018-06-25", True, False), ("2018-06-26", False, True),
            ("2020-01-06", False, True), ("2020-01-07", False, False),
        ]:
            with self.subTest(date=date):
                observation = row(clean, "ROAD.L", date)
                self.assertEqual(observation["known_unlisted"], unlisted)
                self.assertEqual(observation["known_nex_listing"], nex)
        pd.testing.assert_series_equal(clean["observed_row"], self.ns["uk_panel"]["observed_row"])

    def test_unrelated_tickers_are_unchanged(self):
        before, after = self.ns["uk_panel"], self.ns["uk_clean_panel"]
        affected = {"AOM.L", "FIN.L", "COIL.L", "SUP.L", "BCE.L", "PXEN.L", "ROAD.L"}
        untouched = ~before["ticker"].isin(affected)
        pd.testing.assert_frame_equal(after.loc[untouched, before.columns], before.loc[untouched])

    def test_observed_prices_are_positive_finite_and_original_gaps_remain(self):
        before, after = self.ns["uk_panel"], self.ns["uk_clean_panel"]
        prices = after[PRICES]
        self.assertFalse((prices.notna() & (~np.isfinite(prices) | prices.le(0))).any().any())
        self.assertTrue((~before[PRICES].isna() | prices.isna()).all().all())
        self.assertFalse(after["Volume"].dropna().lt(0).any())

    def test_rerunning_repair_cells_does_not_compound_adjustments(self):
        first = next(i for i, (_, source) in enumerate(self.cells) if defines(source, "uk_unit_panel"))
        repeated = self.ns.copy()
        run_cells(self.cells[first:], repeated)
        pd.testing.assert_frame_equal(repeated["uk_clean_panel"], self.ns["uk_clean_panel"])


if __name__ == "__main__":
    unittest.main()
