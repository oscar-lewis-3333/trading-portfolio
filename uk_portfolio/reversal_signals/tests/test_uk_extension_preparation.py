
#ensuring data has no holes, just general prep testing
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import reversal_namespace  # widens the trading_portfolio package path
import trading_portfolio.reversal_signals.src.reversal_preparation as prep


class SuspensionBoundaryTests(unittest.TestCase):
    def test_intraday_halt_preserves_open_and_restoration_is_exclusive(self):
        dates = pd.bdate_range("2024-04-22", "2024-04-29", name="Date")
        index = pd.MultiIndex.from_product([dates, ["A.L", "B.L"]], names=["Date", "ticker"])
        original = pd.DataFrame(10., index=index, columns=prep.DATA_COLUMNS)
        event = dict(ticker="A.L", start="2024-04-24", first_suspended_open="2024-04-25", resume="2024-04-29")
        result = prep.apply_suspensions(original, [event])
        self.assertEqual(result.loc[("2024-04-24", "A.L"), "Open"], 10.)
        self.assertTrue(np.isnan(result.loc[("2024-04-24", "A.L"), "Close"]))
        self.assertFalse(result.loc[("2024-04-24", "A.L"), "known_suspended"])
        self.assertTrue(result.loc[("2024-04-25", "A.L"), prep.PRICE_COLUMNS].isna().all())
        self.assertTrue(result.loc[("2024-04-25", "A.L"), "vendor_volume_during_suspension"])
        pd.testing.assert_series_equal(result.loc[("2024-04-29", "A.L"), prep.DATA_COLUMNS], original.loc[("2024-04-29", "A.L")])
        pd.testing.assert_frame_equal(result.xs("B.L", level="ticker")[prep.DATA_COLUMNS], original.xs("B.L", level="ticker"))
        pd.testing.assert_frame_equal(prep.apply_suspensions(result, [event]), result)
        self.assertTrue(original.eq(10.).all().all())

    def test_open_ended_halt_masks_last_day(self):
        index = pd.MultiIndex.from_product([pd.bdate_range("2025-12-29", "2025-12-31", name="Date"), ["A.L"]], names=["Date", "ticker"])
        original = pd.DataFrame(1., index=index, columns=prep.DATA_COLUMNS)
        result = prep.apply_suspensions(original, [dict(ticker="A.L", start="2025-12-30", resume=None)])
        self.assertFalse(result.known_suspended.iloc[0])
        self.assertTrue(result.iloc[1:][prep.PRICE_COLUMNS].isna().all().all())


class PreparedSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = PROJECT / "data/uk_extension_2023_2025_v1"
        if not (cls.directory / "prepared_2015_2025_v1.pkl").exists():
            raise unittest.SkipTest("Prepared local snapshot required")
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Offline preparation")):
            cls.result = prep.prepare_uk_extension(PROJECT)
        cls.panel = cls.result["panel"]

    def test_fixed_universe_calendar_and_evaluation_boundary(self):
        summary = self.result["manifest"]["summary"]
        self.assertEqual([summary[k] for k in ["declared_tickers", "available_tickers", "unavailable_tickers", "sessions", "evaluation_sessions"]], [646, 573, 73, 2779, 507])
        self.assertEqual(len(self.panel), 573 * 2779)
        self.assertTrue(self.panel.index.is_unique)
        self.assertTrue(self.panel.index.is_monotonic_increasing)
        self.assertIn(pd.Timestamp("2022-05-30"), self.result["schedule"].index)
        self.assertEqual(self.panel.index.get_level_values("Date").max(), pd.Timestamp("2025-12-31"))

    def test_missing_and_suspended_quotes_are_unavailable(self):
        self.assertTrue(self.panel.loc[~self.panel.observed_row, prep.PRICE_COLUMNS + ["Volume"]].isna().all().all())
        self.assertTrue(self.panel.loc[self.panel.known_suspended, prep.PRICE_COLUMNS].isna().all().all())
        self.assertFalse(self.result["market"].loc[self.panel.known_suspended, "open_reference_usable"].any())
        values = self.panel[prep.PRICE_COLUMNS]
        self.assertFalse((values.notna() & (~np.isfinite(values) | values.le(0))).any().any())

    def test_old_price_unit_repairs_do_not_scale_dividends_or_volume(self):
        old = pd.read_pickle(self.directory / "historical_input_gbp_v1.pkl")
        mask = old.index.get_level_values("ticker").isin(prep.UNIT_CORRECTIONS) & (old.index.get_level_values("Date") < pd.Timestamp("2023-01-01"))
        old = old.loc[mask]
        new = self.panel.loc[old.index]
        tradable = ~new.close_suspended
        for column in ["Open", "High", "Low", "Close"]:
            np.testing.assert_allclose(new.loc[tradable, column], old.loc[tradable, column] * 100., equal_nan=True)
        pd.testing.assert_frame_equal(new[["Volume", "Dividends"]], old[["Volume", "Dividends"]], check_dtype=False)
        self.assertAlmostEqual(self.panel.loc[("2023-01-03", "HSM.L"), "Close"], 4.5)

    def test_tin_consistent_synthetic_shares_preserve_traded_value(self):
        old = pd.read_pickle(self.directory / "historical_input_gbp_v1.pkl").xs("TIN.L", level="ticker").loc[:"2022-12-31"]
        new = self.panel.xs("TIN.L", level="ticker").loc[old.index]
        np.testing.assert_allclose(new.Close, old.Close * 10., equal_nan=True)
        np.testing.assert_allclose(new.Volume, old.Volume / 10., equal_nan=True)
        np.testing.assert_allclose(new.Close * new.Volume, old.Close * old.Volume, equal_nan=True)

    def test_unaffected_historical_cash_fields_are_preserved(self):
        old = pd.read_pickle(self.directory / "historical_input_gbp_v1.pkl")
        mask = (old.index.get_level_values("Date") < pd.Timestamp("2023-01-01")) & ~old.index.get_level_values("ticker").isin(prep.UNIT_CORRECTIONS + ["TIN.L", "PPHC.L", "SPSC.L", "AST.L", "RFG.L"])
        old = old.loc[mask]
        new = self.panel.loc[old.index]
        usable = ~new.close_suspended & ~new.unverified_quote
        columns = ["Open", "High", "Low", "Close", "Dividends", "Volume"]
        pd.testing.assert_frame_equal(new.loc[usable, columns], old.loc[usable, columns], check_dtype=False)

    def test_documented_actions_and_quarantined_quotes(self):
        ast = self.panel.xs("AST.L", level="ticker")
        self.assertAlmostEqual(ast.loc["2015-12-01", "Stock Splits"], .05)
        self.assertAlmostEqual(ast.loc["2015-11-30", "Close"], .96, places=6)
        rfg = self.panel.xs("RFG.L", level="ticker")
        self.assertEqual(rfg.loc["2021-11-23", "Dividends"], 1.66)
        implied_total_return = rfg.loc["2021-11-23", "Adj Close"] / rfg.loc["2021-11-22", "Adj Close"]
        cash_total_return = (rfg.loc["2021-11-23", "Close"] + 1.66) / rfg.loc["2021-11-22", "Close"]
        self.assertAlmostEqual(implied_total_return, cash_total_return)
        self.assertTrue(self.panel.loc[self.panel.unverified_quote, prep.PRICE_COLUMNS].isna().all().all())
        self.assertEqual(set(self.result["quarantined_quotes"].index.get_level_values("ticker")), {"TRB.L", "IES.L"})
        self.assertTrue(self.panel.loc[("2021-03-22", "AURA.L"), "known_suspended"])
        self.assertFalse(self.panel.loc[("2021-03-26", "AURA.L"), "known_suspended"])

    def test_price_adjustments_match_one_vendor_vintage(self):
        for ticker in prep.FULL_HISTORY_REPAIRS:
            with self.subTest(ticker=ticker):
                source = pd.read_pickle(self.directory / "reconciliation_sources_v1" / f"{ticker}.pkl")["repaired"]
                new = self.panel.xs(ticker, level="ticker")
                dates = new.index.intersection(source.index)
                mask = new.loc[dates, "Close"].notna()
                np.testing.assert_allclose((new["Adj Close"] / new.Close).loc[dates][mask], (source["Adj Close"] / source.Close).loc[dates][mask])

    def test_original_results_strategy_sources_and_inputs_unchanged(self):
        protected = json.loads((self.directory / "preparation_backup_v1/protected_sha256.json").read_text())
        for filename, digest in protected.items():
            prep.verify_frozen_source(PROJECT / filename, digest)
        for filename, digest in self.result["manifest"]["input_sha256"].items():
            path = Path(prep.__file__) if filename == "preparation_source" else PROJECT / filename
            self.assertEqual(prep.sha256(path), digest, filename)

    def test_preparation_stops_before_signal_or_outcome_computation(self):
        self.assertEqual(self.result["manifest"]["summary"]["stage"], "ready_for_features")
        self.assertFalse(set(self.result) & {"equity", "returns", "features", "choices", "inference"})
        self.assertFalse(any("score" in column or "return" in column for column in self.panel.columns))


if __name__ == "__main__":
    unittest.main()
