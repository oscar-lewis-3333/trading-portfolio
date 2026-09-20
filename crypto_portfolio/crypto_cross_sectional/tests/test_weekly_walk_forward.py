#testing the walk-forward works as intended
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CRYPTO_RESEARCH_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
import weekly_walk_forward as wf
from weekly_sweep import weekly_protocol


def choices_fixture():
    rows = []
    for fold, (lookback, fraction) in zip(weekly_protocol()["folds"], [(90, .1), (30, .2), (7, .3)]):
        rows.append({**fold, "lookback_days": lookback, "top_fraction": fraction})
    choices = pd.DataFrame(rows).set_index("fold")
    for field in ["training_start", "training_end", "test_start", "test_end"]:
        choices[field] = pd.to_datetime(choices[field], utc=True)
    return choices


def market_fixture():
    dates = pd.date_range("2023-01-01", "2026-08-31", freq="D", tz="UTC")
    n = len(dates)
    clock = np.arange(n)
    pieces = []
    for i in range(20):
        pieces.append(pd.DataFrame({
            "product_id": f"ASSET{i:02d}-USD", "timestamp": dates,
            "close": 100 * np.exp(.0002 * clock + .1 * np.sin(clock / (13 + i) + i)),
            "signal_available_at": dates + pd.Timedelta(days=1),
            "universe_candidate": clock >= 180,
            "median_dollar_volume": 2000000. + i,
        }))
    panel = pd.concat(pieces, ignore_index=True)
    weeks = pd.date_range("2024-01-01", "2026-08-31", freq="W-MON", tz="UTC")
    schedule = pd.DataFrame({"day": weeks, "execution_at": weeks + pd.Timedelta(minutes=5),
                             "status": "ready", "required_assets": 20})
    return panel, schedule


class WeeklyWalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel, cls.schedule = market_fixture()

    def test_each_selection_uses_its_own_frozen_training_cutoff(self):
        calls = []
        winning = {"2024-01-01": (90, .1), "2025-01-01": (30, .2), "2026-01-01": (7, .3)}

        def training(*args, training_end, progress):
            calls.append(training_end)
            n = (pd.Timestamp(training_end) - pd.Timestamp("2022-01-01")).days
            rows = []
            for l in weekly_protocol()["lookback_days"]:
                for f in weekly_protocol()["top_fractions"]:
                    score = 2. if (l, f) == winning[training_end] else 0.
                    rows.append({"lookback_days": l, "top_fraction": f, "observations": n,
                                 "sharpe_difference": score, "sharpe_zero_cash_rate": score + .1,
                                 "benchmark_sharpe": .1, "annualised_one_way_turnover": 1.})
            return pd.DataFrame(rows).set_index(["lookback_days", "top_fraction"]), {}, None

        with patch.object(wf, "run_training_sweep", side_effect=training):
            choices, tables = wf.select_annual_parameters(None, None, None, None, weekly_protocol(), progress=False)
        self.assertEqual(calls, list(winning))
        self.assertEqual(list(zip(choices.lookback_days, choices.top_fraction)), list(winning.values()))
        self.assertEqual(set(tables), {1, 2, 3})

    def test_parameter_changes_wait_for_first_scheduled_monday(self):
        targets, benchmark, audit = wf.build_walk_forward_targets(self.panel, choices_fixture(), self.schedule, weekly_protocol())
        firsts = audit.groupby("selection_fold").first()
        self.assertEqual(firsts.loc[1, "decision_at"], pd.Timestamp("2024-01-01", tz="UTC"))
        self.assertEqual(firsts.loc[2, "decision_at"], pd.Timestamp("2025-01-06", tz="UTC"))
        self.assertEqual(firsts.loc[3, "decision_at"], pd.Timestamp("2026-01-05", tz="UTC"))
        self.assertEqual(audit.loc[audit.decision_at.eq(pd.Timestamp("2024-12-30", tz="UTC")), "selection_fold"].iloc[0], 1)
        self.assertEqual(audit.loc[audit.decision_at.eq(pd.Timestamp("2025-12-29", tz="UTC")), "selection_fold"].iloc[0], 2)
        np.testing.assert_allclose(targets.sum(axis=1), 1., rtol=0, atol=1e-12)
        for fold, count in [(1, 2), (2, 4), (3, 6)]:
            days = audit.loc[audit.selection_fold.eq(fold), "decision_at"]
            self.assertTrue(targets.loc[days].drop(columns="CASH").gt(0).sum(axis=1).eq(count).all())
        self.assertTrue(benchmark.drop(columns="CASH").gt(0).sum(axis=1).eq(20).all())

    def test_later_prices_and_later_choices_cannot_change_earlier_targets(self):
        choices = choices_fixture()
        original, _, _ = wf.build_walk_forward_targets(self.panel, choices, self.schedule, weekly_protocol())
        modified = self.panel.copy()
        boundary = pd.Timestamp("2025-01-01", tz="UTC")
        modified.loc[modified.timestamp.ge(boundary), "close"] *= 7
        choices.loc[2, ["lookback_days", "top_fraction"]] = [180, .3]
        changed, _, _ = wf.build_walk_forward_targets(modified, choices, self.schedule, weekly_protocol())
        pd.testing.assert_frame_equal(original.loc[original.index < boundary], changed.loc[changed.index < boundary])

    def test_incomplete_choices_future_training_or_unfrozen_parameters_fail(self):
        cases = [choices_fixture().drop(index=3)]
        future = choices_fixture()
        future.loc[1, "training_end"] = pd.Timestamp("2024-02-01", tz="UTC")
        cases.append(future)
        bad_grid = choices_fixture()
        bad_grid.loc[1, "lookback_days"] = 21
        cases.append(bad_grid)
        for choices in cases:
            with self.assertRaises(ValueError):
                wf.build_walk_forward_targets(self.panel, choices, self.schedule, weekly_protocol())

    def test_omitted_week_and_unresolved_execution_fail(self):
        with self.assertRaisesRegex(ValueError, "every Monday"):
            wf.build_walk_forward_targets(self.panel, choices_fixture(), self.schedule.drop(index=20), weekly_protocol())
        unresolved = self.schedule.copy()
        unresolved.loc[20, "status"] = "unresolved"
        with self.assertRaisesRegex(ValueError, "Resolve"):
            wf.build_walk_forward_targets(self.panel, choices_fixture(), unresolved, weekly_protocol())

    def test_portfolios_run_once_each_over_all_years_without_annual_resets(self):
        seen = []
        prices = self.panel.rename(columns={"close": "close_gbp"})
        prices["open_gbp"] = prices["close_gbp"]
        refs = pd.DataFrame({"day": self.schedule.day})

        def ledger(p, targets, schedule, references, **kwargs):
            seen.append(targets.copy())
            self.assertEqual(targets.index.min(), pd.Timestamp("2024-01-01", tz="UTC"))
            self.assertEqual(targets.index.max(), pd.Timestamp("2026-08-31", tz="UTC"))
            self.assertTrue(schedule.day.isin(targets.index).all())
            self.assertEqual(kwargs, {"initial_cash": 1000., "fee_bps": 9., "other_cost_bps": 3.5})
            return "one continuous ledger", "trades", "positions"

        with patch.object(wf, "run_intraday_ledger", side_effect=ledger):
            runs, _, _, _ = wf.run_walk_forward(self.panel, prices, self.schedule, refs, choices_fixture(), weekly_protocol())
        self.assertEqual(len(seen), 2)
        self.assertEqual(set(runs), {"walk_forward", "benchmark"})

    def test_incomplete_evaluation_history_fails(self):
        prices = self.panel.loc[self.panel.timestamp.lt(pd.Timestamp("2026-08-31", tz="UTC"))]
        with self.assertRaisesRegex(ValueError, "frozen end date"):
            wf.run_walk_forward(self.panel, prices, self.schedule, pd.DataFrame(), choices_fixture(), weekly_protocol())


if __name__ == "__main__":
    unittest.main()
