"""Selection weights, timing and gap checks for reversal signals."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversal_signals import rank_reversal_candidates


class ReversalSignalTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "development_start": "2022-01-03", "development_end_exclusive": "2022-01-08",
            "signal": {"formation_days": 1, "selection_fraction": 0.5, "negative_return_required": False},
        }
        trajectories = {"A-USD": [100, 90, 99], "B-USD": [100, 95, 95],
                        "C-USD": [100, 105, 100], "D-USD": [100, 110, 99]}
        self.prices = pd.DataFrame([
            {"product_id": product, "timestamp": timestamp, "close": float(close)}
            for product, first in trajectories.items()
            for timestamp, close in zip(pd.date_range("2022-01-01", periods=10, tz="UTC"), first + [first[-1]] * 7)
        ])
        self.members = self.prices.loc[self.prices["timestamp"].between(
            pd.Timestamp("2022-01-02", tz="UTC"), pd.Timestamp("2022-01-06", tz="UTC")
        )].copy()
        self.members["decision_at"] = self.members["timestamp"] + pd.Timedelta(days=1)
        self.members["liquidity_rank"] = self.members.groupby("decision_at").cumcount() + 1
        self.coverage = pd.DataFrame({"active": True, "universe_size": 4},
            index=pd.date_range("2022-01-03", "2022-01-07", tz="UTC", name="decision_at"))

    def rank(self):
        return rank_reversal_candidates(self.prices, self.members, self.protocol)

    def first_decision(self):
        ranked = self.rank()
        return ranked.loc[ranked["decision_at"].eq(self.coverage.index[0])].copy()


    def test_known_losers_and_equal_weights(self):
        ranked = self.first_decision()
        self.assertEqual(ranked["product_id"].tolist(), ["A-USD", "B-USD", "C-USD", "D-USD"])
        np.testing.assert_allclose(ranked["formation_return"], [-.1, -.05, .05, .1])
        np.testing.assert_allclose(ranked["reversal_weight"], [.5, .5, 0, 0])
        np.testing.assert_allclose(ranked["benchmark_weight"], [.25] * 4)


    def test_ceiling_ties_and_no_negative_return_requirement(self):
        self.prices["close"] = np.where(self.prices["timestamp"].eq(pd.Timestamp("2022-01-01", tz="UTC")), 100., 110.)
        self.members["close"] = 110.
        self.protocol["signal"]["selection_fraction"] = .3  # ceil(4 * .3) = 2
        ranked = self.first_decision()
        self.assertTrue(ranked["formation_return"].gt(0).all())
        self.assertEqual(ranked.loc[ranked["selected"], "product_id"].tolist(), ["A-USD", "B-USD"])

    def test_future_prices_cannot_change_earlier_selection(self):
        baseline = self.first_decision()
        prices = self.prices.copy()
        prices.loc[prices["timestamp"].ge(pd.Timestamp("2022-01-03", tz="UTC")), "close"] *= 100
        members = self.members.loc[self.members["decision_at"].eq(self.coverage.index[0])]
        changed = rank_reversal_candidates(prices, members, self.protocol)
        pd.testing.assert_frame_equal(baseline.reset_index(drop=True), changed)

    def test_formation_cannot_bridge_gap(self):
        prices = self.prices.loc[~(self.prices["product_id"].eq("A-USD") & self.prices["timestamp"].eq(pd.Timestamp("2022-01-01", tz="UTC")))]
        with self.assertRaisesRegex(ValueError, "contiguous formation"):
            rank_reversal_candidates(prices, self.members, self.protocol)







if __name__ == "__main__":
    unittest.main()
