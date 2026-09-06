import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plotting_ml_trading_signals import plot_oos_top_frac
from walk_forward import evaluate_oos_score, evaluate_oos_top_frac

#file made for out of sample evaluation and testing that it is performing as required. use similar synthetic dataset trick as before

def synthetic_oos(n_dates=12, n_tickers=10):
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []

    for date_i, date in enumerate(dates):
        for ticker_i in range(n_tickers):
            probability = ticker_i / (n_tickers - 1)
            rows.append(
                {
                    "fold": date_i // 4,
                    "date": date,
                    "ticker": f"T{ticker_i:02d}",
                    "label": int(ticker_i >= n_tickers // 2),
                    "fwd_return": (ticker_i - 4.5) / 100.0,
                    "probability": probability,
                })

    return pd.DataFrame(rows)


class OosEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oos = synthetic_oos()

    def test_top_fraction_and_bucket_arithmetic(self):
        daily = evaluate_oos_top_frac(self.oos, top_frac=0.2, min_universe=5)

        self.assertEqual(len(daily), self.oos["date"].nunique()) #each date unique
        self.assertTrue(daily["n_selected"].eq(2).all())
        self.assertFalse(daily["date"].duplicated().any()) #no duplicates
        np.testing.assert_allclose(daily["excess_return"], daily["top_return"] - daily["universe_return"])

        bucket_daily, bucket_summary = evaluate_oos_score(self.oos, n_buckets=5, min_universe=5)

        self.assertEqual(len(bucket_summary), 5)
        self.assertTrue(bucket_daily.groupby("date").size().eq(5).all())
        self.assertTrue(bucket_summary["mean_probability"].is_monotonic_increasing)
        self.assertGreater(bucket_summary["label_rate"].iloc[-1], bucket_summary["label_rate"].iloc[0])
        self.assertGreater(bucket_summary["mean_return"].iloc[-1], bucket_summary["mean_return"].iloc[0])

        figure = plot_oos_top_frac(daily)
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)

    def test_duplicate_rows_are_rejected(self):
        duplicated = pd.concat([self.oos, self.oos.iloc[[0]]], ignore_index=True)

        with self.assertRaises(ValueError):
            evaluate_oos_top_frac(duplicated)

        with self.assertRaises(ValueError):
            evaluate_oos_score(duplicated)


if __name__ == "__main__":
    unittest.main()
