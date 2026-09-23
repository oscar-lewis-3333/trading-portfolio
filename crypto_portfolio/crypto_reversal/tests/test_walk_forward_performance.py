"""Independent small-path checks for metrics, alignment and plot data."""
import sys
import unittest
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from reversal_walk_forward import development_walk_forward_rules
from walk_forward_performance import summarise_walk_forward, plot_walk_forward_equity


class WalkForwardPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {'development_start': '2022-01-01',
                         'development_end_exclusive': '2023-01-04', 'holdout_start': '2023-01-04'}
        self.rules = development_walk_forward_rules(self.protocol)
        self.dates = pd.date_range('2023-01-01', periods=3, tz='UTC')
        self.results = {}
        for name, r in [('strategy', [-.1, .2, -.1]), ('benchmark', [0., .1, -.05])]:
            frame = pd.DataFrame({'net_return': r, 'valuation_at': self.dates + pd.Timedelta(days=1),
                'equity_close_gbp': 1000 * np.cumprod(1 + np.array(r)),
                'traded_notional_gbp': [100., 0., 200.], 'equity_before_rebalance_gbp': [1000., np.nan, 1080.],
                'fee_gbp': [.09, 0., .18], 'other_cost_gbp': [.035, 0., .07],
                'accounting_error_gbp': 0., 'daily_pnl_error_gbp': 0.}, index=self.dates)
            self.results[name] = (frame, pd.DataFrame(), pd.DataFrame())

    def test_manual_metrics_include_initial_loss_and_do_not_double_charge(self):
        summary, returns, equity = summarise_walk_forward(self.results, self.protocol, self.rules)
        row = summary.loc['strategy']
        self.assertAlmostEqual(row.total_return, -.028)
        self.assertAlmostEqual(row.final_equity_gbp, 972.)
        self.assertAlmostEqual(row.cagr, .972 ** (365 / 3) - 1)
        self.assertAlmostEqual(row.annualised_volatility, np.sqrt(.03 * 365))
        self.assertAlmostEqual(row.max_drawdown, -.1)
        self.assertAlmostEqual(row.total_cost_gbp, .375)
        self.assertAlmostEqual(row.annualised_one_way_turnover, (.1 + 200/1080) * 365/3)
        self.assertAlmostEqual(row.mean_daily_net_excess_bps, (-.1 + .1 - .05) / 3 * 10000)
        self.assertEqual(returns.index[0], self.dates[0])
        self.assertEqual(equity.index[0], self.dates[0])
        self.assertEqual(equity.index[-1], pd.Timestamp('2023-01-04', tz='UTC'))
        self.assertEqual(equity.iloc[0].strategy, 1000.)
        self.assertEqual(equity.iloc[1].strategy, 900.)

    def test_identical_paths_have_zero_relative_metrics(self):
        self.results['benchmark'] = self.results['strategy']
        summary, _, _ = summarise_walk_forward(self.results, self.protocol, self.rules)
        np.testing.assert_allclose(summary.mean_daily_net_excess_bps, 0.)
        np.testing.assert_allclose(summary.sharpe_difference, 0.)

    def test_misaligned_and_holdout_rows_are_rejected(self):
        self.results['benchmark'][0].drop(self.dates[1], inplace=True)
        with self.assertRaisesRegex(ValueError, 'every walk-forward day'):
            summarise_walk_forward(self.results, self.protocol, self.rules)
        self.setUp()
        self.results['strategy'][0].loc[pd.Timestamp('2023-01-04', tz='UTC')] = self.results['strategy'][0].iloc[-1]
        with self.assertRaisesRegex(ValueError, 'every walk-forward day'):
            summarise_walk_forward(self.results, self.protocol, self.rules)

    def test_wrong_equity_or_valuation_time_is_rejected(self):
        self.results['strategy'][0].loc[self.dates[1], 'equity_close_gbp'] += 1
        with self.assertRaisesRegex(ValueError, 'reconcile'):
            summarise_walk_forward(self.results, self.protocol, self.rules)
        self.setUp()
        self.results['strategy'][0]['valuation_at'] = self.dates
        with self.assertRaisesRegex(ValueError, 'following midnight'):
            summarise_walk_forward(self.results, self.protocol, self.rules)

    def test_plot_drawdown_and_wealth_match_inputs(self):
        _, _, equity = summarise_walk_forward(self.results, self.protocol, self.rules)
        fig, axes, dd = plot_walk_forward_equity(equity, self.rules, show=False)
        try:
            np.testing.assert_allclose(dd.strategy, [0., -.1, 0., -.1])
            np.testing.assert_allclose(axes[0].lines[0].get_ydata(), equity.strategy)
            np.testing.assert_allclose(axes[1].lines[0].get_ydata(), dd.strategy * 100)
            self.assertEqual(axes[0].get_yscale(), 'log')
        finally:
            plt.close(fig)

    def test_inputs_are_unchanged(self):
        copies = {name: value[0].copy(deep=True) for name, value in self.results.items()}
        protocol = dict(self.protocol)
        summarise_walk_forward(self.results, self.protocol, self.rules)
        self.assertEqual(self.protocol, protocol)
        for name, frame in copies.items():
            pd.testing.assert_frame_equal(frame, self.results[name][0])


if __name__ == '__main__':
    unittest.main()
