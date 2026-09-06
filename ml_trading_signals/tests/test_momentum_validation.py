import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from plotting_ml_trading_signals import plot_momentum_sweep
from walk_forward import hac_test, momentum_baseline_sweep, walk_forward_momentum_rule

#this file acts as a safety check for our functions, that a momentum test selects on predictive data, chronological training, testing, working HAC tests etc.

def synthetic_pooled_data(n_dates=900, n_tickers=12, seed=7): #create artificial dataset for which the momentum method is predictive by design
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-01', periods=n_dates, name='date') 
    tickers = [f'T{i:02d}' for i in range(n_tickers)] #generate artificial tickers
    rows = []

    for date in dates: #generate momentum values and some noise
        predictive_momentum = rng.normal(size=n_tickers)
        outcome_noise = rng.normal(scale=0.003, size=n_tickers)

        for ticker_i, ticker in enumerate(tickers):
            rows.append({
                'date': date,
                'ticker': ticker,
                'return_5d': rng.normal(),
                'return_10d': rng.normal(),
                'return_21d': rng.normal(),
                'return_63d': predictive_momentum[ticker_i], #deliberately set the predictive value to 63d return (63d momentum)
                'ranking_return': 0.02 * predictive_momentum[ticker_i] + outcome_noise[ticker_i],
            })

    return pd.DataFrame(rows).set_index('date').sort_index()


class MomentumValidationTests(unittest.TestCase): #specifically for momentum tests
    @classmethod
    def setUpClass(cls):
        cls.pooled = synthetic_pooled_data() #store above data set

    def test_walk_forward(self):
        folds, daily = walk_forward_momentum_rule(self.pooled, lookback_cols=['return_5d', 'return_63d'], top_fracs=[0.2, 0.5], horizon=21, train_size=200, test_size=63, embargo=21, expanding=True,) 
        #test momentum rule on synthetic dataset using one useless feature and return_63d which is deliberately predictive


        self.assertFalse(folds.empty)
        self.assertFalse(daily.empty) #fail if either empty
        self.assertTrue((folds['lookback'] == 'return_63d').all()) #ensure each fold chose the predictive feature
        self.assertFalse(daily['date'].duplicated().any())
        self.assertTrue((folds['test_start'] > folds['train_end']).all()) #ensure no duplicated dates and that theres no overlap between test, training periods

        t_value, p_value = hac_test(daily['excess'], horizon=21)
        self.assertTrue(np.isfinite(t_value))
        self.assertTrue(np.isfinite(p_value))

    def test_sweep_plot_returns_comparison(self):
        sweep = momentum_baseline_sweep(self.pooled, horizon=21) 
        bottom_figure, universe_figure = plot_momentum_sweep(sweep) #perform, plot momentum sweep

        self.assertEqual(len(bottom_figure.axes), 3)
        self.assertEqual(len(universe_figure.axes), 3)

        for figure in (bottom_figure, universe_figure):
            first_effect_line = figure.axes[0].lines[0]
            x_values = np.asarray(first_effect_line.get_xdata())
            self.assertTrue(np.all(np.diff(x_values) > 0)) #ensure lookbacks in correct order


if __name__ == '__main__':
    unittest.main()
