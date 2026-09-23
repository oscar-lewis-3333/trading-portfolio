"""Independent accounting checks across quarterly configuration transitions."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from reversal_backtest import run_walk_forward_ledger
from reversal_walk_forward import development_walk_forward_rules


class WalkForwardLedgerTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {'development_start': '2022-01-01',
                         'development_end_exclusive': '2023-04-08',
                         'holdout_start': '2023-04-08'}
        self.rules = development_walk_forward_rules(self.protocol)
        self.dates = pd.date_range('2023-01-01', '2023-04-07', tz='UTC')
        self.days = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-04-06', '2023-04-07'], utc=True)
        self.targets = pd.DataFrame({'A': [1., 1., 0., 0.], 'B': [0., 0., 1., 1.], 'CASH': 0.}, index=self.days)
        self.prices = pd.concat([
            pd.DataFrame({'timestamp': self.dates, 'product_id': 'A',
                          'open_gbp': np.where(self.dates < pd.Timestamp('2023-04-01', tz='UTC'), 10., 20.),
                          'close_gbp': np.where(self.dates < pd.Timestamp('2023-04-01', tz='UTC'), 10., 24.)}),
            pd.DataFrame({'timestamp': self.dates, 'product_id': 'B', 'open_gbp': 10., 'close_gbp': 12.})], ignore_index=True)
        self.schedule = pd.DataFrame({'day': self.days, 'execution_at': self.days + pd.Timedelta(minutes=8),
                                      'status': 'ready', 'required_assets': 2})
        self.refs = pd.DataFrame([
            {'day': row.day, 'execution_at': row.execution_at, 'product_id': asset,
             'reference_bar_start': row.execution_at - pd.Timedelta(minutes=1),
             'assumed_available_at': row.execution_at, 'reference_price_gbp': value}
            for i, row in self.schedule.iterrows()
            for asset, value in [('A', 10. if i < 2 else 22.), ('B', 10.)]])

    def run_ledger(self):
        return run_walk_forward_ledger(self.prices, self.targets, self.schedule, self.refs,
                                       self.protocol, self.rules)

    def test_carries_units_until_delayed_switch_and_charges_net_costs(self):
        ledger, trades, positions = self.run_ledger()
        c = .00125
        a = 1000. / (10. * (1. + c))
        b = a * 22. * (1. - c) / ((1. + c) * 10.)
        april1 = pd.Timestamp('2023-04-01', tz='UTC')
        april6 = self.days[2]
        self.assertEqual(len(ledger), 97)
        self.assertFalse(ledger.loc[april1, 'scheduled_rebalance'])
        self.assertAlmostEqual(positions.loc[april1, 'A'], a)
        self.assertAlmostEqual(ledger.loc[april1, 'equity_close_gbp'], a * 24.)
        self.assertAlmostEqual(ledger.loc[april6, 'equity_before_rebalance_gbp'], a * 22.)
        self.assertAlmostEqual(ledger.iloc[-1].equity_close_gbp, b * 12.)
        switch = trades.loc[trades.decision_at.eq(april6)].set_index('product_id')
        self.assertAlmostEqual(switch.loc['A', 'signed_notional_gbp'], -a * 22.)
        self.assertAlmostEqual(switch.loc['B', 'signed_notional_gbp'], b * 10.)
        self.assertAlmostEqual(ledger.loc[april6, 'fee_gbp'] + ledger.loc[april6, 'other_cost_gbp'],
                               c * (a * 22. + b * 10.))
        self.assertLess(ledger.loc[self.days[1], 'traded_notional_gbp'], 1e-9)
        self.assertLess(ledger.loc[self.days[3], 'traded_notional_gbp'], 1e-9)
        self.assertAlmostEqual(ledger.net_return.iloc[0], 1 / (1 + c) - 1)
        self.assertGreater(positions.iloc[-1].B, 0.)  #No artificial terminal liquidation.

    def test_reconstructs_every_day_from_signed_trades(self):
        ledger, trades, positions = self.run_ledger()
        units = pd.Series(0., index=['A', 'B']); cash = 1000.
        for day in self.dates:
            for trade in trades.loc[trades.decision_at.eq(day)].itertuples():
                self.assertAlmostEqual(units[trade.product_id], trade.units_before)
                units[trade.product_id] += trade.signed_notional_gbp / trade.reference_price_gbp
                cash -= trade.signed_notional_gbp + trade.fee_gbp + trade.other_cost_gbp
            closes = self.prices.loc[self.prices.timestamp.eq(day)].set_index('product_id').close_gbp
            self.assertAlmostEqual(cash + (units * closes).sum(), ledger.loc[day, 'equity_close_gbp'])
            np.testing.assert_allclose(units, positions.loc[day], atol=1e-12)
        self.assertLess(ledger.daily_pnl_error_gbp.abs().max(), 1e-9)

    def test_cash_entry_and_empty_universe_exit(self):
        self.targets.loc[self.days[0]] = [0., 0., 1.]
        self.targets.loc[self.days[2]:] = [0., 0., 1.]
        ledger, trades, positions = self.run_ledger()
        self.assertEqual(ledger.iloc[0].equity_close_gbp, 1000.)
        self.assertEqual(ledger.iloc[0].net_return, 0.)
        expected = 1000 / 1.00125 / 10 * 22 * .99875
        np.testing.assert_allclose(ledger.loc[self.days[2]:, 'equity_close_gbp'], expected)
        self.assertTrue(positions.loc[self.days[2]:].eq(0).all().all())

    def test_missing_exit_reference_is_rejected_even_with_matching_counts(self):
        self.refs = self.refs.loc[~(self.refs.day.eq(self.days[2]) & self.refs.product_id.eq('A'))]
        self.schedule.loc[2, 'required_assets'] = 1
        with self.assertRaisesRegex(ValueError, 'invalid execution prices'):
            self.run_ledger()

    def test_missing_held_valuation_during_wait_is_rejected(self):
        self.prices = self.prices.loc[~(self.prices.timestamp.eq(pd.Timestamp('2023-04-02', tz='UTC')) & self.prices.product_id.eq('A'))]
        with self.assertRaisesRegex(ValueError, 'invalid opening prices'):
            self.run_ledger()

    def test_holdout_and_missing_initial_event_are_rejected(self):
        extra = self.prices.iloc[[0]].copy(); extra['timestamp'] = pd.Timestamp('2023-04-08', tz='UTC')
        self.prices = pd.concat([self.prices, extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'restricted to development'):
            self.run_ledger()
        self.setUp(); self.targets = self.targets.iloc[1:]
        with self.assertRaisesRegex(ValueError, 'first evaluation day'):
            self.run_ledger()

    def test_future_prices_do_not_change_earlier_path(self):
        before = self.run_ledger()[0]
        self.prices.loc[self.prices.timestamp.ge(self.days[2]), ['open_gbp', 'close_gbp']] *= 2
        after = self.run_ledger()[0]
        pd.testing.assert_frame_equal(before.loc[:self.days[2] - pd.Timedelta(days=1)],
                                      after.loc[:self.days[2] - pd.Timedelta(days=1)])

    def test_inputs_are_not_mutated_and_rules_cannot_drift(self):
        frames = [self.prices, self.targets, self.schedule, self.refs]
        copies = [frame.copy(deep=True) for frame in frames]
        self.run_ledger()
        for before, after in zip(copies, frames): pd.testing.assert_frame_equal(before, after)
        self.rules['fee_bps'] = 0.
        with self.assertRaisesRegex(ValueError, 'rules differ'):
            self.run_ledger()


if __name__ == '__main__':
    unittest.main()
