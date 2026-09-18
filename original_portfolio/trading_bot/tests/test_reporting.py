import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from plotting_trading_bot import plot_rebalance_history
from reporting import ORDER_COLUMNS, REBALANCE_COLUMNS, latest_rebalance_summary, load_rebalance_history

#file dedicating to ensuring the reporting system works as intended, including missing data scenarios and many more.

def completed_archive(rebalance_id, completed_at='2024-01-03T15:00:00+00:00', signal_date='2024-01-02', execution_date='2024-01-03', ticker='A', side='buy', snapshot=True):
    universe = ['A', 'B', 'C', 'D', 'E']
    strategy = {
        'lookback_days': 63,
        'top_frac': 0.2,
        'rebalance_days': 21,
        'universe': universe,
        'weighting': 'equal_at_rebalance'
    }
    execution_policy = {
        'cash_reserve_fraction': 0.01,
        'execution_window_minutes': 15,
        'price_max_age_seconds': 60.0,
        'order_type': 'market',
        'time_in_force': 'day',
        'extended_hours': False
    }
    proposed_state = {
        'schema_version': 2,
        'strategy': strategy,
        'selection_date': signal_date,
        'tickers': [ticker]
    }
    execution_open = f'{execution_date}T09:30:00-05:00'
    execution_window_end = f'{execution_date}T09:45:00-05:00'
    submission_quote_at = f'{execution_date}T09:34:00-05:00'
    submission_attempted_at = f'{execution_date}T09:35:00-05:00'
    reference_price = 100.0
    fill_price = 101.0 if side == 'buy' else 99.0
    order = {'ticker': ticker,
        'side': side,
        'qty': 10.0,
        'reference_price': reference_price,
        'submission_reference_price': reference_price,
        'submission_quote_at': submission_quote_at,
        'reason': 'rebalance',
        'client_order_id': f'mom-{rebalance_id}-00',
        'broker_order_id': f'broker-{rebalance_id}',
        'status': 'filled',
        'filled_qty': 10.0,
        'filled_avg_price': fill_price,
        'submission_attempted_at': submission_attempted_at
    }
    completed_state = {
        **proposed_state,
        'rebalance_id': rebalance_id,
        'account_id': 'paper-account',
        'planned_execution_date': execution_date,
        'execution_policy': execution_policy,
        'completed_at': completed_at,
        'positions_at_completion': {ticker: 10.0}
    }
    if snapshot:
        completed_state['account_snapshot'] = {
            'cash': 100.0,
            'equity': 10_000.0,
            'portfolio_value': 10_000.0
        }

    return {
        'schema_version': 2,
        'rebalance_id': rebalance_id,
        'account_id': 'paper-account',
        'status': 'completed',
        'signal_date': signal_date,
        'execution_date': execution_date,
        'execution_open': execution_open,
        'execution_window_end': execution_window_end,
        'execution_policy': execution_policy,
        'proposed_state': proposed_state,
        'starting_positions': {} if side == 'buy' else {ticker: 20.0},
        'expected_positions': {ticker: 10.0},
        'orders': [order],
        'completed_state': completed_state,
        'completed_at': completed_at
    }


def write_archive(directory, archive, filename=None):
    path = Path(directory) / f"{filename or archive['rebalance_id']}.json"
    path.write_text(json.dumps(archive), encoding='utf-8')
    return path


class ReportingTests(unittest.TestCase):
    def test_missing_or_empty_history_returns_typed_empty_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing'
            rebalances, orders = load_rebalance_history(missing)

            self.assertTrue(rebalances.empty)
            self.assertTrue(orders.empty)
            self.assertEqual(list(rebalances.columns), REBALANCE_COLUMNS)
            self.assertEqual(list(orders.columns), ORDER_COLUMNS)
            self.assertEqual(latest_rebalance_summary(rebalances, orders), 'No completed rebalances')
            with self.assertRaisesRegex(ValueError, 'No completed rebalances'):
                plot_rebalance_history(rebalances, orders)

    def test_loader_orders_archives_and_calculates_execution_metrics(self):
        early_id = '00000000000000000000000000000001'
        late_id = '00000000000000000000000000000002'

        with tempfile.TemporaryDirectory() as directory:
            late = completed_archive(late_id,
                completed_at='2024-02-02T15:00:00+00:00',
                signal_date='2024-02-01',
                execution_date='2024-02-02',
                ticker='B',
                side='sell')
            early = completed_archive(early_id)
            write_archive(directory, late)
            write_archive(directory, early)

            rebalances, orders = load_rebalance_history(directory)

        self.assertEqual(rebalances['rebalance_id'].tolist(), [early_id, late_id])
        self.assertEqual(len(orders), 2)
        self.assertEqual(rebalances['n_orders'].tolist(), [1, 1])
        self.assertEqual(rebalances['n_buys'].tolist(), [1, 0])
        self.assertEqual(rebalances['n_sells'].tolist(), [0, 1])
        self.assertAlmostEqual(rebalances.iloc[0]['buy_notional'], 1010.0)
        self.assertAlmostEqual(rebalances.iloc[1]['sell_notional'], 990.0)
        self.assertAlmostEqual(orders.iloc[0]['adverse_slippage_bps'], 100.0)
        self.assertAlmostEqual(orders.iloc[1]['adverse_slippage_bps'], 100.0)

        summary = latest_rebalance_summary(rebalances, orders)
        self.assertIn('2024-02-02 15:00 UTC', summary)
        self.assertIn('63-session momentum, top 20%', summary)
        self.assertIn('B (10)', summary)
        self.assertIn('1 sells', summary)

    def test_completion_inside_execution_window_after_submission_is_valid(self):
        rebalance_id = '00000000000000000000000000000009'
        archive = completed_archive(rebalance_id, completed_at='2024-01-03T14:40:00+00:00')

        with tempfile.TemporaryDirectory() as directory:
            write_archive(directory, archive)
            rebalances, orders = load_rebalance_history(directory)

        self.assertEqual(rebalances['rebalance_id'].tolist(), [rebalance_id])
        self.assertEqual(orders['rebalance_id'].tolist(), [rebalance_id])

    def test_invalid_archive_timestamp_sequences_are_rejected(self):
        rebalance_id = '00000000000000000000000000000010'
        valid = completed_archive(rebalance_id)

        stale_quote = copy.deepcopy(valid)
        stale_quote['orders'][0]['submission_quote_at'] = '2024-01-03T09:33:00-05:00'
        completion_before_submission = copy.deepcopy(valid)
        completion_before_submission['completed_at'] = '2024-01-03T14:34:00+00:00'
        
        completion_before_submission['completed_state']['completed_at'] = '2024-01-03T14:34:00+00:00'

        confirmation_after_completion = copy.deepcopy(valid)
        confirmation_after_completion['orders'] = []
        confirmation_after_completion['starting_positions'] = {'A': 10.0}
        confirmation_after_completion['completed_at'] = '2024-01-03T14:34:00+00:00'
        confirmation_after_completion['completed_state']['completed_at'] = '2024-01-03T14:34:00+00:00'
        confirmation_after_completion['no_order_confirmed_at'] = '2024-01-03T09:35:00-05:00'
        cases = [(stale_quote, 'submission quote exceeds its maximum age'),
            (completion_before_submission, 'submission timestamp follows completion'),
            (confirmation_after_completion,
             'no-order confirmation lies outside its execution window')]

        for archive, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                write_archive(directory, archive)
                with self.assertRaisesRegex(ValueError, message):
                    load_rebalance_history(directory)

    def test_weighted_slippage_uses_reference_notional_for_both_sides(self):
        rebalance_id = '00000000000000000000000000000008'
        archive = completed_archive(rebalance_id, ticker='B', side='buy')
        sell_order = copy.deepcopy(archive['orders'][0])
        sell_order.update({
            'ticker': 'A',
            'side': 'sell',
            'client_order_id': f'mom-{rebalance_id}-01',
            'broker_order_id': f'broker-{rebalance_id}-sell',
            'filled_avg_price': 99.0
        })
        archive['orders'].insert(0, sell_order)
        archive['starting_positions'] = {'A': 10.0}

        with tempfile.TemporaryDirectory() as directory:
            write_archive(directory, archive)
            rebalances, orders = load_rebalance_history(directory)

        self.assertEqual(orders['adverse_slippage_bps'].tolist(), [100.0, 100.0])
        self.assertAlmostEqual(rebalances.iloc[0]['weighted_adverse_slippage_bps'], 100.0)
        self.assertAlmostEqual(orders['reference_notional'].sum(), 2000.0)
        self.assertAlmostEqual(orders['adverse_shortfall'].sum(), 20.0)

    def test_missing_snapshot_is_rejected(self):
        rebalance_id = '00000000000000000000000000000003'
        archive = completed_archive(rebalance_id, snapshot=False)

        with tempfile.TemporaryDirectory() as directory:
            write_archive(directory, archive)
            with self.assertRaisesRegex(ValueError, 'invalid account snapshot'):
                load_rebalance_history(directory)

    def test_no_order_completion_is_a_valid_history_entry(self):
        rebalance_id = '00000000000000000000000000000007'
        archive = completed_archive(rebalance_id)
        archive['orders'] = []
        archive['starting_positions'] = {'A': 10.0}
        archive['no_order_confirmed_at'] = '2024-01-03T09:35:00-05:00'

        with tempfile.TemporaryDirectory() as directory:
            write_archive(directory, archive)
            rebalances, orders = load_rebalance_history(directory)

        self.assertEqual(rebalances.iloc[0]['n_orders'], 0)
        self.assertEqual(rebalances.iloc[0]['gross_traded_notional'], 0.0)
        self.assertTrue(np.isnan(rebalances.iloc[0]['weighted_adverse_slippage_bps']))
        self.assertTrue(orders.empty)
        self.assertNotIn('adverse slippage', latest_rebalance_summary(rebalances, orders).lower())

    def test_invalid_archives_fail_loudly(self):
        rebalance_id = '00000000000000000000000000000004'
        valid = completed_archive(rebalance_id)
        cases = {}

        wrong_account = copy.deepcopy(valid)
        wrong_account['completed_state']['account_id'] = 'other-account'
        cases['identity mismatch'] = (wrong_account, rebalance_id)

        wrong_quantity = copy.deepcopy(valid)
        wrong_quantity['orders'][0]['filled_qty'] = 9.0
        cases['quantity mismatch'] = (wrong_quantity, rebalance_id)

        bad_snapshot = copy.deepcopy(valid)
        bad_snapshot['completed_state']['account_snapshot'] = {}
        cases['malformed snapshot'] = (bad_snapshot, rebalance_id)

        wrong_reconciliation = copy.deepcopy(valid)
        wrong_reconciliation['starting_positions'] = {'A': 1.0}
        cases['holdings do not reconcile'] = (wrong_reconciliation, rebalance_id)

        wrong_schema = copy.deepcopy(valid)
        wrong_schema['schema_version'] = 1
        cases['wrong schema'] = (wrong_schema, rebalance_id)

        incomplete = copy.deepcopy(valid)
        incomplete['orders'][0]['status'] = 'partially_filled'
        cases['incomplete order'] = (incomplete, rebalance_id)

        bad_price = copy.deepcopy(valid)
        bad_price['orders'][0]['submission_reference_price'] = 0.0
        cases['bad reference price'] = (bad_price, rebalance_id)

        bad_timestamp = copy.deepcopy(valid)
        bad_timestamp['orders'][0]['submission_attempted_at'] = '2024-01-03T10:00:00-05:00'
        cases['submission outside window'] = (bad_timestamp, rebalance_id)

        cases['filename mismatch'] = copy.deepcopy(valid), '00000000000000000000000000000005'

        for name, (archive, filename) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                write_archive(directory, archive, filename=filename)
                with self.assertRaisesRegex(ValueError, 'Invalid completed rebalance archive'):
                    load_rebalance_history(directory)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f'{rebalance_id}.json'
            path.write_text('{bad json', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Could not read rebalance archive'):
                load_rebalance_history(directory)

        with tempfile.TemporaryDirectory() as directory:
            legacy_log = Path(directory) / 'bot_log.jsonl'
            legacy_log.write_text('{}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'must be a directory'):
                load_rebalance_history(legacy_log)

    def test_plot_uses_rebalance_and_order_history(self):
        rebalance_id = '00000000000000000000000000000006'

        with tempfile.TemporaryDirectory() as directory:
            write_archive(directory, completed_archive(rebalance_id))
            rebalances, orders = load_rebalance_history(directory)

        figure = plot_rebalance_history(rebalances, orders)
        axes = figure.axes
        self.assertEqual(len(axes), 3)
        self.assertEqual(axes[0].get_ylabel(), 'Portfolio value ($)')
        self.assertEqual(axes[1].get_ylabel(), 'Filled notional ($)')
        self.assertEqual(axes[2].get_ylabel(), 'Adverse slippage (bps)')
        matplotlib.pyplot.close(figure)


if __name__ == '__main__':
    unittest.main()
