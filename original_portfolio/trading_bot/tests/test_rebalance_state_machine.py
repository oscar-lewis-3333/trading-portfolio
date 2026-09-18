import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from alpaca.common.exceptions import APIError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from broker import complete_rebalance, prepare_rebalance, reconcile_pending_orders, submit_pending_order
from pipeline import advance_pending_rebalance
from reporting import load_rebalance_history


#file dedicated to testing that all rebalance tools act as they're supposed to. fake client does what it says, we then create a pending order, then try a sequence of tests to ensure it satisfies all necessary conditions.

def missing_order_error():
    http_error = SimpleNamespace(response=SimpleNamespace(status_code=404), request=None)
    return APIError('{"code": 404, "message": "not found"}', http_error)


def broker_order(saved_order, status='filled', order_type='market'):
    filled_qty = saved_order['qty'] if status == 'filled' else 0.0
    filled_price = saved_order['reference_price'] if filled_qty else None
    return SimpleNamespace(id=f"broker-{saved_order['client_order_id']}", client_order_id=saved_order['client_order_id'], symbol=saved_order['ticker'], side=saved_order['side'], qty=saved_order['qty'], filled_qty=filled_qty, filled_avg_price=filled_price, status=status, order_type=order_type, type=order_type, time_in_force='day', extended_hours=False)


def position(ticker, quantity):
    return SimpleNamespace(symbol=ticker, qty=str(quantity), qty_available=str(quantity), side='long')


class FakeClient:
    def __init__(self, timestamp='2024-01-03 09:35', positions=None):
        self.account = SimpleNamespace(id='paper-account', status='ACTIVE', account_blocked=False, trading_blocked=False, trade_suspended_by_user=False, cash='10000', equity='10000', portfolio_value='10000')
        self.clock = SimpleNamespace(timestamp=pd.Timestamp(timestamp, tz='America/New_York'), is_open=True)
        self.orders = {}
        self.positions = positions if positions is not None else [[]]
        self.submitted = []

    def get_account(self):
        return self.account

    def get_clock(self):
        return self.clock

    def get_calendar(self, filters=None):
        return [SimpleNamespace(open=pd.Timestamp('2024-01-03 09:30').to_pydatetime(), close=pd.Timestamp('2024-01-03 16:00').to_pydatetime())]

    def get_order_by_client_id(self, client_order_id):
        if client_order_id not in self.orders:
            raise missing_order_error()
        return self.orders[client_order_id]

    def get_orders(self, filter=None):
        active = {'accepted', 'pending_new', 'new', 'partially_filled', 'accepted_for_bidding'}
        return [order for order in self.orders.values() if order.status in active]

    def get_all_positions(self):
        if len(self.positions) > 1:
            return self.positions.pop(0)
        return self.positions[0]

    def get_asset(self, ticker):
        return SimpleNamespace(tradable=True, fractionable=True)

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        response = SimpleNamespace(id=f'broker-{order_data.client_order_id}', client_order_id=order_data.client_order_id, status='new')
        return response


def create_pending(directory, orders, selected, current_positions=None, top_frac=0.2):
    directory = Path(directory)
    state_path = directory / 'basket_state.json'
    pending_path = directory / 'pending_rebalance.json'
    plan = {
        'signal_date': '2024-01-02',
        'rebalance_needed': True,
        'state_path': str(state_path),
        'execution_policy': {
            'cash_reserve_fraction': 0.01,
            'execution_window_minutes': 15,
            'price_max_age_seconds': 60.0,
            'order_type': 'market',
            'time_in_force': 'day',
            'extended_hours': False,
        },
        'proposed_state': {
            'schema_version': 2,
            'strategy': {
                'lookback_days': 63,
                'top_frac': top_frac,
                'rebalance_days': 21,
                'universe': ['A', 'B', 'C', 'D', 'E'],
                'weighting': 'equal_at_rebalance',
            },
            'selection_date': '2024-01-02',
            'tickers': selected}}
    pending = prepare_rebalance(plan, orders, account_id='paper-account', execution_date=pd.Timestamp('2024-01-03'), execution_open=pd.Timestamp('2024-01-03 09:30', tz='America/New_York'), execution_window_end=pd.Timestamp('2024-01-03 09:45', tz='America/New_York'), current_positions=current_positions or {}, pending_path=pending_path)
    return state_path, pending_path, pending


def record_submission_attempts(pending_path):
    pending = json.loads(Path(pending_path).read_text(encoding='utf-8'))
    for order in pending['orders']:
        order['status'] = 'submitting'
        order['submission_reference_price'] = order['reference_price']
        order['submission_quote_at'] = pd.Timestamp(
            '2024-01-03 09:34:30', tz='America/New_York'
        ).isoformat()
        order['submission_attempted_at'] = pd.Timestamp(
            '2024-01-03 09:35', tz='America/New_York'
        ).tz_convert('UTC').isoformat()
    Path(pending_path).write_text(json.dumps(pending), encoding='utf-8')
    return pending


class RebalanceStateMachineTests(unittest.TestCase):
    def test_reconciliation_accepts_known_states_and_rejects_terminal_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            client = FakeClient()

            refreshed = reconcile_pending_orders(client, pending_path)
            self.assertEqual(refreshed['status'], 'prepared')

            pending = record_submission_attempts(pending_path)
            saved_order = pending['orders'][0]
            client.orders[saved_order['client_order_id']] = broker_order(saved_order, status='new')
            refreshed = reconcile_pending_orders(client, pending_path)
            self.assertEqual(refreshed['status'], 'in_progress')

            client.orders[saved_order['client_order_id']] = broker_order(saved_order, status='canceled')
            refreshed = reconcile_pending_orders(client, pending_path)
            self.assertEqual(refreshed['status'], 'needs_attention')
            self.assertIn('canceled', refreshed['attention_reason'])

    def test_reconciliation_rejects_unrecorded_broker_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            client = FakeClient()
            saved_order = pending['orders'][0]
            client.orders[saved_order['client_order_id']] = broker_order(saved_order, status='new')

            with self.assertRaisesRegex(
                    RuntimeError, 'without a recorded submission attempt'):
                reconcile_pending_orders(client, pending_path)

            saved = json.loads(pending_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'needs_attention')
            self.assertEqual(saved['halt_code'], 'unrecorded_submission')

    def test_reconciliation_rejects_changed_order_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(
                directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            client = FakeClient()
            pending = record_submission_attempts(pending_path)
            saved_order = pending['orders'][0]
            client.orders[saved_order['client_order_id']] = broker_order(saved_order, order_type='limit')

            with self.assertRaisesRegex(RuntimeError, 'differs from the saved plan'):
                reconcile_pending_orders(client, pending_path)

            saved = json.loads(pending_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'needs_attention')
            self.assertEqual(saved['halt_code'], 'broker_order_mismatch')

    def test_reconciliation_persists_an_inconsistent_partial_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            client = FakeClient()
            pending = record_submission_attempts(pending_path)
            saved_order = pending['orders'][0]
            client.orders[saved_order['client_order_id']] = broker_order(saved_order, status='partially_filled')

            with self.assertRaisesRegex(RuntimeError, 'inconsistent filled quantity'):
                reconcile_pending_orders(client, pending_path)

            saved = json.loads(pending_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'needs_attention')
            self.assertEqual(saved['halt_code'], 'inconsistent_partial_fill')

    def test_direct_submission_persists_a_terminal_preflight_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            client = FakeClient()
            client.account.trading_blocked = True

            with self.assertRaisesRegex(RuntimeError, 'cannot currently trade'):
                submit_pending_order(client, pending_path, pending['orders'][0]['client_order_id'])

            saved = json.loads(pending_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'needs_attention')
            self.assertEqual(saved['halt_code'], 'account_cannot_trade')

    def test_missed_submission_window_is_saved_as_attention(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            client = FakeClient(timestamp='2024-01-03 10:00')

            with self.assertRaisesRegex(RuntimeError, 'execution window has passed'):
                submit_pending_order(client, pending_path, pending['orders'][0]['client_order_id'])

            saved = json.loads(pending_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'needs_attention')
            self.assertEqual(saved['halt_code'], 'missed_execution_window')

    def test_submission_saves_the_current_quote_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 95.0}],
                ['A'])
            client = FakeClient()
            saved_order = pending['orders'][0]

            def submit_order(order_data):
                broker_response = broker_order(saved_order, status='new')
                client.orders[saved_order['client_order_id']] = broker_response
                return broker_response

            client.submit_order = submit_order
            quote_client = SimpleNamespace(get_stock_latest_quote=lambda request: {
                'A': SimpleNamespace(bid_price=99.0, ask_price=100.0, timestamp=pd.Timestamp('2024-01-03 09:34:30', tz='America/New_York'))})

            with patch('broker.get_data_client', return_value=quote_client):
                result, submitted = submit_pending_order(client, pending_path, saved_order['client_order_id'], env_path=Path(directory) / '.env')

            self.assertTrue(submitted)
            self.assertEqual(result['status'], 'in_progress')
            saved = json.loads(pending_path.read_text(encoding='utf-8'))['orders'][0]
            self.assertEqual(saved['reference_price'], 95.0)
            self.assertEqual(saved['submission_reference_price'], 100.0)
            self.assertEqual(pd.Timestamp(saved['submission_quote_at']), pd.Timestamp('2024-01-03 09:34:30', tz='America/New_York'))
            self.assertEqual(pd.Timestamp(saved['submission_attempted_at']), pd.Timestamp('2024-01-03 09:35', tz='America/New_York'))

    def test_submission_refreshes_clock_after_quote_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 95.0}],
                ['A'])
            client = FakeClient()
            saved_order = pending['orders'][0]
            events = []
            clocks = iter([
                SimpleNamespace(timestamp=pd.Timestamp('2024-01-03 09:35:00', tz='America/New_York'), is_open=True),
                SimpleNamespace(timestamp=pd.Timestamp('2024-01-03 09:35:06', tz='America/New_York'), is_open=True),
                SimpleNamespace(timestamp=pd.Timestamp('2024-01-03 09:35:07', tz='America/New_York'), is_open=True),
            ])

            def get_clock():
                events.append('clock')
                return next(clocks)

            def submit_order(order_data):
                broker_response = broker_order(saved_order, status='new')
                client.orders[saved_order['client_order_id']] = broker_response
                return broker_response

            def get_quotes(request):
                events.append('quotes')
                return {'A': SimpleNamespace(bid_price=99.0, ask_price=100.0, timestamp=pd.Timestamp('2024-01-03 09:35:05', tz='America/New_York'))}

            client.get_clock = get_clock
            client.submit_order = submit_order
            quote_client = SimpleNamespace(get_stock_latest_quote=get_quotes)

            with patch('broker.get_data_client', return_value=quote_client):
                result, submitted = submit_pending_order(client, pending_path, saved_order['client_order_id'], env_path=Path(directory) / '.env')

            self.assertTrue(submitted)
            self.assertEqual(result['status'], 'in_progress')
            self.assertEqual(events, ['clock', 'quotes', 'clock', 'clock'])

    def test_completion_waits_for_position_propagation_and_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path, pending_path, pending = create_pending(directory,[
                    {'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0},
                    {'ticker': 'B', 'side': 'buy', 'qty': 2.0, 'reference_price': 200.0},
                ],
                ['A', 'B'], top_frac=0.4)
            client = FakeClient(positions=[[], [position('A', 1.0), position('B', 2.0)]])

            pending = record_submission_attempts(pending_path)
            for saved_order in pending['orders']:
                client.orders[saved_order['client_order_id']] = broker_order(saved_order)
            with patch('broker.get_data_client', return_value=SimpleNamespace()):
                result = advance_pending_rebalance(client, pending_path, env_path=Path(directory) / '.env', max_wait_seconds=1, poll_interval_seconds=0.001)

            self.assertEqual(result['status'], 'completed')
            self.assertFalse(pending_path.exists())
            self.assertTrue(state_path.exists())

            completed = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual(completed['positions_at_completion'], {'A': 1.0, 'B': 2.0})
            self.assertEqual(completed['account_snapshot'], {
                'cash': 10000.0,
                'equity': 10000.0,
                'portfolio_value': 10000.0
            })
            archive = Path(directory) / 'rebalance_history' / f"{pending['rebalance_id']}.json"
            self.assertTrue(archive.exists())
            archived = json.loads(archive.read_text(encoding='utf-8'))
            self.assertEqual(archived['completed_state']['account_snapshot'], completed['account_snapshot'])
            rebalances, orders = load_rebalance_history(archive.parent)
            self.assertEqual(len(rebalances), 1)
            self.assertEqual(len(orders), 2)
            self.assertEqual(rebalances.iloc[0]['n_orders'], 2)
            self.assertEqual(rebalances.iloc[0]['tickers'], ('A', 'B'))
            self.assertEqual(set(orders['ticker']), {'A', 'B'})

    def test_completion_persists_position_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, _ = create_pending(directory,
                [],
                ['A'],
                current_positions={'A': 1.0})
            result, completed = complete_rebalance(FakeClient(), pending_path)

            self.assertFalse(completed)
            self.assertEqual(result['position_mismatches'], ['A'])

            saved = json.loads(pending_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['position_mismatches'], ['A'])

    def test_archive_recovery_after_link_before_pending_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path, pending_path, pending = create_pending(directory,
                [],
                ['A'],
                current_positions={'A': 1.0})
            client = FakeClient(positions=[[position('A', 1.0)]])
            archive = Path(directory) / 'rebalance_history' / f"{pending['rebalance_id']}.json"
            path_type = type(pending_path)
            original_unlink = path_type.unlink

            def interrupt_pending_removal(path, *args, **kwargs):
                if path.resolve() == pending_path.resolve() and archive.exists():
                    raise RuntimeError('simulated archive crash')
                return original_unlink(path, *args, **kwargs)

            with patch.object(path_type, 'unlink', interrupt_pending_removal):
                with self.assertRaisesRegex(RuntimeError, 'simulated archive crash'):
                    complete_rebalance(client, pending_path)

            self.assertTrue(state_path.exists())
            self.assertTrue(pending_path.exists())
            self.assertTrue(archive.exists())
            frozen_snapshot = json.loads(state_path.read_text(encoding='utf-8'))['account_snapshot']

            client.account.cash = '9000'
            client.account.equity = '11000'
            client.account.portfolio_value = '11000'

            reconcile_pending_orders(client, pending_path)
            result, completed = complete_rebalance(client, pending_path)

            self.assertTrue(completed)
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(pending_path.exists())
            self.assertTrue(archive.exists())
            recovered = json.loads(archive.read_text(encoding='utf-8'))
            self.assertEqual(recovered['completed_state']['account_snapshot'], frozen_snapshot)

    def test_invalid_completion_snapshot_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path, pending_path, _ = create_pending(directory,
                [],
                ['A'],
                current_positions={'A': 1.0})
            client = FakeClient(positions=[[position('A', 1.0)]])
            client.account.cash = True

            with self.assertRaisesRegex(RuntimeError, 'retry completion'):
                complete_rebalance(client, pending_path)

            self.assertFalse(state_path.exists())
            self.assertTrue(pending_path.exists())
            self.assertNotEqual(json.loads(pending_path.read_text(encoding='utf-8'))['status'], 'needs_attention')
            self.assertIn('no_order_confirmed_at', json.loads(pending_path.read_text(encoding='utf-8')))

            client.account.cash = '10000'
            client.clock.timestamp = pd.Timestamp('2024-01-03 10:00', tz='America/New_York')
            with patch('broker.get_data_client', return_value=SimpleNamespace()):
                result = advance_pending_rebalance(client, pending_path, env_path=Path(directory) / '.env', max_wait_seconds=1, poll_interval_seconds=0.001)

            self.assertEqual(result['status'], 'completed')

    def test_completion_stores_the_final_account_read(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path, pending_path, _ = create_pending(directory,
                [],
                ['A'],
                current_positions={'A': 1.0})
            client = FakeClient(positions=[[position('A', 1.0)]])
            account_reads = [SimpleNamespace(id='paper-account', cash='10000', equity='10000',portfolio_value='10000'),
                SimpleNamespace(id='paper-account', cash='10000', equity='10000', portfolio_value='10000'),
                SimpleNamespace(id='paper-account', cash='250', equity='10100', portfolio_value='10100')]

            def get_account():
                return account_reads.pop(0)

            client.get_account = get_account
            _, completed = complete_rebalance(client, pending_path)

            self.assertTrue(completed)
            saved = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['account_snapshot'], {
                'cash': 250.0,
                'equity': 10100.0,
                'portfolio_value': 10100.0
            })
            rebalances, orders = load_rebalance_history(Path(directory) / 'rebalance_history')
            self.assertEqual(len(rebalances), 1)
            self.assertTrue(orders.empty)
            self.assertEqual(rebalances.iloc[0]['positions'], {'A': 1.0})

    def test_loaded_pending_record_revalidates_frozen_invariants(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, _ = create_pending(
                directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            saved = json.loads(pending_path.read_text(encoding='utf-8'))
            saved['execution_window_end'] = pd.Timestamp('2024-01-03 10:00', tz='America/New_York').isoformat()
            pending_path.write_text(json.dumps(saved), encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'conflicts with its policy'):
                reconcile_pending_orders(FakeClient(), pending_path)

        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, _ = create_pending(
                directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            saved = json.loads(pending_path.read_text(encoding='utf-8'))
            saved['expected_positions'] = {}
            pending_path.write_text(json.dumps(saved), encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'do not match the proposed basket'):
                reconcile_pending_orders(FakeClient(), pending_path)

    def test_controller_retries_temporary_cash_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            filled_pending = deepcopy(pending)
            filled_pending['status'] = 'orders_filled'
            client = FakeClient()
            low_cash = deepcopy(client.account)
            low_cash.cash = '0'
            funded = deepcopy(client.account)
            funded.cash = '1000'
            client.get_account = unittest.mock.Mock(side_effect=[low_cash, funded])
            quote_client = SimpleNamespace(get_stock_latest_quote=lambda request: {
                'A': SimpleNamespace(ask_price=100.0, timestamp=pd.Timestamp('2024-01-03 09:34:30', tz='America/New_York'))})

            with patch('broker.get_data_client', return_value=quote_client), patch('broker.reconcile_pending_orders', side_effect=[pending, pending, filled_pending]), patch('broker.check_pending_orders', return_value=(False, [])), patch('broker.submit_pending_order', return_value=(pending, True)) as submit, patch('broker.complete_rebalance', return_value=(filled_pending, True)):
                result = advance_pending_rebalance(client, pending_path, env_path=Path(directory) / '.env', max_wait_seconds=1, poll_interval_seconds=0.001)

            self.assertEqual(result['status'], 'completed')
            self.assertEqual(client.get_account.call_count, 2)
            submit.assert_called_once()

    def test_controller_refreshes_clock_after_funding_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pending_path, pending = create_pending(directory,
                [{'ticker': 'A', 'side': 'buy', 'qty': 1.0, 'reference_price': 100.0}],
                ['A'])
            client = FakeClient()
            events = []
            clocks = iter([
                SimpleNamespace(timestamp=pd.Timestamp('2024-01-03 09:35:00', tz='America/New_York'), is_open=True),
                SimpleNamespace(timestamp=pd.Timestamp('2024-01-03 09:35:06', tz='America/New_York'), is_open=True),
            ])

            def get_clock():
                events.append('clock')
                return next(clocks)

            def get_quotes(request):
                events.append('quotes')
                return {'A': SimpleNamespace(ask_price=100.0, timestamp=pd.Timestamp('2024-01-03 09:35:05', tz='America/New_York'))}

            client.get_clock = get_clock
            quote_client = SimpleNamespace(get_stock_latest_quote=get_quotes)

            with patch('broker.get_data_client', return_value=quote_client), patch('broker.reconcile_pending_orders', return_value=pending), patch('broker.check_pending_orders', return_value=(False, [])), patch('broker.submit_pending_order', return_value=(pending, True)) as submit:
                result = advance_pending_rebalance(client, pending_path, env_path=Path(directory) / '.env', max_wait_seconds=0, poll_interval_seconds=0.001)

            self.assertEqual(result['status'], 'order_submitted')
            self.assertEqual(events, ['clock', 'quotes', 'clock'])
            submit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
