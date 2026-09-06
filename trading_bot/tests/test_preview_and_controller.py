import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from pipeline import build_rebalance_preview, run_bot

#test our preview function works as intended. start with creating a preview client, then a quote client, and ensure our preview acts as a preview (builds orders but does not submit/advance orders)

class PreviewClient:
    def __init__(self):
        self.clock = SimpleNamespace(timestamp=pd.Timestamp('2024-06-04 09:35', tz='America/New_York'), is_open=True)
        self.account = SimpleNamespace(id='paper-account', status='ACTIVE', account_blocked=False, trading_blocked=False, trade_suspended_by_user=False, cash='10000', equity='10000', portfolio_value='10000')

    def get_clock(self):
        return self.clock

    def get_calendar(self, filters=None):
        return [SimpleNamespace(date=pd.Timestamp('2024-06-03').date(), open=pd.Timestamp('2024-06-03 09:30').to_pydatetime(), close=pd.Timestamp('2024-06-03 16:00').to_pydatetime()),
            SimpleNamespace(date=pd.Timestamp('2024-06-04').date(), open=pd.Timestamp('2024-06-04 09:30').to_pydatetime(), close=pd.Timestamp('2024-06-04 16:00').to_pydatetime())]

    def get_account(self):
        return self.account

    def get_all_positions(self):
        return []

    def get_asset(self, ticker):
        return SimpleNamespace(tradable=True, fractionable=True)


class QuoteClient:
    def __init__(self, tickers):
        self.quotes = {
            ticker: SimpleNamespace(bid_price=99.0, ask_price=100.0, timestamp=pd.Timestamp('2024-06-04 09:34:30', tz='America/New_York'))for ticker in tickers}

    def get_stock_latest_quote(self, request):
        return self.quotes


class PreviewAndControllerTests(unittest.TestCase):
    def test_preview_builds_fundable_bid_ask_plan_without_writing_state(self):
        tickers = [f'T{i:02d}' for i in range(20)]
        selected = tickers[:4]
        decisions = pd.DataFrame({
            'ticker': selected,
            'target_weight': [0.25] * 4,
        })

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / 'basket_state.json'
            plan = {
                'signal_date': '2024-06-03',
                'rebalance_needed': True,
                'reason': 'initial_selection',
                'trading_days_since': None,
                'current_basket': [],
                'state_path': str(state_path),
                'proposed_state': {
                    'schema_version': 2,
                    'strategy': {
                        'lookback_days': 63,
                        'top_frac': 0.2,
                        'rebalance_days': 21,
                        'universe': sorted(tickers),
                        'weighting': 'equal_at_rebalance',
                    },
                    'selection_date': '2024-06-03',
                    'tickers': selected
                }
            }
            client = PreviewClient()

            with patch('pipeline.generate_trading_decisions', return_value=(decisions, plan)), patch('broker.get_data_client', return_value=QuoteClient(selected)):
                preview = build_rebalance_preview(client, tickers, env_path=Path(directory) / '.env', state_path=state_path, cash_reserve_fraction=0.10)

            self.assertEqual(preview['status'], 'ready_to_prepare')
            self.assertTrue(preview['execution_allowed'])
            self.assertEqual(preview['bid_prices'], {ticker: 99.0 for ticker in selected})
            self.assertEqual(preview['ask_prices'], {ticker: 100.0 for ticker in selected})
            self.assertEqual(len(preview['orders']), 4)
            self.assertTrue(all(order['qty'] == 22.0 for order in preview['orders']))
            self.assertEqual(preview['plan']['execution_policy']['cash_reserve_fraction'], 0.10)
            self.assertFalse(state_path.exists())
            self.assertFalse((Path(directory) / 'pending_rebalance.json').exists())

    def test_dry_run_does_not_prepare_or_advance_orders(self):
        tickers = [f'T{i:02d}' for i in range(20)]
        preview = {
            'status': 'ready_to_prepare',
            'execution_allowed': True,
            'plan': {'rebalance_needed': True},
            'orders': [{'ticker': 'T00', 'side': 'buy', 'qty': 1.0}],
        }
        client = SimpleNamespace()

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / 'basket_state.json'

            with patch('pipeline.bot_run_lock', return_value=nullcontext()), patch('broker.get_client', return_value=client), patch('broker.check_pending_orders', return_value=(False, [])), patch('pipeline.build_rebalance_preview', return_value=preview), patch('broker.prepare_rebalance') as prepare, patch('pipeline.advance_pending_rebalance') as advance:
                result = run_bot(tickers, submit_orders=False, env_path=Path(directory) / '.env', state_path=state_path)

            self.assertEqual(result['mode'], 'dry_run')
            prepare.assert_not_called()
            advance.assert_not_called()
            self.assertFalse((Path(directory) / 'pending_rebalance.json').exists())

    def test_hold_run_reports_broker_position_drift(self):
        tickers = [f'T{i:02d}' for i in range(20)]
        preview = {
            'status': 'holding_existing_basket',
            'plan': {'rebalance_needed': False},
            'blockers': [],
        }
        client = PreviewClient()
        client.get_all_positions = lambda: [SimpleNamespace(symbol='T00', side='long', qty='2')]

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / 'basket_state.json'
            state_path.write_text(json.dumps({
                'schema_version': 2,
                'account_id': 'paper-account',
                'positions_at_completion': {'T00': 1.0},
            }), encoding='utf-8')

            with patch('pipeline.bot_run_lock', return_value=nullcontext()), patch('broker.get_client', return_value=client), patch('broker.check_pending_orders', return_value=(False, [])), patch('pipeline.build_rebalance_preview', return_value=preview):
                result = run_bot(tickers, submit_orders=False, env_path=Path(directory) / '.env', state_path=state_path)

            self.assertEqual(result['status'], 'holdings_mismatch')
            self.assertEqual(result['current_positions'], {'T00': 2.0})
            self.assertIn('T00', result['blockers'][0])

    def test_execution_run_prepares_then_advances_the_frozen_plan(self):
        tickers = [f'T{i:02d}' for i in range(20)]
        client = SimpleNamespace()
        preview = {
            'status': 'ready_to_prepare',
            'execution_allowed': True,
            'plan': {'rebalance_needed': True},
            'orders': [{'ticker': 'T00', 'side': 'buy', 'qty': 1.0}],
            'account_id': 'paper-account',
            'execution_date': pd.Timestamp('2024-06-04'),
            'execution_open': pd.Timestamp('2024-06-04 09:30', tz='America/New_York'),
            'execution_window_end': pd.Timestamp('2024-06-04 09:45', tz='America/New_York'),
            'current_positions': {},
        }
        completed = {'mode': 'paper_execution', 'status': 'completed'}

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / 'basket_state.json'
            pending_path = state_path.resolve().parent / 'pending_rebalance.json'

            with patch('pipeline.bot_run_lock', return_value=nullcontext()), patch('broker.get_client', return_value=client), patch('broker.check_pending_orders', return_value=(False, [])), patch('pipeline.build_rebalance_preview', return_value=preview), patch('broker.prepare_rebalance') as prepare, patch('pipeline.advance_pending_rebalance', return_value=completed) as advance:
                result = run_bot(tickers, submit_orders=True, env_path=Path(directory) / '.env', state_path=state_path)

            self.assertEqual(result, completed)
            prepare.assert_called_once_with(plan=preview['plan'], orders=preview['orders'], account_id='paper-account', execution_date=preview['execution_date'], execution_open=preview['execution_open'], execution_window_end=preview['execution_window_end'], current_positions={}, pending_path=pending_path)
            advance.assert_called_once()


if __name__ == '__main__':
    unittest.main()
