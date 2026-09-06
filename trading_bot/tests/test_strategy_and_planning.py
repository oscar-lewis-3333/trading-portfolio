import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from broker import decisions_to_orders, get_positions, prepare_rebalance
from pipeline import generate_trading_decisions, select_momentum_basket

#ensure the system enacts the correct strategy (ranking 63d momentum and selecting top 20% (4) of tickers), before testing correct scheduling, duplicate handling amongst other things.

class StrategyAndPlanningTests(unittest.TestCase):
    def setUp(self):
        self.tickers = [f'T{i:02d}' for i in range(20)]

    def test_selector_uses_sixty_three_sessions_and_selects_top_four(self):
        dates = pd.bdate_range('2024-01-02', periods=64)
        prices = {ticker: np.linspace(100.0, 100.0 + ticker_i, len(dates)) for ticker_i, ticker in enumerate(self.tickers)}
        close_prices = pd.DataFrame(prices, index=dates)

        selected = select_momentum_basket(close_prices, self.tickers, dates[-1], lookback_days=63, top_frac=0.20)

        self.assertEqual(selected['ticker'].tolist(), ['T19', 'T18', 'T17', 'T16'])
        np.testing.assert_allclose(selected['target_weight'], 0.25)
        self.assertTrue(selected['signal_date'].eq(dates[-1]).all())

    def test_decision_schedule_holds_then_rebalances_and_rejects_a_miss(self):
        tickers = [f'T{i:02d}' for i in range(5)]
        sessions = pd.bdate_range('2024-01-02', periods=10)
        strategy = {
            'lookback_days': 3,
            'top_frac': 0.4,
            'rebalance_days': 2,
            'universe': sorted(tickers),
            'weighting': 'equal_at_rebalance',
        }
        saved_state = {
            'schema_version': 2,
            'strategy': strategy,
            'selection_date': sessions[3].strftime('%Y-%m-%d'),
            'tickers': ['T03', 'T04'],
        }

        module = types.ModuleType('data_loader')

        def fetch_price_data(ticker, period='1y'):
            ticker_i = tickers.index(ticker)
            values = np.linspace(100.0, 101.0 + ticker_i, len(sessions))
            return pd.DataFrame({'Close': values}, index=sessions)

        module.fetch_price_data = fetch_price_data

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / 'basket_state.json'
            state_path.write_text(json.dumps(saved_state), encoding='utf-8')

            with patch.dict(sys.modules, {'data_loader': module}):
                decisions, hold_plan = generate_trading_decisions(tickers, sessions[4], sessions, top_frac=0.4, rebalance_days=2, lookback_days=3, state_path=state_path)

                self.assertIsNone(decisions)
                self.assertFalse(hold_plan['rebalance_needed'])
                self.assertEqual(hold_plan['trading_days_since'], 1)

                decisions, due_plan = generate_trading_decisions(tickers, sessions[5], sessions, top_frac=0.4, rebalance_days=2, lookback_days=3, state_path=state_path)

                self.assertTrue(due_plan['rebalance_needed'])
                self.assertEqual(len(decisions), 2)
                self.assertEqual(json.loads(state_path.read_text(encoding='utf-8')), saved_state)

                with self.assertRaisesRegex(RuntimeError, 'Refusing to silently re-anchor'):
                    generate_trading_decisions(tickers, sessions[6], sessions, top_frac=0.4, rebalance_days=2, lookback_days=3, state_path=state_path)

    def test_planner_uses_bids_asks_reserve_and_sell_first(self):
        decisions = pd.DataFrame({
            'ticker': ['A', 'B'],
            'target_weight': [0.5, 0.5],
        })
        managed = ['A', 'B', 'C', 'D', 'E']
        bids = {'A': 99.0, 'B': 199.0, 'C': 100.0}
        asks = {'A': 101.0, 'B': 201.0, 'C': 102.0}

        orders = decisions_to_orders(decisions, current_positions={'C': 10.0}, available_cash=0.0, bid_prices=bids, ask_prices=asks, managed_tickers=managed, cash_reserve_fraction=0.01)

        self.assertEqual([(order['ticker'], order['side']) for order in orders], [('C', 'sell'), ('A', 'buy'), ('B', 'buy')])
        self.assertEqual([order['qty'] for order in orders], [10.0, 4.0, 2.0])
        self.assertEqual(orders[0]['reference_price'], bids['C'])
        self.assertEqual(orders[1]['reference_price'], asks['A'])

        sale_proceeds = sum(order['qty'] * bids[order['ticker']] for order in orders if order['side'] == 'sell')
        buy_cost = sum(order['qty'] * asks[order['ticker']] for order in orders if order['side'] == 'buy')
        self.assertLessEqual(buy_cost, sale_proceeds)

    def test_preparation_freezes_policy_prices_and_expected_positions(self):
        with tempfile.TemporaryDirectory() as directory:
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
                        'top_frac': 0.2,
                        'rebalance_days': 21,
                        'universe': ['A', 'B', 'C', 'D', 'E'],
                        'weighting': 'equal_at_rebalance',
                    },
                    'selection_date': '2024-01-02',
                    'tickers': ['A', 'B'],
                },
            }
            orders = [
                {'ticker': 'B', 'side': 'buy', 'qty': 2.0, 'reference_price': 201.0},
                {'ticker': 'C', 'side': 'sell', 'qty': 10.0, 'reference_price': 100.0},
                {'ticker': 'A', 'side': 'buy', 'qty': 4.0, 'reference_price': 101.0},
            ]
            execution_open = pd.Timestamp('2024-01-03 09:30', tz='America/New_York')
            execution_end = pd.Timestamp('2024-01-03 09:45', tz='America/New_York')

            pending = prepare_rebalance(plan, orders, account_id='paper-account', execution_date=pd.Timestamp('2024-01-03'), execution_open=execution_open, execution_window_end=execution_end, current_positions={'C': 10.0}, pending_path=pending_path)

            self.assertEqual(pending['schema_version'], 2)
            self.assertEqual([order['side'] for order in pending['orders']], ['sell', 'buy', 'buy'])
            self.assertEqual(pending['orders'][0]['reference_price'], 100.0)
            self.assertEqual(pending['expected_positions'], {'A': 4.0, 'B': 2.0})
            self.assertEqual(pending['execution_open'], execution_open.isoformat())
            self.assertEqual(pending['execution_window_end'], execution_end.isoformat())
            self.assertTrue(pending_path.exists())

    def test_position_reader_rejects_duplicate_or_non_long_holdings(self):
        duplicate_client = types.SimpleNamespace(get_all_positions=lambda: [
            types.SimpleNamespace(symbol='A', side='long', qty='1'),
            types.SimpleNamespace(symbol='A', side='long', qty='2'),
        ])

        with self.assertRaisesRegex(RuntimeError, 'Unexpected broker position'):
            get_positions(duplicate_client)

        short_client = types.SimpleNamespace(get_all_positions=lambda: [
            types.SimpleNamespace(symbol='A', side='short', qty='1'),
        ])

        with self.assertRaisesRegex(RuntimeError, 'Unexpected broker position'):
            get_positions(short_client)


if __name__ == '__main__':
    unittest.main()
