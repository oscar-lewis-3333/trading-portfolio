import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from monitoring import format_run_alert, send_alert, send_run_alert

#file dedicated to testing the monitoring (discord notifications), including that the alerts are actually sent even if bot fails, the right reason is sent, it is sent to the right place and that it is not a quiet send.

class MonitoringTests(unittest.TestCase):
    def test_formats_success_pending_blocked_and_quiet_results(self):
        success = format_run_alert({
            'mode': 'paper_execution',
            'status': 'order_submitted',
            'pending': {
                'rebalance_id': 'rebalance-1',
                'orders': [{'status': 'new'}]
            }
        })
        pending = format_run_alert({
            'mode': 'paper_execution',
            'status': 'orders_in_progress',
            'pending': {
                'orders': [{'status': 'filled'}, {'status': 'new'}],
            }
        })
        blocked = format_run_alert({
            'mode': 'paper_execution',
            'status': 'execution_blocked',
            'blockers': ['account_blocked']
        })

        self.assertEqual(success[0], 'Trading Bot Order Submitted')
        self.assertIn('Orders: new: 1', success[1])
        self.assertEqual(pending[0], 'Trading Bot Orders Pending')
        self.assertIn('filled: 1, new: 1', pending[1])
        self.assertEqual(blocked[0], 'Trading Bot Blocked')
        self.assertIn('account_blocked', blocked[1])
        self.assertIsNone(format_run_alert({
            'mode': 'paper_execution',
            'status': 'holding_existing_basket'
        }))

    def test_send_run_alert_dispatches_the_formatted_result(self):
        result = {
            'mode': 'paper_execution',
            'status': 'order_submitted'
        }

        with patch('monitoring.send_alert', return_value=True) as mocked_send:
            sent = send_run_alert(result, env_path='/tmp/example.env')

        self.assertTrue(sent)
        mocked_send.assert_called_once_with(subject='Trading Bot Order Submitted', body='Status: order_submitted\nMode: paper_execution', env_path='/tmp/example.env')

    def test_completed_alert_includes_portfolio_value_and_change(self):
        completed = format_run_alert({
            'mode': 'paper_execution',
            'status': 'completed',
            'pending': {
                'rebalance_id': 'rebalance-2',
                'orders': [{'status': 'filled'}],
                'completed_state': {
                    'account_snapshot': {'portfolio_value': 105000.0},
                },
                'previous_state': {
                    'account_snapshot': {'portfolio_value': 100000.0},
                },
            },
        })
        first_baseline = format_run_alert({
            'mode': 'paper_execution',
            'status': 'completed',
            'pending': {
                'completed_state': {
                    'account_snapshot': {'portfolio_value': 101651.33},
                },
                'previous_state': None,
            },
        })

        self.assertIn('Portfolio value: $105,000.00', completed[1])
        self.assertIn('Change since previous completed rebalance: +5.00%', completed[1])
        self.assertIn('Portfolio value: $101,651.33', first_baseline[1])
        self.assertIn('Change since previous completed rebalance: N/A (first recorded baseline)', first_baseline[1])

    def test_send_run_alert_does_not_dispatch_a_quiet_hold(self):
        result = {
            'mode': 'paper_execution',
            'status': 'holding_existing_basket'
        }

        with patch('monitoring.send_alert') as mocked_send:
            sent = send_run_alert(result)

        self.assertFalse(sent)
        mocked_send.assert_not_called()

    def test_send_alert_posts_to_the_configured_discord_webhook(self):
        response = Mock()
        webhook = 'https://discord.example/webhook'

        with patch.dict(os.environ, {'DISCORD_WEBHOOK': webhook}, clear=True):
            with patch('monitoring.requests.post', return_value=response) as mocked_post:
                sent = send_alert('Test Subject', 'Test Body')

        self.assertTrue(sent)
        mocked_post.assert_called_once_with(webhook, json={'content': '**Test Subject**\nTest Body'}, timeout=10)
        response.raise_for_status.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
