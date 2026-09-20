#testing that the ledger executes correctly
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get('CRYPTO_RESEARCH_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / 'src'))
from preparation import load_saved_daily_prices
from eligibility import add_research_eligibility
from universe_policy import apply_universe_policy
from features import add_momentum_features
from selection import build_weekly_universe, build_momentum_targets
from fx import load_fx_history, build_gbp_valuation_prices
from execution import build_execution_schedule
from portfolio import rebalance_gbp
from backtest import run_daily_ledger

SNAPSHOT = PROJECT / 'data/raw/coinbase_coverage_20260917T202028_505122Z'
FX_SNAPSHOT = PROJECT / 'data/raw/ecb_gbp_usd_20260918T133519_016371Z.json'
DAY = pd.Timedelta(days=1)


def reference_rates():
    dates = pd.to_datetime(['2023-12-29', '2024-01-02', '2024-01-03'], utc=True)
    return pd.DataFrame({'reference_date': dates, 'assumed_available_at': dates + DAY, 'usd_per_gbp': [1.25, 2., 4.]})


def dollar_prices():
    return pd.DataFrame({'product_id': 'A-USD', 'timestamp': pd.date_range('2024-01-01', periods=3, tz='UTC'),
                        'open': [100., 110., 120.], 'close': [105., 115., 125.]})


def ledger_fixture():
    dates = pd.date_range('2024-01-02', periods=14, tz='UTC')
    prices = pd.DataFrame({'product_id': 'A-USD', 'timestamp': dates, 'open_gbp': 100., 'close_gbp': 110.})
    targets = pd.DataFrame({'A-USD': [.5, 0.], 'FUTURE-USD': [0., 0.], 'CASH': [.5, 1.]}, index=dates[[0, 7]])
    return prices, targets


def analytic_rebalance_equity(values, weights, cash, rate):
    #enumerate buy/sell regions, solving each linear equation
    equity = cash + sum(values)
    for signs in itertools.product([-1., 1.], repeat=len(values)):
        answer = (equity + rate * sum(s*v for s, v in zip(signs, values))) / (1. + rate * sum(s*w for s, w in zip(signs, weights)))
        if all(s*(w*answer-v) >= -1e-8 for s, w, v in zip(signs, weights, values)):
            return answer
    raise AssertionError('No consistent analytic region')


def prepare_pipeline(prices, rates):
    feature = add_momentum_features(apply_universe_policy(add_research_eligibility(prices)))
    members, summary = build_weekly_universe(feature)
    _, momentum, benchmark = build_momentum_targets(members, summary)
    gbp, fx_audit = build_gbp_valuation_prices(prices, rates)
    start = summary.index[summary.breadth_ok].min() + DAY
    runs = {}
    for name, targets in [('momentum', momentum), ('benchmark', benchmark)]:
        planned, schedule = build_execution_schedule(targets, gbp)
        inside = pd.DatetimeIndex(schedule.loc[schedule.status.eq('within_history'), 'execution_at'])
        runs[name] = planned.loc[inside].loc[start:]
    return gbp, fx_audit, runs


class OfflineCase(unittest.TestCase):
    def setUp(self):
        blocker = patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden'))
        blocker.start()
        self.addCleanup(blocker.stop)


class FXTests(OfflineCase):
    def test_open_close_timing_carry_and_input_immutability(self):
        prices, rates = dollar_prices(), reference_rates()
        original_p, original_r = prices.copy(deep=True), rates.copy(deep=True)
        result, audit = build_gbp_valuation_prices(prices, rates)
        pd.testing.assert_frame_equal(prices, original_p)
        pd.testing.assert_frame_equal(rates, original_r)
        np.testing.assert_allclose(result.open_usd_per_gbp, [1.25, 1.25, 2.])
        np.testing.assert_allclose(result.close_usd_per_gbp, [1.25, 2., 4.])
        np.testing.assert_allclose(result.open_gbp, [80., 88., 60.])
        np.testing.assert_allclose(result.close_gbp, [84., 57.5, 31.25])
        self.assertEqual(audit.reference_age_days.tolist(), [3., 4., 1., 1.])
        self.assertTrue(audit.assumed_available_at.le(audit.valuation_at).all())

    def test_exact_age_limit_missing_history_and_stale_rate(self):
        rates = reference_rates().iloc[[0]]
        p = dollar_prices().iloc[[0]].assign(timestamp=pd.Timestamp('2024-01-04', tz='UTC'))
        _, audit = build_gbp_valuation_prices(p, rates, max_age_days=7)
        self.assertEqual(audit.reference_age_days.tolist(), [6., 7.])
        for bad in [p.assign(timestamp=pd.Timestamp('2024-01-05', tz='UTC')),
                    p.assign(timestamp=pd.Timestamp('2023-12-29', tz='UTC'))]:
            with self.assertRaises(ValueError):
                build_gbp_valuation_prices(bad, rates)

    def test_fx_rejects_duplicate_invalid_and_early_availability(self):
        r = reference_rates()
        for bad in [pd.concat([r, r.iloc[[0]]]), r.assign(usd_per_gbp=0.),
                    r.assign(usd_per_gbp=np.nan), r.assign(assumed_available_at=r.reference_date)]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                build_gbp_valuation_prices(dollar_prices(), bad)

    def test_fx_snapshot_rates_against_raw_and_metadata_failures(self):
        record = json.loads(FX_SNAPSHOT.read_text())
        result = load_fx_history(FX_SNAPSHOT)
        raw = json.loads(record['response_text'])
        expected = {pd.Timestamp(row['date'], tz='UTC'): float(row['rate']) for row in raw}
        self.assertEqual(len(result), len(expected))
        for row in result.itertuples(index=False):
            self.assertEqual(row.usd_per_gbp, expected[row.reference_date])
            self.assertEqual(row.assumed_available_at, row.reference_date + DAY)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.json'
            for field, value in [('source', 'wrong'), ('http_status', 500)]:
                path.write_text(json.dumps(dict(record, **{field: value})))
                with self.assertRaises(ValueError):
                    load_fx_history(path)
            payload = raw + [raw[0]]
            path.write_text(json.dumps(dict(record, response_text=json.dumps(payload))))
            with self.assertRaises(ValueError):
                load_fx_history(path)

    def test_future_fx_and_prices_leave_earlier_valuations_unchanged(self):
        prices, rates = dollar_prices(), reference_rates()
        baseline, _ = build_gbp_valuation_prices(prices, rates)
        changed = rates.copy()
        changed.loc[changed.reference_date.ge(pd.Timestamp('2024-01-03', tz='UTC')), 'usd_per_gbp'] *= 10
        result, _ = build_gbp_valuation_prices(prices, changed)
        pd.testing.assert_frame_equal(baseline.iloc[:2], result.iloc[:2])


class ExecutionTests(OfflineCase):
    def test_day_delay_end_boundary_and_unchanged_weights(self):
        p, t = ledger_fixture()
        t.index = t.index - DAY
        original = t.copy(deep=True)
        planned, audit = build_execution_schedule(t, p.iloc[:7])
        pd.testing.assert_frame_equal(t, original)
        self.assertEqual(audit.status.tolist(), ['within_history', 'after_history'])
        self.assertTrue((audit.execution_at - audit.decision_at).eq(DAY).all())
        np.testing.assert_array_equal(planned.to_numpy(), t.to_numpy())
        pd.testing.assert_index_equal(planned.index, pd.DatetimeIndex(t.index + DAY, name='execution_at'))

    def test_bad_decision_times_and_weights_rejected(self):
        p, t = ledger_fixture()
        with self.assertRaises(ValueError):
            build_execution_schedule(t, p)  # Tuesday decisions are invalid.
        t.index -= DAY
        for bad in [t.iloc[::-1], pd.concat([t, t.iloc[[0]]]), t.assign(CASH=2.)]:
            with self.assertRaises(ValueError):
                build_execution_schedule(bad, p)


class RebalanceTests(OfflineCase):
    def test_initial_purchase_exit_and_unheld_missing_price(self):
        prices = {'A-USD': 100., 'B-USD': 200., 'NEW-USD': np.nan}
        units, cash, trades, audit = rebalance_gbp({'NEW-USD': 0.}, 1000., prices,
                                                  {'A-USD': .5, 'B-USD': .5, 'NEW-USD': 0., 'CASH': 0.})
        expected = 1000 / 1.00125
        self.assertAlmostEqual(audit.equity_after_gbp, expected)
        self.assertAlmostEqual(audit.total_cost_gbp, 1000-expected)
        self.assertAlmostEqual(units['A-USD'], expected/200)
        exited, exit_cash, _, _ = rebalance_gbp(units, cash, prices, {'CASH': 1.})
        self.assertTrue(exited.eq(0).all())
        self.assertAlmostEqual(exit_cash, expected*.99875)

    def test_random_rebalances_against_analytic_piecewise_solution(self):
        rng = np.random.default_rng(101)
        assets = ['A-USD', 'B-USD', 'C-USD']
        for trial in range(30):
            with self.subTest(trial=trial):
                p = rng.uniform(10, 300, 3)
                units = rng.uniform(0, 12, 3)
                weights = rng.dirichlet(np.ones(4))
                cash = float(rng.uniform(0, 500))
                fee, other = (0., 0.) if trial == 0 else (9., 3.5)
                result, cash_after, trades, audit = rebalance_gbp(
                    dict(zip(assets, units)), cash, dict(zip(assets, p)),
                    dict(zip(assets+['CASH'], weights)), fee, other)
                equity = analytic_rebalance_equity(units*p, weights[:3], cash, (fee+other)/10000)
                np.testing.assert_allclose(result.to_numpy()*p, weights[:3]*equity, rtol=1e-12, atol=1e-9)
                self.assertAlmostEqual(cash_after, weights[3]*equity, places=8)
                self.assertAlmostEqual(audit.equity_after_gbp, equity, places=8)
                np.testing.assert_allclose(trades.fee_gbp, trades.signed_notional_gbp.abs()*fee/10000)

    def test_zero_turnover_all_cash_and_missing_held_or_targeted_price(self):
        result, cash, trades, audit = rebalance_gbp({'A-USD': 5.}, 500., {'A-USD': 100.}, {'A-USD': .5, 'CASH': .5})
        self.assertAlmostEqual(audit.traded_notional_gbp, 0., places=10)
        self.assertAlmostEqual(cash, 500.)
        _, cash, trades, _ = rebalance_gbp({}, 1000., {}, {'CASH': 1.})
        self.assertEqual(cash, 1000.)
        self.assertTrue(trades.empty)
        for held, target in [({'A-USD': 1.}, {'CASH': 1.}), ({}, {'A-USD': 1., 'CASH': 0.})]:
            with self.assertRaises(ValueError):
                rebalance_gbp(held, 1000., {'A-USD': np.nan}, target)

    def test_rebalance_rejects_borrowing_leverage_and_invalid_costs(self):
        cases = [({'A-USD': -1.}, 1000., {'CASH': 1.}, 9., 3.5),
                 ({}, -1., {'CASH': 1.}, 9., 3.5),
                 ({}, 1000., {'A-USD': 1., 'CASH': .5}, 9., 3.5),
                 ({}, 1000., {'CASH': 1.}, -1., 3.5),
                 ({}, 1000., {'CASH': 1.}, 10000., 0.)]
        for held, cash, target, fee, other in cases:
            with self.assertRaises(ValueError):
                rebalance_gbp(held, cash, {'A-USD': 100.}, target, fee, other)


class LedgerTests(OfflineCase):
    def test_hand_calculated_drift_cash_exit_and_no_terminal_sale(self):
        p, t = ledger_fixture()
        original_p, original_t = p.copy(deep=True), t.copy(deep=True)
        ledger, trades, positions = run_daily_ledger(p, t, fee_bps=0., other_cost_bps=0.)
        pd.testing.assert_frame_equal(p, original_p)
        pd.testing.assert_frame_equal(t, original_t)
        np.testing.assert_allclose(ledger.equity_close_gbp.iloc[:7], 1050.)
        np.testing.assert_allclose(ledger.equity_close_gbp.iloc[7:], 1000.)
        np.testing.assert_allclose(positions['A-USD'].iloc[:7], 5.)
        self.assertAlmostEqual(ledger.iloc[0].crypto_weight_close, 550/1050)
        self.assertAlmostEqual(ledger.iloc[0].net_return, .05)
        self.assertAlmostEqual(ledger.iloc[7].net_return, 1000/1050-1)
        self.assertEqual(len(trades), 2)
        self.assertTrue(ledger.loc[~ledger.scheduled_rebalance, 'traded_notional_gbp'].eq(0).all())
        no_exit = t.copy(); no_exit.loc[:, ['A-USD', 'CASH']] = [.5, .5]
        _, _, end_positions = run_daily_ledger(p, no_exit)
        self.assertGreater(end_positions.iloc[-1]['A-USD'], 0.)

    def test_entry_costs_compounding_and_trade_fee_totals(self):
        p, t = ledger_fixture()
        ledger, trades, _ = run_daily_ledger(p, t)
        equity_after = 1000/(1+.00125*.5)
        expected_close = equity_after*(.5 + .5*1.1)
        self.assertAlmostEqual(ledger.iloc[0].equity_close_gbp, expected_close)
        self.assertAlmostEqual(ledger.iloc[0].net_return, expected_close/1000-1)
        self.assertAlmostEqual((1+ledger.net_return).prod(), ledger.iloc[-1].equity_close_gbp/1000)
        self.assertAlmostEqual(ledger.fee_gbp.sum(), trades.fee_gbp.sum())
        self.assertAlmostEqual(ledger.other_cost_gbp.sum(), trades.other_cost_gbp.sum())

    def test_missing_price_for_held_or_targeted_asset_stops_run(self):
        p, t = ledger_fixture()
        for column, position in [('open_gbp', 0), ('close_gbp', 3), ('open_gbp', 7)]:
            bad = p.copy(); bad.loc[position, column] = np.nan
            with self.subTest(column=column, position=position), self.assertRaises(ValueError):
                run_daily_ledger(bad, t)
        with self.assertRaises(ValueError):
            run_daily_ledger(p.drop(index=3), t)
        cash_targets = t.copy(); cash_targets[['A-USD', 'FUTURE-USD']] = 0.; cash_targets['CASH'] = 1.
        ledger, trades, positions = run_daily_ledger(p.assign(open_gbp=np.nan, close_gbp=np.nan), cash_targets)
        self.assertTrue(ledger.equity_close_gbp.eq(1000.).all())
        self.assertTrue(trades.empty)

    def test_missing_schedule_and_duplicate_or_invalid_data_rejected(self):
        p, t = ledger_fixture()
        for bad_targets in [t.iloc[:1], t.iloc[::-1], t.set_axis(t.index - DAY)]:
            with self.assertRaises(ValueError):
                run_daily_ledger(p, bad_targets)
        with self.assertRaises(ValueError):
            run_daily_ledger(pd.concat([p, p.iloc[[0]]]), t)

    def test_future_price_change_does_not_change_prior_ledger(self):
        p, t = ledger_fixture()
        baseline = run_daily_ledger(p, t)
        changed = p.copy(); changed.loc[7:, ['open_gbp', 'close_gbp']] *= 10.
        result = run_daily_ledger(changed, t)
        pd.testing.assert_frame_equal(baseline[0].iloc[:7], result[0].iloc[:7])
        pd.testing.assert_frame_equal(baseline[2].iloc[:7], result[2].iloc[:7])


class FrozenAccountingTests(OfflineCase):
    @classmethod
    def setUpClass(cls):
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            cls.prices, _ = load_saved_daily_prices(SNAPSHOT)
            cls.rates = load_fx_history(FX_SNAPSHOT)
            cls.gbp, cls.fx_audit, cls.targets = prepare_pipeline(cls.prices, cls.rates)
            cls.runs = {name: run_daily_ledger(cls.gbp, target) for name, target in cls.targets.items()}

    def test_every_fx_match_and_conversion_against_raw_date_lookup(self):
        raw = json.loads(json.loads(FX_SNAPSHOT.read_text())['response_text'])
        observations = sorted((pd.Timestamp(r['date'], tz='UTC'), float(r['rate'])) for r in raw)
        expected = {}
        for row in self.fx_audit.itertuples(index=False):
            reference, rate = max((d, x) for d, x in observations if d + DAY <= row.valuation_at)
            self.assertEqual(row.reference_date, reference)
            self.assertEqual(row.usd_per_gbp, rate)
            self.assertLessEqual((row.valuation_at-reference)/DAY, 7)
            expected[row.valuation_at] = rate
        np.testing.assert_allclose(self.gbp.open_gbp, [r.open/expected[r.timestamp] for r in self.gbp.itertuples(index=False)])
        np.testing.assert_allclose(self.gbp.close_gbp, [r.close/expected[r.timestamp+DAY] for r in self.gbp.itertuples(index=False)])

    def test_reconstruct_both_ledgers_from_signed_trades(self):
        quotes = {(r.product_id, r.timestamp): (r.open_gbp, r.close_gbp) for r in self.gbp.itertuples(index=False)}
        for name, (ledger, trades, positions) in self.runs.items():
            with self.subTest(strategy=name):
                cash, previous = 1000., 1000.
                holdings = {asset: 0. for asset in positions.columns}
                by_date = {d: rows for d, rows in trades.groupby('execution_at')}
                expected_dates = pd.date_range('2021-12-21', '2026-08-31', tz='UTC', freq='D')
                pd.testing.assert_index_equal(ledger.index, expected_dates.rename('candle_start'))
                for row in ledger.itertuples():
                    date = row.Index
                    pre = cash + sum(u*quotes[(a,date)][0] for a,u in holdings.items() if u > 0)
                    self.assertAlmostEqual(pre, row.equity_open_gbp, places=7)
                    fees = other = notional = 0.
                    if date in by_date:
                        for trade in by_date[date].itertuples(index=False):
                            p = quotes[(trade.product_id, date)][0]
                            self.assertEqual(trade.reference_price_gbp, p)
                            self.assertEqual(trade.decision_at + DAY, date)
                            self.assertEqual(date.dayofweek, 1)
                            self.assertAlmostEqual(trade.units_before, holdings[trade.product_id], places=7)
                            delta = trade.signed_notional_gbp
                            self.assertAlmostEqual(trade.fee_gbp, abs(delta)*.0009, places=9)
                            self.assertAlmostEqual(trade.other_cost_gbp, abs(delta)*.00035, places=9)
                            # Infer new holdings from signed GBP trades, not reported units_after.
                            after = holdings[trade.product_id] + delta/p
                            self.assertAlmostEqual(after, trade.units_after, places=7)
                            holdings[trade.product_id] = 0. if trade.units_after == 0 else after
                            cash -= delta + trade.fee_gbp + trade.other_cost_gbp
                            fees += trade.fee_gbp; other += trade.other_cost_gbp; notional += abs(delta)
                    self.assertAlmostEqual(cash, row.cash_gbp, places=7)
                    self.assertAlmostEqual(fees, row.fee_gbp, places=7)
                    self.assertAlmostEqual(other, row.other_cost_gbp, places=7)
                    self.assertAlmostEqual(notional, row.traded_notional_gbp, places=7)
                    equity = cash + sum(u*quotes[(a,date)][1] for a,u in holdings.items() if u > 0)
                    self.assertAlmostEqual(equity, row.equity_close_gbp, places=7)
                    self.assertAlmostEqual(equity/previous-1, row.net_return, places=10)
                    np.testing.assert_allclose(positions.loc[date], [holdings[a] for a in positions.columns], atol=1e-7, rtol=1e-10)
                    if date in self.targets[name].index:
                        opening_equity = cash + sum(u*quotes[(a,date)][0] for a,u in holdings.items() if u > 0)
                        for asset, weight in self.targets[name].loc[date].items():
                            actual = cash if asset == 'CASH' else (holdings[asset]*quotes[(asset,date)][0] if holdings[asset] > 0 else 0.)
                            self.assertAlmostEqual(actual, weight*opening_equity, places=7)
                    previous = equity
                self.assertAlmostEqual((1+ledger.net_return).prod(), ledger.iloc[-1].equity_close_gbp/1000, places=9)

    def test_future_raw_prices_and_fx_leave_full_pipeline_prefix_unchanged(self):
        cutoff = pd.Timestamp('2025-01-01', tz='UTC')
        prices, rates = self.prices.copy(), self.rates.copy()
        prices.loc[prices.timestamp.ge(cutoff), ['open', 'high', 'low', 'close']] *= 1.7
        rates.loc[rates.reference_date.ge(cutoff), 'usd_per_gbp'] *= 1.2
        gbp, _, targets = prepare_pipeline(prices, rates)
        ledger, trades, positions = run_daily_ledger(gbp, targets['momentum'])
        before = self.runs['momentum']
        pd.testing.assert_frame_equal(before[0].loc[lambda x: x.index < cutoff], ledger.loc[lambda x: x.index < cutoff])
        pd.testing.assert_frame_equal(before[2].loc[lambda x: x.index < cutoff], positions.loc[lambda x: x.index < cutoff])
        pd.testing.assert_frame_equal(before[1].loc[lambda x: x.execution_at < cutoff], trades.loc[lambda x: x.execution_at < cutoff])


if __name__ == '__main__':
    unittest.main()
