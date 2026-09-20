#testing for research scope, momentum timing and target construction
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get('CRYPTO_RESEARCH_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / 'src'))
from preparation import load_saved_daily_prices
from eligibility import add_research_eligibility
from universe_policy import apply_universe_policy, meme_exclusion_table
from features import add_momentum_features
from selection import build_weekly_universe, build_momentum_targets

DAY = pd.Timedelta(days=1)
SNAPSHOT = PROJECT / 'data/raw/coinbase_coverage_20260917T202028_505122Z'


def weekly_fixture():
    rows = []
    for date, names in [('2024-01-01', ['D', 'C', 'B', 'A']),
                        ('2024-01-15', ['A', 'B']),
                        ('2024-01-22', ['C', 'B', 'D'])]:
        for name in names:
            rows.append(dict(product_id=name + '-USD', signal_available_at=pd.Timestamp(date, tz='UTC'),
                             universe_candidate=True, median_dollar_volume=2e6,
                             momentum_return={'A': -.1, 'B': -.1, 'C': -.2, 'D': -.3}[name]))
    return pd.DataFrame(rows)


def pipeline(prices):
    daily = add_momentum_features(apply_universe_policy(add_research_eligibility(prices)))
    members, summary = build_weekly_universe(daily)
    ranked, targets, benchmark = build_momentum_targets(members, summary)
    return daily, members, summary, ranked, targets, benchmark


class OfflineCase(unittest.TestCase):
    def setUp(self):
        blocker = patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden'))
        blocker.start()
        self.addCleanup(blocker.stop)


class PolicyFeatureTests(OfflineCase):
    def test_policy_preserves_rows_input_and_independent_eligibility(self):
        ids = ['DOGE-USD', 'BTC-USD', 'ETH-USD']
        panel = pd.DataFrame({'product_id': ids, 'research_candidate': [True, True, False]})
        original = panel.copy(deep=True)
        result = apply_universe_policy(panel)
        pd.testing.assert_frame_equal(panel, original)
        self.assertEqual(result.product_id.tolist(), ids)
        self.assertEqual(result.research_candidate.tolist(), [True, True, False])
        self.assertEqual(result.universe_candidate.tolist(), [False, True, False])
        self.assertEqual(result.scope_exclusion_reason.iloc[1], '')

    def test_documented_exclusions_and_invalid_flags(self):
        table = meme_exclusion_table()
        self.assertEqual(set(table.index), {x + '-USD' for x in
                         ['DOGE', 'SHIB', 'BONK', 'FLOKI', 'PEPE', 'WIF', 'SPX', 'TRUMP', 'PENGU']})
        self.assertTrue(table.source_url.str.startswith('https://').all())
        self.assertTrue(table.exclusion_reason.str.len().gt(0).all())
        self.assertTrue(table.reviewed_at.eq('2026-09-18').all())
        for bad in [[1], [None]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                apply_universe_policy(pd.DataFrame({'product_id': ['BTC-USD'], 'research_candidate': bad}))

    def test_gap_reset_daily_timing_and_product_independence(self):
        rows = []
        for name, dates, closes in [('A-USD', [1, 2, 3, 6, 7, 8], [10, 12, 15, 30, 24, 18]),
                                    ('B-USD', [1, 2, 3], [100, 90, 80])]:
            for day, close in zip(dates, closes):
                rows.append(dict(product_id=name, timestamp=pd.Timestamp(f'2024-01-{day:02d}', tz='UTC'),
                                 close=close, universe_candidate=True, segment_id=999))
        source = pd.DataFrame(rows).sample(frac=1, random_state=7)
        original = source.copy(deep=True)
        result = add_momentum_features(source, 2).set_index(['product_id', 'timestamp'])
        pd.testing.assert_frame_equal(source, original)
        expected = {('A-USD', 3): .5, ('A-USD', 8): -.4, ('B-USD', 3): -.2}
        for (product, day), value in expected.items():
            row = result.loc[(product, pd.Timestamp(f'2024-01-{day:02d}', tz='UTC'))]
            self.assertAlmostEqual(row.momentum_return, value)
            self.assertEqual(row.signal_available_at, row.name[1] + DAY)
        self.assertEqual(int(result.momentum_ready.sum()), 3)
        self.assertTrue(result.universe_candidate.all())

    def test_thirty_days_not_thirty_weeks(self):
        source = pd.DataFrame({'product_id': 'A-USD', 'timestamp': pd.date_range('2024-01-01', periods=50, tz='UTC'),
                               'close': np.arange(1, 51, dtype=float) ** 2})
        result = add_momentum_features(source, 30)
        self.assertEqual(int(result.momentum_ready.sum()), 20)
        self.assertAlmostEqual(result.iloc[30].momentum_return, 31 ** 2 - 1)
        monday = result.loc[result.signal_available_at.eq(pd.Timestamp('2024-02-05', tz='UTC'))].iloc[0]
        self.assertAlmostEqual(monday.momentum_return, 35 ** 2 / 5 ** 2 - 1)

    def test_features_reject_bad_prices_duplicates_and_lookbacks(self):
        source = pd.DataFrame({'product_id': ['A-USD'], 'timestamp': [pd.Timestamp('2024-01-01', tz='UTC')], 'close': [1.]})
        for close in [0., -1., np.nan, np.inf]:
            with self.subTest(close=close), self.assertRaises(ValueError):
                add_momentum_features(source.assign(close=close))
        for lookback in [0, -1, True, 1.5]:
            with self.subTest(lookback=lookback), self.assertRaises(ValueError):
                add_momentum_features(source, lookback)
        with self.assertRaises(ValueError):
            add_momentum_features(pd.concat([source, source]))


class SelectionTests(OfflineCase):
    def test_liquidity_cap_ties_schedule_and_input_immutability(self):
        source = weekly_fixture()
        original = source.copy(deep=True)
        members, summary = build_weekly_universe(source, max_assets=3, min_assets=3)
        pd.testing.assert_frame_equal(source, original)
        self.assertEqual(summary.eligible_assets.tolist(), [4, 0, 2, 3])
        self.assertEqual(summary.universe_size.tolist(), [3, 0, 0, 3])
        self.assertEqual(members.iloc[:3].product_id.tolist(), ['A-USD', 'B-USD', 'C-USD'])
        self.assertEqual(members.iloc[:3].liquidity_rank.tolist(), [1, 2, 3])

    def test_only_exact_monday_eligible_rows_used(self):
        source = weekly_fixture()
        extra = source.iloc[[0]].assign(signal_available_at=pd.Timestamp('2024-01-02', tz='UTC'), median_dollar_volume=1e12)
        excluded = source.iloc[[0]].assign(product_id='EXCLUDED-USD', universe_candidate=False, median_dollar_volume=1e12)
        a = build_weekly_universe(source, 3, 3)
        b = build_weekly_universe(pd.concat([source, extra, excluded]), 3, 3)
        for left, right in zip(a, b):
            pd.testing.assert_frame_equal(left, right)

    def test_negative_winners_cash_weeks_and_departures(self):
        members, summary = build_weekly_universe(weekly_fixture(), 3, 3)
        original = members.copy(deep=True)
        ranked, targets, benchmark = build_momentum_targets(members, summary, .5)
        pd.testing.assert_frame_equal(members, original)
        self.assertEqual(ranked.loc[ranked.selected, 'product_id'].tolist(), ['A-USD', 'B-USD', 'B-USD', 'C-USD'])
        np.testing.assert_allclose(targets.iloc[0][['A-USD', 'B-USD', 'C-USD', 'D-USD', 'CASH']], [.5, .5, 0, 0, 0])
        for matrix in [targets, benchmark]:
            np.testing.assert_allclose(matrix.sum(axis=1), 1)
            np.testing.assert_allclose(matrix.iloc[1:3].CASH, 1)
            self.assertTrue(matrix.iloc[1:3].drop(columns='CASH').eq(0).all().all())
            self.assertEqual(matrix.iloc[-1]['A-USD'], 0)
        np.testing.assert_allclose(benchmark.iloc[0][['A-USD', 'B-USD', 'C-USD']], 1/3)

    def test_rounding_at_realistic_universe_sizes(self):
        for n, k in [(20, 4), (21, 5), (27, 6), (30, 6)]:
            with self.subTest(n=n):
                source = pd.DataFrame({'product_id': [f'A{i:02d}-USD' for i in range(n)],
                                       'signal_available_at': pd.Timestamp('2024-01-01', tz='UTC'),
                                       'universe_candidate': True, 'median_dollar_volume': 2e6,
                                       'momentum_return': np.arange(n, dtype=float)})
                members, summary = build_weekly_universe(source)
                ranked, targets, benchmark = build_momentum_targets(members, summary)
                self.assertEqual(int(ranked.selected.sum()), k)
                np.testing.assert_allclose(ranked.loc[ranked.selected, 'momentum_weight'], 1/k)
                np.testing.assert_allclose(benchmark.drop(columns='CASH'), 1/n)
                self.assertEqual(targets.iloc[0].CASH, 0)

    def test_all_cash_when_no_week_qualifies(self):
        members, summary = build_weekly_universe(weekly_fixture())
        ranked, targets, benchmark = build_momentum_targets(members, summary)
        self.assertTrue(ranked.empty)
        self.assertEqual(targets.columns.tolist(), ['CASH'])
        self.assertEqual(targets.CASH.tolist(), [1., 1., 1., 1.])
        pd.testing.assert_frame_equal(targets, benchmark)

    def test_shuffling_and_full_fraction(self):
        members, summary = build_weekly_universe(weekly_fixture(), 3, 3)
        a = build_momentum_targets(members, summary)
        b = build_momentum_targets(members.sample(frac=1, random_state=2), summary)
        for left, right in zip(a, b):
            pd.testing.assert_frame_equal(left, right)
        _, targets, benchmark = build_momentum_targets(members, summary, 1.)
        pd.testing.assert_frame_equal(targets, benchmark)

    def test_invalid_membership_or_signal_fails_instead_of_dropping(self):
        members, summary = build_weekly_universe(weekly_fixture(), 3, 3)
        for bad in [members.iloc[1:], pd.concat([members, members.iloc[[0]]]),
                    members.assign(momentum_return=np.nan),
                    members.assign(decision_at=pd.Timestamp('2025-01-06', tz='UTC'))]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                build_momentum_targets(bad, summary)
        for fraction in [0, -1, True, 1.1, np.nan]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                build_momentum_targets(members, summary, fraction)

    def test_invalid_universe_inputs(self):
        source = weekly_fixture()
        for bad in [pd.concat([source, source.iloc[[0]]]),
                    source.assign(median_dollar_volume=np.inf),
                    source.assign(signal_available_at=source.signal_available_at + pd.Timedelta(hours=1))]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                build_weekly_universe(bad, 3, 3)
        with self.assertRaises(ValueError):
            build_weekly_universe(source, max_assets=2, min_assets=3)


class FrozenPipelineTests(OfflineCase):
    @classmethod
    def setUpClass(cls):
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            cls.prices, _ = load_saved_daily_prices(SNAPSHOT)
            cls.outputs = pipeline(cls.prices)

    def test_latest_report_and_schedule(self):
        daily, members, summary, ranked, targets, benchmark = self.outputs
        self.assertEqual(summary.index[summary.breadth_ok].min(), pd.Timestamp('2021-12-20', tz='UTC'))
        self.assertEqual(int((~summary.loc['2021-12-20':].breadth_ok).sum()), 19)
        self.assertEqual(int(summary.iloc[-1].universe_size), 27)
        selected = ranked.loc[ranked.decision_at.eq(summary.index[-1]) & ranked.selected, 'product_id']
        self.assertEqual(selected.tolist(), ['ENA-USD', 'HYPE-USD', 'VVV-USD', 'CRV-USD', 'SOL-USD', 'LINK-USD'])
        self.assertFalse(members.product_id.isin(meme_exclusion_table().index).any())
        np.testing.assert_allclose(targets.iloc[-1][selected], 1/6)
        np.testing.assert_allclose(benchmark.iloc[-1].loc[lambda x: x > 0], 1/27)

    def test_all_weekly_targets_against_calendar_and_python_sort_oracle(self):
        daily, members, summary, ranked, targets, benchmark = self.outputs
        closes = {(r.product_id, r.timestamp): r.close for r in self.prices.itertuples(index=False)}
        weekly = {date: list(rows.itertuples(index=False)) for date, rows in daily.groupby('signal_available_at')}
        for date in summary.index:
            candidates = [r for r in weekly.get(date, []) if r.universe_candidate]
            candidates.sort(key=lambda r: (-r.median_dollar_volume, r.product_id))
            expected = candidates[:30] if len(candidates) >= 20 else []
            self.assertEqual(int(summary.loc[date, 'eligible_assets']), len(candidates))
            momentum = {}
            for row in expected:
                end = date - DAY
                self.assertTrue(all((row.product_id, end - i * DAY) in closes for i in range(31)))
                momentum[row.product_id] = closes[(row.product_id, end)] / closes[(row.product_id, end - 30 * DAY)] - 1
            order = sorted(momentum, key=lambda product: (-momentum[product], product))
            k = math.ceil(len(order) / 5)
            wanted = {product: 1/k for product in order[:k]}
            wanted['CASH'] = float(not order)
            baseline = {product: 1/len(order) for product in order}
            baseline['CASH'] = float(not order)
            for matrix, oracle in [(targets, wanted), (benchmark, baseline)]:
                np.testing.assert_allclose(matrix.loc[date], [oracle.get(c, 0.) for c in matrix.columns], atol=1e-14, rtol=0)

    def test_future_prices_volumes_and_new_products_do_not_change_past_targets(self):
        cutoff = pd.Timestamp('2024-07-01', tz='UTC')
        changed = self.prices.copy(deep=True)
        future = changed.timestamp.ge(cutoff)
        changed.loc[future, ['open', 'high', 'low', 'close']] *= 7
        changed.loc[future, 'volume'] *= 100
        new = changed.loc[future & changed.product_id.eq('BTC-USD')].assign(product_id='FUTURE-USD')
        mutated = pipeline(pd.concat([changed, new], ignore_index=True))
        truncated = pipeline(self.prices.loc[self.prices.timestamp.lt(cutoff)])
        for alternative in [mutated, truncated]:
            pd.testing.assert_frame_equal(self.outputs[2].loc[:cutoff], alternative[2].loc[:cutoff])
            for pos in [4, 5]:
                left = self.outputs[pos].loc[:cutoff]
                right = alternative[pos].loc[:cutoff]
                # A future-only asset may add a zero column to the matrix schema.
                columns = sorted(set(left.columns) | set(right.columns))
                pd.testing.assert_frame_equal(left.reindex(columns=columns, fill_value=0.),
                                              right.reindex(columns=columns, fill_value=0.))


if __name__ == '__main__':
    unittest.main()
