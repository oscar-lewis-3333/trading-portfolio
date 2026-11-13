#check archive replay, calendar is continuous and eligibility of universe
import json
import os
from pathlib import Path
from statistics import median
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT = Path(os.environ.get('CRYPTO_RESEARCH_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT / 'src'))
import preparation
import eligibility

SNAPSHOT = PROJECT / 'data/raw/coinbase_coverage_20260917T202028_505122Z'
DAY = pd.Timedelta(days=1)
NUMERIC = ['low', 'high', 'open', 'close', 'volume']


def sample_panel():
    rows = []
    for product in ['AAA-USD', 'BBB-USD']:
        for day in range(430):
            if product == 'BBB-USD' and 200 <= day < 215:
                continue
            close = 10.0 + (day % 11)
            volume = 110_000.0 if day % 7 else 40_000.0
            rows.append(dict(product_id=product, timestamp=pd.Timestamp('2020-01-01', tz='UTC') + day * DAY,
                             low=close - 1, high=close + 1, open=close, close=close, volume=volume))
    return pd.DataFrame(rows)


def calendar_oracle(prices, product, timestamp, history_days=180, liquidity_days=30, threshold=1e6):
    records = {(r.product_id, r.timestamp.value): (float(r.close), float(r.volume))
               for r in prices.itertuples(index=False)}
    history_ready = all((product, timestamp.value - n * DAY.value) in records
                        for n in range(history_days + 1))
    keys = [(product, timestamp.value - n * DAY.value) for n in range(liquidity_days)]
    liquidity = median([records[k][0] * records[k][1] for k in keys]) if all(k in records for k in keys) else np.nan
    return history_ready, liquidity, history_ready and liquidity >= threshold


class OfflineCase(unittest.TestCase):
    def setUp(self):
        blocker = patch('requests.sessions.Session.request', side_effect=AssertionError('Network access is forbidden in this suite'))
        blocker.start()
        self.addCleanup(blocker.stop)


class LoaderTests(OfflineCase):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        spec = dict(products=['AAA-USD', 'BBB-USD'], start='2020-01-01', end_exclusive='2020-01-05',
                    windows={'a': ['2020-01-01', '2020-01-03'], 'b': ['2020-01-03', '2020-01-05']})
        (self.root / 'download_spec.json').write_text(json.dumps(spec))
        reports = []
        for product in spec['products']:
            for label, (a, b) in spec['windows'].items():
                start, end = pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC')
                # Include an extra candle on BOTH sides of each request.
                payload = [[int(t.timestamp()), 9, 11, 10, 10, 100_000]
                           for t in pd.date_range(start - DAY, end, freq='D')]
                name = f'{product}_{label}.json'
                raw = dict(source='coinbase_exchange', product_id=product, http_status=200,
                           request_url=f'https://api.exchange.coinbase.com/products/{product}/candles?granularity=86400&start={a}T00:00:00Z&end={b}T00:00:00Z',
                           response_text=json.dumps(payload))
                (self.root / name).write_text(json.dumps(raw))
                reports.append(dict(product_id=product, sample=label, outcome='ok', rows_in_window=2,
                                    raw_path=f'/old/computer/location/{name}'))
        pd.DataFrame(reports).to_csv(self.root / 'coverage_report.csv', index=False)

    def edit_raw(self, change):
        path = self.root / 'AAA-USD_a.json'
        raw = json.loads(path.read_text())
        change(raw)
        path.write_text(json.dumps(raw))

    def test_half_open_boundaries_relocation_and_no_mutation(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        prices, _ = preparation.load_saved_daily_prices(self.root)
        self.assertEqual(len(prices), 8)
        self.assertFalse(prices.duplicated(['product_id', 'timestamp']).any())
        self.assertEqual(prices.timestamp.min(), pd.Timestamp('2020-01-01', tz='UTC'))
        self.assertEqual(prices.timestamp.max(), pd.Timestamp('2020-01-04', tz='UTC'))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_missing_or_duplicate_request_rejected(self):
        path = self.root / 'coverage_report.csv'
        original = pd.read_csv(path)
        for frame in [original.iloc[:-1], pd.concat([original, original.iloc[:1]], ignore_index=True)]:
            with self.subTest(rows=len(frame)):
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError): preparation.load_saved_daily_prices(self.root)

    def test_unsuccessful_request_and_count_disagreement_rejected(self):
        path = self.root / 'coverage_report.csv'
        original = pd.read_csv(path)
        for column, value in [('outcome', 'http_error'), ('rows_in_window', 3)]:
            frame = original.copy(); frame.loc[0, column] = value; frame.to_csv(path, index=False)
            with self.subTest(column=column), self.assertRaises(ValueError): preparation.load_saved_daily_prices(self.root)

    def test_wrong_source_product_and_granularity_rejected(self):
        path = self.root / 'AAA-USD_a.json'
        original = json.loads(path.read_text())
        variants = [dict(source='other'), dict(product_id='CCC-USD'), dict(http_status=404),
                    dict(request_url=original['request_url'].replace('86400', '3600')),
                    dict(request_url=original['request_url'].replace('2020-01-01', '2020-01-02'))]
        for changes in variants:
            path.write_text(json.dumps({**original, **changes}))
            with self.subTest(changes=changes), self.assertRaises(ValueError): preparation.load_saved_daily_prices(self.root)

    def test_duplicate_candles_remain_visible(self):
        def change(raw):
            payload = json.loads(raw['response_text']); payload[2][0] = payload[1][0]
            raw['response_text'] = json.dumps(payload)
        self.edit_raw(change)
        prices, _ = preparation.load_saved_daily_prices(self.root)
        audit, _ = preparation.audit_daily_prices(prices)
        self.assertEqual(audit.loc['AAA-USD', 'duplicates'], 1)
        self.assertEqual(audit.loc['AAA-USD', 'internal_missing_days'], 1)

    def test_malformed_values_retained_for_audit(self):
        def change(raw):
            payload = json.loads(raw['response_text']); payload[1][4] = 'invalid'
            raw['response_text'] = json.dumps(payload)
        self.edit_raw(change)
        prices, _ = preparation.load_saved_daily_prices(self.root)
        audit, _ = preparation.audit_daily_prices(prices)
        self.assertEqual(audit.loc['AAA-USD', 'nonfinite_rows'], 1)


class PreparationTests(OfflineCase):
    def test_internal_gap_is_distinct_from_late_start(self):
        prices = sample_panel()
        audit, missing = preparation.audit_daily_prices(prices)
        self.assertEqual(audit.loc['AAA-USD', 'internal_missing_days'], 0)
        self.assertEqual(audit.loc['BBB-USD', 'internal_missing_days'], 15)
        self.assertEqual(len(missing), 15)
        late = prices.loc[prices.timestamp.ge(pd.Timestamp('2020-03-01', tz='UTC'))]
        other, _ = preparation.audit_daily_prices(late)
        self.assertEqual(other.loc['AAA-USD', 'internal_missing_days'], 0)

    def test_each_numeric_defect_is_detected(self):
        cases = [('close', np.nan, 'nonfinite_rows'), ('close', 0, 'nonpositive_price_rows'),
                 ('high', 1, 'invalid_ohlc_rows'), ('volume', -1, 'negative_volume_rows'),
                 ('volume', 0, 'zero_volume_rows')]
        for column, value, count in cases:
            with self.subTest(column=column, value=value):
                prices = sample_panel(); prices.loc[0, column] = value
                audit, _ = preparation.audit_daily_prices(prices)
                self.assertEqual(audit.loc['AAA-USD', count], 1)

    def test_segments_reset_and_signal_is_available_next_day(self):
        original = sample_panel(); untouched = original.copy(deep=True)
        result = preparation.add_history_segments(original.sample(frac=1, random_state=4))
        pd.testing.assert_frame_equal(original, untouched)
        b = result.loc[result.product_id.eq('BBB-USD')].set_index('timestamp')
        restart = pd.Timestamp('2020-01-01', tz='UTC') + 215 * DAY
        self.assertEqual(b.loc[restart, 'segment_id'], 2)
        self.assertEqual(b.loc[restart, 'contiguous_observations'], 1)
        pd.testing.assert_series_equal(result.signal_available_at, result.timestamp + DAY, check_names=False)
        sizes = result.groupby(['product_id', 'segment_id']).size().tolist()
        self.assertEqual(sizes, [430, 200, 215])

    def test_duplicate_naive_and_off_midnight_timestamps_rejected(self):
        prices = sample_panel()
        naive = prices.copy(); naive['timestamp'] = naive.timestamp.dt.tz_localize(None)
        shifted = prices.copy(); shifted.loc[0, 'timestamp'] += pd.Timedelta(hours=1)
        for bad in [pd.concat([prices, prices.iloc[:1]], ignore_index=True), naive, shifted]:
            with self.subTest(), self.assertRaises(ValueError): preparation.add_history_segments(bad)


class EligibilityTests(OfflineCase):
    def test_independent_calendar_and_median_oracle(self):
        prices = sample_panel(); result = eligibility.add_research_eligibility(prices).set_index(['product_id', 'timestamp'])
        for product in ['AAA-USD', 'BBB-USD']:
            for day in [0, 28, 29, 179, 180, 199, 215, 243, 244, 394, 395, 429]:
                time = pd.Timestamp('2020-01-01', tz='UTC') + day * DAY
                expected = calendar_oracle(prices, product, time)
                row = result.loc[(product, time)]
                self.assertEqual(bool(row.history_ready), expected[0])
                np.testing.assert_allclose(row.median_dollar_volume, expected[1], equal_nan=True)
                self.assertEqual(bool(row.research_candidate), expected[2])

    def test_exact_threshold_and_history_boundary(self):
        p = sample_panel(); p['close'] = 10.; p['volume'] = 100_000.
        result = eligibility.add_research_eligibility(p)
        a = result.loc[result.product_id.eq('AAA-USD')].reset_index(drop=True)
        self.assertFalse(a.loc[179, 'history_ready']); self.assertTrue(a.loc[180, 'history_ready'])
        self.assertTrue(a.loc[180, 'research_candidate'])
        p['volume'] = 99_999.
        self.assertFalse(eligibility.add_research_eligibility(p).research_candidate.any())

    def test_gap_resets_both_windows(self):
        result = eligibility.add_research_eligibility(sample_panel())
        b = result.loc[result.product_id.eq('BBB-USD') & result.segment_id.eq(2)].reset_index(drop=True)
        self.assertTrue(b.loc[:28, 'median_dollar_volume'].isna().all())
        self.assertTrue(np.isfinite(b.loc[29, 'median_dollar_volume']))
        self.assertFalse(b.loc[179, 'history_ready']); self.assertTrue(b.loc[180, 'history_ready'])

    def test_future_changes_and_truncation_do_not_change_past(self):
        p = sample_panel(); cutoff = pd.Timestamp('2020-10-01', tz='UTC')
        full = eligibility.add_research_eligibility(p)
        changed = p.copy(); changed.loc[changed.timestamp.ge(cutoff), ['close','volume']] *= 100
        changed = changed.loc[~(changed.product_id.eq('BBB-USD') & changed.timestamp.eq(cutoff + DAY))]
        mutated = eligibility.add_research_eligibility(changed)
        truncated = eligibility.add_research_eligibility(p.loc[p.timestamp.lt(cutoff)])
        past = full.loc[full.timestamp.lt(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(past, mutated.loc[mutated.timestamp.lt(cutoff)].reset_index(drop=True))
        pd.testing.assert_frame_equal(past, truncated.reset_index(drop=True))

    def test_products_are_independent_and_inputs_unchanged(self):
        p = sample_panel(); original = p.copy(deep=True)
        combined = eligibility.add_research_eligibility(p)
        alone = eligibility.add_research_eligibility(p.loc[p.product_id.eq('BBB-USD')])
        pd.testing.assert_frame_equal(combined.loc[combined.product_id.eq('BBB-USD')].reset_index(drop=True), alone)
        pd.testing.assert_frame_equal(p, original)

    def test_invalid_parameters_and_inputs_rejected(self):
        p = sample_panel()
        for arguments in [dict(history_days=0), dict(history_days=True), dict(liquidity_days=2.5),
                          dict(min_median_dollar_volume=np.nan), dict(min_median_dollar_volume=-1)]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError): eligibility.add_research_eligibility(p, **arguments)
        for column, value in [('close', 0), ('close', np.inf), ('volume', -1), ('volume', np.nan), ('product_id', 'AAA-GBP')]:
            bad = p.copy(); bad.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError): eligibility.add_research_eligibility(bad)


class FrozenHistoryTests(OfflineCase):
    @classmethod
    def setUpClass(cls):
        with patch("requests.sessions.Session.request", side_effect=AssertionError("Network access is forbidden in this suite")):
            cls.prices, cls.spec = preparation.load_saved_daily_prices(SNAPSHOT)
            cls.panel = eligibility.add_research_eligibility(cls.prices)

    def test_saved_archive_against_direct_independent_reconstruction(self):
        rows = []
        for path in SNAPSHOT.glob('*-USD_*.json'):
            raw = json.loads(path.read_text())
            from urllib.parse import parse_qs, urlparse
            query = parse_qs(urlparse(raw['request_url']).query)
            start, end = [pd.Timestamp(query[k][0]).timestamp() for k in ['start', 'end']]
            for candle in json.loads(raw['response_text']):
                if start <= candle[0] < end: rows.append([raw['product_id'], *candle])
        expected = pd.DataFrame(rows, columns=['product_id','time',*NUMERIC])
        expected['timestamp'] = pd.to_datetime(expected.pop('time'), unit='s', utc=True)
        columns = ['product_id','timestamp',*NUMERIC]
        expected = expected.sort_values(['product_id','timestamp']).reset_index(drop=True)
        pd.testing.assert_frame_equal(self.prices[columns], expected[columns], check_dtype=False)
        self.assertEqual(len(self.prices), 87_874)
        self.assertEqual(self.prices.product_id.nunique(), 59)

    def test_xrp_gap_and_numeric_audit(self):
        audit, missing = preparation.audit_daily_prices(self.prices)
        self.assertEqual(len(missing), 904)
        self.assertEqual(set(missing.product_id), {'XRP-USD'})
        self.assertEqual(missing.timestamp.min(), pd.Timestamp('2021-01-20', tz='UTC'))
        self.assertEqual(missing.timestamp.max(), pd.Timestamp('2023-07-12', tz='UTC'))
        columns = ['duplicates','off_midnight','nonfinite_rows','nonpositive_price_rows','invalid_ohlc_rows','negative_volume_rows','zero_volume_rows']
        self.assertFalse(audit[columns].to_numpy().any())

    def test_real_calendar_oracle_around_restart_and_recent_date(self):
        indexed = self.panel.set_index(['product_id','timestamp'])
        points = [('XRP-USD', '2023-07-13'), ('XRP-USD','2024-01-08'), ('XRP-USD','2024-01-09'),
                  ('BTC-USD','2019-06-29'), ('BTC-USD','2019-06-30'), ('HYPE-USD','2026-08-31'), ('FIDA-USD','2026-08-31')]
        for product, day in points:
            time = pd.Timestamp(day, tz='UTC'); expected = calendar_oracle(self.prices, product, time)
            row = indexed.loc[(product,time)]
            with self.subTest(product=product, day=day):
                self.assertEqual(bool(row.history_ready), expected[0])
                np.testing.assert_allclose(row.median_dollar_volume, expected[1], equal_nan=True)
                self.assertEqual(bool(row.research_candidate), expected[2])

    def test_reported_counts_and_full_prefix_invariance(self):
        counts = self.panel.groupby('signal_available_at')[['history_ready','research_candidate']].sum()
        dates = ['2020-01-01','2021-01-01','2022-01-01','2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-09-01']
        expected = [[8,6],[11,11],[21,21],[28,23],[36,31],[44,40],[54,39],[59,29]]
        np.testing.assert_array_equal(counts.loc[pd.to_datetime(dates,utc=True)].to_numpy(), expected)
        cutoff = pd.Timestamp('2024-01-01',tz='UTC')
        short = eligibility.add_research_eligibility(self.prices.loc[self.prices.timestamp.lt(cutoff)])
        pd.testing.assert_frame_equal(self.panel.loc[self.panel.timestamp.lt(cutoff)].reset_index(drop=True), short)


if __name__ == '__main__':
    unittest.main(verbosity=2)
