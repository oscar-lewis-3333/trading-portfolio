
#offline candle loading and completed minute execution semantics

import json
import pandas as pd
import pytest
from foundation.data.crypto_reference import fetch_candles, MINUTE
from foundation.signals.crypto_trend import minute_fills, load_monday_minutes

DAY = pd.Timestamp('2024-01-01', tz='UTC')

def test_missing_input_and_download_request_fail_without_creating_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_monday_minutes(['BTC-GBP'], [DAY], tmp_path)
    with pytest.raises(ValueError, match='offline'):
        fetch_candles('BTC-GBP', DAY, DAY + pd.Timedelta(hours=1), MINUTE, tmp_path, allow_download=True)
    assert list(tmp_path.iterdir()) == []

def test_cached_candles_preserve_clock_and_are_not_rewritten(tmp_path):
    folder = tmp_path / 'minute'
    folder.mkdir()
    path = folder / 'BTC-GBP_60s_20240101T0000_20240101T0100.json'
    record = {'http_status': 200, 'payload': [[int(DAY.timestamp()) + m*60, 90, 110, 100, 100+m, 1] for m in [5, 4]]}
    path.write_text(json.dumps(record))

    before = path.read_bytes()
    frame = load_monday_minutes(['BTC-GBP'], [DAY], tmp_path)
    fills = minute_fills(frame, [DAY], ['BTC-GBP'])
    assert fills.loc[DAY, 'execution_at'] == DAY + pd.Timedelta(minutes=5)
    assert fills.loc[DAY, 'BTC-GBP'] == 104
    assert path.read_bytes() == before

def test_stale_empty_and_incomplete_candles_do_not_fill():
    minutes = pd.DataFrame({'symbol': ['BTC-GBP']*3, 'timestamp': [DAY-pd.Timedelta(minutes=10), DAY, DAY+pd.Timedelta(minutes=20)], 'volume': [1., 0., 1.], 'close': [50., 80., 120.]})
    fills = minute_fills(minutes, [DAY], ['BTC-GBP'])
    assert fills.loc[DAY, 'execution_at'] == DAY + pd.Timedelta(minutes=21)
    assert fills.loc[DAY, 'BTC-GBP'] == 120
    assert minute_fills(minutes, [DAY], ['BTC-GBP','ETH-GBP']).loc[DAY,'status'] == 'unresolved'

def test_failed_cached_response_is_rejected(tmp_path):
    folder = tmp_path / 'minute'
    folder.mkdir()
    (folder / 'BTC-GBP_60s_20240101T0000_20240101T0100.json').write_text(json.dumps({'http_status': 429, 'payload': None}))
    with pytest.raises(ValueError, match='Saved request failed'):
        load_monday_minutes(['BTC-GBP'], [DAY], tmp_path)
