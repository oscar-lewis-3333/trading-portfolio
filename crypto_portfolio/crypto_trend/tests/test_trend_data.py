import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import trend_data  # noqa: E402

MONDAY = pd.Timestamp("2026-09-21", tz="UTC")


def minute_rows(symbol, minutes, price=100.0, volume=1.0):
    return pd.DataFrame({"symbol": symbol,
                         "timestamp": [MONDAY + pd.Timedelta(minutes=m) for m in minutes],
                         "open": price, "high": price, "low": price,
                         "close": [price + m for m in minutes], "volume": volume})


def test_fill_at_first_common_fresh_minute():
    minutes = pd.concat([minute_rows("BTC-GBP", [0, 1, 2, 3, 4, 5, 6]),
                         minute_rows("ETH-GBP", [0, 1, 2, 3, 4])])
    fills = trend_data.minute_fills(minutes, [MONDAY], ["BTC-GBP", "ETH-GBP"])
    row = fills.loc[MONDAY]
    assert row["status"] == "ready"
    assert row["execution_at"] == MONDAY + pd.Timedelta(minutes=5)
    #at 00:05 the latest completed candles started at 00:04
    assert row["BTC-GBP"] == 104 and row["ETH-GBP"] == 104


def test_stale_or_empty_candles_delay_the_fill():
    #ETH's 23:50 candle is too old by 00:05, its 00:10 candle has no volume, next trade at 00:20
    minutes = pd.concat([minute_rows("BTC-GBP", list(range(0, 30))),
                         minute_rows("ETH-GBP", [-10]),
                         minute_rows("ETH-GBP", [10], volume=0.0),
                         minute_rows("ETH-GBP", [20])])
    fills = trend_data.minute_fills(minutes, [MONDAY], ["BTC-GBP", "ETH-GBP"])
    row = fills.loc[MONDAY]
    assert row["execution_at"] == MONDAY + pd.Timedelta(minutes=21)
    assert row["ETH-GBP"] == 120 and row["BTC-GBP"] == 120


def test_no_fill_within_the_hour_is_unresolved():
    minutes = minute_rows("BTC-GBP", [0, 1, 2])
    fills = trend_data.minute_fills(minutes, [MONDAY], ["BTC-GBP", "ETH-GBP"])
    assert fills.loc[MONDAY, "status"] == "unresolved"


def daily(symbol, days, opens, closes):
    return pd.DataFrame({"symbol": symbol, "timestamp": days, "open": opens, "close": closes,
                         "high": closes, "low": closes, "volume": 1.0})


def test_executions_measure_fills_from_the_previous_close():
    days = pd.date_range(MONDAY - pd.Timedelta(days=1), periods=3, freq="D")
    prices = daily("BTC-GBP", days, [100.0, 100.0, 110.0], [100.0, 110.0, 120.0])
    fills = pd.DataFrame({"execution_at": [MONDAY + pd.Timedelta(minutes=5)], "status": ["ready"],
                          "BTC-GBP": [102.0]}, index=pd.DatetimeIndex([MONDAY], name="day"))
    executions = trend_data.executions_from_fills([MONDAY], fills, prices, ["BTC-GBP"])
    assert executions.loc[MONDAY, "execution_day"] == MONDAY
    assert executions.loc[MONDAY, "BTC-GBP"] == pytest.approx(0.02)

    following = trend_data.executions_next_open([MONDAY], prices, ["BTC-GBP"])
    assert following.loc[MONDAY, "execution_day"] == MONDAY + pd.Timedelta(days=1)
    assert following.loc[MONDAY, "BTC-GBP"] == pytest.approx(0.0)


def test_unresolved_fill_is_an_error():
    days = pd.date_range(MONDAY - pd.Timedelta(days=1), periods=2, freq="D")
    prices = daily("BTC-GBP", days, [100.0, 100.0], [100.0, 100.0])
    fills = pd.DataFrame({"execution_at": [pd.NaT], "status": ["unresolved"], "BTC-GBP": [np.nan]},
                         index=pd.DatetimeIndex([MONDAY], name="day"))
    with pytest.raises(ValueError, match="No fill"):
        trend_data.executions_from_fills([MONDAY], fills, prices, ["BTC-GBP"])


def test_cached_request_replays_offline(tmp_path):
    start, end = MONDAY, MONDAY + pd.Timedelta(minutes=2)
    folder = tmp_path / "minute"
    folder.mkdir()
    payload = [[int((MONDAY + pd.Timedelta(minutes=1)).timestamp()), 1, 2, 1.5, 1.8, 3.0],
               [int(MONDAY.timestamp()), 1, 2, 1.4, 1.5, 2.0]]
    record = {"source": "coinbase_exchange", "http_status": 200, "payload": payload}
    (folder / f"BTC-GBP_60s_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json").write_text(json.dumps(record))

    candles = trend_data.fetch_candles("BTC-GBP", start, end, 60, tmp_path)
    assert list(candles["close"]) == [1.5, 1.8]
    with pytest.raises(FileNotFoundError):
        trend_data.fetch_candles("ETH-GBP", start, end, 60, tmp_path)


def test_failed_saved_request_is_not_used(tmp_path):
    start, end = MONDAY, MONDAY + pd.Timedelta(minutes=2)
    (tmp_path / "minute").mkdir()
    record = {"source": "coinbase_exchange", "http_status": 429, "payload": None, "error": "slow down"}
    (tmp_path / "minute" / f"BTC-GBP_60s_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="failed"):
        trend_data.fetch_candles("BTC-GBP", start, end, 60, tmp_path)
