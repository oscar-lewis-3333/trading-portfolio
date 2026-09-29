"""Coinbase GBP candles for the crypto trend project, and execution fills in the house convention.

Every request is saved raw before use and replayed from disk afterwards, so a notebook
run is reproducible offline. Nothing here chooses parameters or looks at returns.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CANDLES_URL = "https://api.exchange.coinbase.com/products/{symbol}/candles"
CANDLE_COLUMNS = ["time", "low", "high", "open", "close", "volume"]
DAILY = 86400
MINUTE = 60
MAX_CANDLES_PER_REQUEST = 300


def fetch_candles(symbol, start, end, granularity, raw_dir, allow_download=False, sleep_s=1.1):
    #one saved request per (symbol, granularity, window); cached files are never rewritten
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Use timezone-aware start < end.")
    if (end - start) / pd.Timedelta(seconds=granularity) > MAX_CANDLES_PER_REQUEST:
        raise ValueError("Window exceeds Coinbase's 300 candles per request.")

    folder = Path(raw_dir) / ("daily" if granularity == DAILY else "minute")
    path = folder / f"{symbol}_{granularity}s_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json"
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing saved request: {path.name}. Set allow_download=True to fetch it.")
        folder.mkdir(parents=True, exist_ok=True)
        params = {"granularity": granularity, "start": start.isoformat(), "end": end.isoformat()}
        record = {"source": "coinbase_exchange", "symbol": symbol, "granularity": granularity,
                  "start": start.isoformat(), "end": end.isoformat(), "http_status": None,
                  "payload": None, "error": None}
        time.sleep(sleep_s)
        try:
            response = requests.get(CANDLES_URL.format(symbol=symbol), params=params, timeout=30)
            record["http_status"] = response.status_code
            record["request_url"] = response.url
            record["payload"] = response.json() if response.status_code == 200 else None
            if response.status_code != 200:
                record["error"] = response.text[:500]
        except (requests.RequestException, ValueError) as error:
            record["error"] = str(error)
        record["received_at_utc"] = datetime.now(timezone.utc).isoformat()
        with path.open("x", encoding="utf-8") as file:
            json.dump(record, file)

    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("http_status") != 200 or not isinstance(record.get("payload"), list):
        raise ValueError(f"Saved request failed ({record.get('http_status')}): {path.name}. Inspect it before retrying.")

    candles = pd.DataFrame(record["payload"], columns=CANDLE_COLUMNS)
    candles["timestamp"] = pd.to_datetime(candles["time"], unit="s", utc=True)
    candles = candles.loc[(candles["timestamp"] >= start) & (candles["timestamp"] < end)]
    candles.insert(0, "symbol", symbol)
    return candles.drop(columns="time").sort_values("timestamp").reset_index(drop=True)


def load_daily_history(symbols, start, end, raw_dir, allow_download=False):
    #completed UTC days in [start, end), fetched in 250-day blocks
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    if start != start.normalize() or end != end.normalize():
        raise ValueError("Use UTC midnight boundaries.")
    if end > pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("Request completed days only.")

    parts = []
    for symbol in symbols:
        block = start
        while block < end:
            block_end = min(block + pd.Timedelta(days=250), end)
            parts.append(fetch_candles(symbol, block, block_end, DAILY, raw_dir, allow_download))
            block = block_end
    prices = pd.concat(parts, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if prices.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Duplicate daily candles.")
    return prices


def audit_daily(prices, symbols, start, end):
    #one row per symbol: missing days and invalid values, never repaired silently
    expected = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="D", inclusive="left")
    rows = []
    for symbol in symbols:
        frame = prices.loc[prices["symbol"] == symbol].set_index("timestamp")
        values = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
        rows.append({"symbol": symbol, "rows": len(frame),
                     "missing_days": len(expected.difference(frame.index)),
                     "nonpositive_or_nonfinite": int((~np.isfinite(values) | (values <= 0)).any(axis=1).sum()),
                     "zero_volume_days": int(frame["volume"].le(0).sum())})
    return pd.DataFrame(rows).set_index("symbol")


def to_factory_panel(prices):
    #signal_factory's long panel; every listed day is eligible for this fixed universe
    panel = prices.rename(columns={"symbol": "product_id"})[["product_id", "timestamp", "close"]].copy()
    panel["close"] = panel["close"].astype(float)
    panel["eligible"] = True
    return panel.sort_values(["timestamp", "product_id"]).reset_index(drop=True)


def load_monday_minutes(symbols, days, raw_dir, allow_download=False, window_minutes=60):
    #1-minute candles from 00:00 to 01:00 UTC on each decision day
    parts = []
    for day in pd.DatetimeIndex(days):
        for symbol in symbols:
            parts.append(fetch_candles(symbol, day, day + pd.Timedelta(minutes=window_minutes), MINUTE,
                                       raw_dir, allow_download))
    return pd.concat(parts, ignore_index=True)


def minute_fills(minutes, days, symbols, first_minute=5, last_minute=60, max_age_minutes=5):
    #house rule: from 00:05, the first minute at which every symbol has a completed, positive-volume
    #candle no more than max_age old; each symbol fills at that candle's close
    rows = []
    for day in pd.DatetimeIndex(days):
        today = minutes.loc[(minutes["timestamp"] >= day) & (minutes["timestamp"] < day + pd.Timedelta(minutes=last_minute))]
        usable = today.loc[today["volume"].astype(float) > 0]
        row = {"day": day, "execution_at": pd.NaT, "status": "unresolved"}
        for minute in range(first_minute, last_minute + 1):
            at = day + pd.Timedelta(minutes=minute)
            fresh = usable.loc[(usable["timestamp"] + pd.Timedelta(minutes=1) <= at)
                               & (at - usable["timestamp"] <= pd.Timedelta(minutes=max_age_minutes))]
            latest = fresh.sort_values("timestamp").groupby("symbol").tail(1).set_index("symbol")
            if set(symbols).issubset(latest.index):
                row.update(execution_at=at, status="ready")
                for symbol in symbols:
                    row[symbol] = float(latest.loc[symbol, "close"])
                    row[f"{symbol}_candle"] = latest.loc[symbol, "timestamp"]
                break
        rows.append(row)
    return pd.DataFrame(rows).set_index("day")


def executions_from_fills(decisions, fills, prices, symbols):
    #signal_factory executions: fill on the decision day, as a return from the previous daily close
    closes = prices.pivot(index="timestamp", columns="symbol", values="close").astype(float)
    frame = pd.DataFrame(index=pd.DatetimeIndex(decisions, name="decision_at"))
    frame["execution_day"] = frame.index
    unresolved = fills.reindex(frame.index)["status"].ne("ready")
    if unresolved.any():
        raise ValueError(f"No fill within the hour on: {list(frame.index[unresolved].date)}")
    for symbol in symbols:
        previous_close = closes[symbol].reindex(frame.index - pd.Timedelta(days=1)).to_numpy()
        frame[symbol] = fills.loc[frame.index, symbol].to_numpy(dtype=float) / previous_close - 1.0
    return frame


def executions_next_open(decisions, prices, symbols):
    #the crypto_momentum convention: execute at the next day's open, for reconciliation only
    opens = prices.pivot(index="timestamp", columns="symbol", values="open").astype(float)
    closes = prices.pivot(index="timestamp", columns="symbol", values="close").astype(float)
    frame = pd.DataFrame(index=pd.DatetimeIndex(decisions, name="decision_at"))
    frame["execution_day"] = frame.index + pd.Timedelta(days=1)
    for symbol in symbols:
        frame[symbol] = (opens[symbol].reindex(frame["execution_day"]).to_numpy()
                         / closes[symbol].reindex(frame.index).to_numpy() - 1.0)
    return frame
