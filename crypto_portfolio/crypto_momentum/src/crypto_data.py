import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
import time


API_BASE = "https://revx.revolut.com/api"
COINBASE_API_BASE = "https://api.exchange.coinbase.com"


def fetch_candle_sample(symbol, start, end, raw_data_dir):
    #request small candle sample and preserve raw response
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    raw_data_dir = Path(raw_data_dir)

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Use timezone-aware start and end dates.")
    if start >= end:
        raise ValueError("start must be earlier than end.")

    params = {
        "interval": 5,
        "since": int(start.timestamp() * 1_000),
        "until": int(end.timestamp() * 1_000),
        "region": "UK"
    }
    response = requests.get(f"{API_BASE}/1.0/public/candles/{symbol}", params=params, timeout=30)
    response.raise_for_status()
    candle_payload = response.json()
    received_at = datetime.now(timezone.utc)

    snapshot = {
        "received_at_utc": received_at.isoformat(),
        "request_url": response.url,
        "payload": candle_payload
    }

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    path = raw_data_dir / (f"candles_{symbol}_{received_at:%Y%m%dT%H%M%S_%fZ}.json")
    with path.open("x", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2)

    return candle_payload


def fetch_coinbase_daily_sample(symbol, start, end, raw_data_dir):
    #request small candle sample and preserve raw response
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    raw_data_dir = Path(raw_data_dir)

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Use timezone-aware start and end dates.")

    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")

    if start != start.normalize() or end != end.normalize():
        raise ValueError("Daily requests must use UTC midnight boundaries.")

    if not pd.Timedelta(0) < end - start <= pd.Timedelta(days=299):
        raise ValueError("Request between 1 and 299 days at a time.")

    response = requests.get(
        f"{COINBASE_API_BASE}/products/{symbol}/candles",
        params={"granularity": 86_400,
            "start": start.isoformat(),
            "end": end.isoformat()}, 
            timeout=30)
    response.raise_for_status()

    candle_payload = response.json()
    received_at = datetime.now(timezone.utc)

    snapshot = {
        "source": "coinbase_exchange",
        "symbol": symbol,
        "received_at_utc": received_at.isoformat(),
        "request_url": response.url,
        "payload": candle_payload
    }

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    path = raw_data_dir / (f"coinbase_daily_{symbol}{received_at:%Y%m%dT%H%M%S_%fZ}.json")

    with path.open("x", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2)
    if not isinstance(candle_payload, list):
        raise ValueError(f"Unexpected candle response; inspect {path}")

    return candle_payload

def fetch_coinbase_daily_history(symbol, start, end, raw_data_dir):
    #download daily candles in blocks, returning prices and request report
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Use timezone-aware start and end dates.")

    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")

    if start != start.normalize() or end != end.normalize():
        raise ValueError("Use UTC midnight boundaries.")
    if start >= end:
        raise ValueError("start must be earlier than end.")
    if end > pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("Request completed days only.")

    chunks = []
    report_rows = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + pd.Timedelta(days=250), end)

        rows = fetch_coinbase_daily_sample(
            symbol=symbol,
            start=chunk_start,
            end=chunk_end,
            raw_data_dir=raw_data_dir
            )

        candles = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        candles["timestamp"] = pd.to_datetime(candles["time"], unit="s", utc=True)
        in_window = ((candles["timestamp"] >= chunk_start) & (candles["timestamp"] < chunk_end))
        selected = candles.loc[in_window].copy()
        selected["symbol"] = symbol
        selected["source"] = "coinbase_exchange"
        chunks.append(selected)

        report_rows.append({
            "symbol": symbol,
            "requested_start": chunk_start,
            "requested_end": chunk_end,
            "expected_rows": (chunk_end - chunk_start).days,
            "returned_rows": len(candles),
            "retained_rows": len(selected),
            "duplicate_timestamps": selected["timestamp"].duplicated().sum(),
            "excluded_rows": (~in_window).sum()
        })

        print(f"{symbol}: {chunk_start.date()} to {chunk_end.date()} — retained {len(selected)} rows")
        chunk_start = chunk_end
        time.sleep(1.1)

    prices = pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    report = pd.DataFrame(report_rows)

    return prices, report