import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests


API_BASE = "https://revx.revolut.com/api/1.0/public"


def capture_uk_market_snapshot(raw_data_dir):
    #save public UK config and quotes
    started_at = datetime.now(timezone.utc)
    run_dir = Path(raw_data_dir) / (f"revolut_uk_{started_at:%Y%m%dT%H%M%S_%fZ}")
    run_dir.mkdir(parents=True, exist_ok=False)
    endpoints = {
        "pairs": "configuration/pairs",
        "currencies": "configuration/currencies",
        "tickers": "tickers"
    }

    with requests.Session() as session:
        for position, (name, endpoint) in enumerate(endpoints.items()):
            if position:
                time.sleep(1.1)

            response = session.get(f"{API_BASE}/{endpoint}", params={"region": "UK"}, timeout=30)
            response.raise_for_status()
            snapshot = {
                "source": "revolut_x_public",
                "requested_region": "UK",
                "received_at_utc": datetime.now(timezone.utc).isoformat(),
                "request_url": response.url,
                "http_status": response.status_code,
                "payload": response.json()
            }

            path = run_dir / f"{name}.json"
            with path.open("x", encoding="utf-8") as file:
                json.dump(snapshot, file, indent=2)

            print(f"Saved: {path.name}")

    return run_dir

def capture_coinbase_products(raw_data_dir):
    #save public product catalogue, including products no longer online
    response = requests.get("https://api.exchange.coinbase.com/products", timeout=30)
    response.raise_for_status()
    received_at = datetime.now(timezone.utc)
    payload = response.json()
    snapshot = {
        "source": "coinbase_exchange",
        "received_at_utc": received_at.isoformat(),
        "request_url": response.url,
        "http_status": response.status_code,
        "payload": payload
    }

    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    path = raw_data_dir / (f"coinbase_products_{received_at:%Y%m%dT%H%M%S_%fZ}.json")
    with path.open("x", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Unexpected catalogue response; inspect {path}")

    return payload, path