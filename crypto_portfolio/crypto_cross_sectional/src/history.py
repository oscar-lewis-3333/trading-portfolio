import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import time



def probe_coinbase_daily(product_id, start, end, raw_data_dir):
    #save one respomse and audit timestamp coverage in [start, end)
    
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Use timezone-aware boundaries.")

    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.normalize() or end != end.normalize():
        raise ValueError("Use UTC midnight boundaries.")
    if not pd.Timedelta(days=1) <= end - start <= pd.Timedelta(days=299):
        raise ValueError("Request between 1 and 299 complete days.")
    if end > pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("Request completed days only.")
    
    url = requests.Request("GET", f"https://api.exchange.coinbase.com/products/{product_id}/candles",
        params={"granularity": 86400, "start": start.isoformat(), "end": end.isoformat()}).prepare().url

    record = dict(
        source="coinbase_exchange",
        product_id=product_id,
        request_url=url,
        http_status=None,
        response_text=None,
        error=None)
    try:
        response = requests.get(url, timeout=30)
        record.update(http_status=response.status_code, response_text=response.text)
    except requests.RequestException as error:
        record["error"] = str(error)

    received_at = datetime.now(timezone.utc)
    record["received_at_utc"] = received_at.isoformat()
    directory = Path(raw_data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"{product_id}_{start:%Y%m%d}_{received_at:%Y%m%dT%H%M%S_%fZ}.json")
    with path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2)

    result = dict(
        product_id=product_id,
        requested_start=start,
        requested_end=end,
        expected_days=(end - start).days,
        http_status=record["http_status"],
        raw_path=str(path))
    
    if record["error"] is not None:
        return dict(result, outcome="request_error")
    if record["http_status"] != 200:
        return dict(result, outcome="http_error")

    try:
        payload = json.loads(record["response_text"])
    except (ValueError, TypeError):
        return dict(result, outcome="invalid_response")
    
    if not isinstance(payload, list) or any(not isinstance(row, list) or len(row) != 6 for row in payload):
        return dict(result, outcome="invalid_response")

    seconds = pd.to_numeric(pd.Series([row[0] for row in payload], dtype=object),errors="coerce")
    timestamps = pd.DatetimeIndex(pd.to_datetime(seconds, unit="s", utc=True, errors="coerce"))
    if timestamps.isna().any():
        return dict(result, outcome="invalid_timestamps", returned_rows=len(payload))

    selected = timestamps[(timestamps >= start) & (timestamps < end)]
    expected = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")

    return dict(result,
        outcome="ok",
        returned_rows=len(payload),
        rows_in_window=len(selected),
        missing_days=len(expected.difference(selected)),
        duplicate_timestamps=int(selected.duplicated().sum()),
        off_midnight_rows=int((selected != selected.normalize()).sum()),
        excluded_rows=len(timestamps) - len(selected),
        first_timestamp=selected.min(),
        last_timestamp=selected.max())

def run_history_probes(product_ids, windows, raw_data_dir):
    #probe a product/window grid. save progress after every request

    products = sorted(set(product_ids))
    if not products or not windows:
        raise ValueError("Provide products and probe windows.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(raw_data_dir) / f"coinbase_coverage_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    total = len(products) * len(windows)
    print(f"Planned requests: {total}")
    print(f"Saving to: {run_dir}")

    for product in products:
        for sample, (start, end) in windows.items():
            if rows:
                time.sleep(1.1)

            row = probe_coinbase_daily(product_id=product, start=pd.Timestamp(start, tz="UTC"), end=pd.Timestamp(end, tz="UTC"),raw_data_dir=run_dir)
            row["sample"] = sample
            rows.append(row)

            report = pd.DataFrame(rows)
            report.to_csv(run_dir / "coverage_report.csv", index=False)

            status = row["http_status"]
            stop = (row["outcome"] == "request_error" or status in (401, 403, 429) or (status is not None and status >= 500))
            if stop:
                print(f"Stopped after {len(rows)}/{total} requests: {row['outcome']}, HTTP {status}. Inspect the saved response before continuing.")
                return report, run_dir

        print(f"Completed {len(rows)}/{total} requests")

    return report, run_dir

def build_history_windows(start, end, chunk_days=250):
    #split UTC date strings into consecutive [start, end) intervals
    
    start = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")

    if start != start.normalize() or end != end.normalize():
        raise ValueError("Use UTC midnight boundaries.")
    if start >= end:
        raise ValueError("start must precede end.")
    if end > pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("Request completed days only.")
    if not isinstance(chunk_days, int) or not 1 <= chunk_days <= 299:
        raise ValueError("chunk_days must be an integer from 1 to 299.")

    windows = {}
    cursor = start

    while cursor < end:
        boundary = min(cursor + pd.Timedelta(days=chunk_days), end)
        label = f"{cursor:%Y%m%d}_{boundary:%Y%m%d}"
        windows[label] = (cursor.date().isoformat(), boundary.date().isoformat())
        cursor = boundary

    return windows