#this is the exact same file as its corresponding one in crypto_cross_sectional

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


def load_saved_daily_prices(history_dir):
    #reconstruct the declared download without filling 

    history_dir = Path(history_dir)
    spec = json.loads((history_dir / "download_spec.json").read_text(encoding="utf-8"))
    report = pd.read_csv(history_dir / "coverage_report.csv")

    keys = ["product_id", "sample"]
    expected = {(product, sample) for product in spec["products"] for sample in spec["windows"]}
    observed = set(zip(report["product_id"], report["sample"]))

    if report.duplicated(keys).any() or observed != expected:
        raise ValueError("The request grid is incomplete or duplicated.")
    if not report["outcome"].eq("ok").all():
        raise ValueError("Resolve unsuccessful requests before loading.")

    frames = []
    for row in report.itertuples(index=False):
        #resolve within this snapshot, even if project moves
        path = history_dir / Path(row.raw_path).name
        record = json.loads(path.read_text(encoding="utf-8"))

        start, end = [pd.Timestamp(value, tz="UTC") for value in spec["windows"][row.sample]]
        url = urlparse(record["request_url"])
        query = parse_qs(url.query)
        if (record["source"] != "coinbase_exchange" or record["http_status"] != 200 or record["product_id"] != row.product_id or url.netloc != "api.exchange.coinbase.com" or url.path != f"/products/{row.product_id}/candles" or query.get("granularity") != ["86400"] or pd.Timestamp(query["start"][0]) != start or pd.Timestamp(query["end"][0]) != end):
            raise ValueError(f"Snapshot metadata mismatch: {path.name}")

        payload = json.loads(record["response_text"])
        if not isinstance(payload, list) or any(not isinstance(candle, list) or len(candle) != 6 for candle in payload):
            raise ValueError(f"Invalid candle structure: {path.name}")

        frame = pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame.pop("time"), errors="raise"), unit="s", utc=True)
        if frame["timestamp"].isna().any():
            raise ValueError(f"Invalid timestamps: {path.name}")

        frame = frame.loc[frame["timestamp"].ge(start) & frame["timestamp"].lt(end)].copy()
        if len(frame) != row.rows_in_window:
            raise ValueError(f"Saved row count disagrees: {path.name}")

        frame["product_id"] = row.product_id
        frame["raw_file"] = path.name
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    numeric = ["open", "high", "low", "close", "volume"]
    prices[numeric] = prices[numeric].apply(pd.to_numeric, errors="coerce")
    if set(prices["product_id"]) != set(spec["products"]):
        raise ValueError("At least one declared product has no retained data.")

    prices = prices.sort_values(["product_id", "timestamp"]).reset_index(drop=True)
    return prices, spec

def audit_daily_prices(prices):
    #audit observed candles and gaps between first, last observations

    summaries = []
    missing_frames = []
    price_columns = ["open", "high", "low", "close"]
    numeric_columns = price_columns + ["volume"]

    for product, group in prices.groupby("product_id", sort=True):
        timestamps = pd.DatetimeIndex(group["timestamp"]).sort_values()
        expected = pd.date_range(timestamps.min(), timestamps.max(), freq="D")
        missing = expected.difference(timestamps)
        invalid_ohlc = (group["low"].gt(group["high"]) | group["open"].lt(group["low"]) | group["open"].gt(group["high"]) | group["close"].lt(group["low"]) | group["close"].gt(group["high"]))
        summaries.append({
            "product_id": product,
            "rows": len(group),
            "first_candle": timestamps.min(),
            "last_candle": timestamps.max(),
            "internal_missing_days": len(missing),
            "duplicates": int(timestamps.duplicated().sum()),
            "off_midnight": int((timestamps != timestamps.normalize()).sum()),
            "nonfinite_rows": int((~np.isfinite(group[numeric_columns])).any(axis=1).sum()),
            "nonpositive_price_rows": int(group[price_columns].le(0).any(axis=1).sum()),
            "invalid_ohlc_rows": int(invalid_ohlc.sum()),
            "negative_volume_rows": int(group["volume"].lt(0).sum()),
            "zero_volume_rows": int(group["volume"].eq(0).sum())
        })
        missing_frames.append(pd.DataFrame({"product_id": [product] * len(missing), "timestamp": missing}))

    audit = pd.DataFrame(summaries).set_index("product_id")
    missing_days = pd.concat(missing_frames, ignore_index=True)
    return audit, missing_days

def add_history_segments(prices):
    #mark uninterrupted daily histories without filling missing candles
    
    if prices.empty:
        raise ValueError("The price panel is empty.")

    frame = prices.sort_values(["product_id", "timestamp"]).reset_index(drop=True).copy()
    if frame[["product_id", "timestamp"]].isna().any().any():
        raise ValueError("Missing product identifiers or timestamps.")
    if frame.duplicated(["product_id", "timestamp"]).any():
        raise ValueError("Resolve duplicate candles before segmentation.")

    timestamps = frame["timestamp"]
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("Timestamps must be timezone-aware datetimes.")
    if str(timestamps.dt.tz) != "UTC":
        raise ValueError("Use UTC timestamps.")
    if not timestamps.eq(timestamps.dt.normalize()).all():
        raise ValueError("Daily candles must start at UTC midnight.")

    elapsed = frame.groupby("product_id")["timestamp"].diff()
    new_segment = elapsed.ne(pd.Timedelta(days=1))

    frame["segment_id"] = (new_segment.groupby(frame["product_id"]).cumsum().astype("int64"))
    frame["contiguous_observations"] = (frame.groupby(["product_id", "segment_id"]).cumcount() + 1)

    #candle's close becomes available after day finishes
    frame["signal_available_at"] = frame["timestamp"] + pd.Timedelta(days=1)

    return frame