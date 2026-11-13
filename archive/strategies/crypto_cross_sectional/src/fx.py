import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


FX_URL = "https://api.frankfurter.dev/v2/providers/ecb/rates"

#crypto data is more readily available in USD,

def capture_fx_history(raw_data_dir, start_date, end_date):
    #save ECB GBP/USD obs. request dates inclusive.
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start > end:
        raise ValueError("start_date must not exceed end_date.")
    if end >= datetime.now(timezone.utc).date():
        raise ValueError("Request completed dates only.")

    params = {
        "base": "GBP",
        "quotes": "USD",
        "from": start.isoformat(),
        "to": end.isoformat()
    }

    response = requests.get(FX_URL, params=params, timeout=60)
    received_at = datetime.now(timezone.utc)

    record = {
        "source": "frankfurter_ecb",
        "received_at_utc": received_at.isoformat(),
        "request_url": response.url,
        "request_params": params,
        "http_status": response.status_code,
        "response_text": response.text
    }

    directory = Path(raw_data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"ecb_gbp_usd_{received_at:%Y%m%dT%H%M%S_%fZ}.json")

    #preserve response before checking its status/contents
    with path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    if response.status_code != 200:
        raise RuntimeError(f"FX request failed; inspect {path}")

    return path

def load_fx_history(raw_path):
    #read saved observations without making another network request

    record = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if (record.get("source") != "frankfurter_ecb" or record.get("http_status") != 200):
        raise ValueError("Expected a successful Frankfurter ECB snapshot.")

    params = record["request_params"]
    if params.get("base") != "GBP" or params.get("quotes") != "USD":
        raise ValueError("Expected USD per GBP.")

    payload = json.loads(record["response_text"])
    if not isinstance(payload, list) or not payload:
        raise ValueError("FX response must contain observations.")

    frame = pd.DataFrame(payload)
    required = {"date", "base", "quote", "rate"}
    if not required.issubset(frame.columns):
        raise ValueError("Unexpected FX response columns.")
    if not frame["base"].eq("GBP").all():
        raise ValueError("Unexpected base currency.")
    if not frame["quote"].eq("USD").all():
        raise ValueError("Unexpected quote currency.")

    frame["reference_date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", utc=True, errors="raise")
    frame["usd_per_gbp"] = pd.to_numeric(frame["rate"], errors="raise")
    rates = frame["usd_per_gbp"].to_numpy(dtype=float)
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("FX rates must be finite and positive.")

    dates = frame["reference_date"]
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Missing or duplicate FX dates.")
    if dates.ne(dates.dt.normalize()).any():
        raise ValueError("Expected date-only FX observations.")
    if dates.gt(pd.Timestamp(params["to"], tz="UTC")).any():
        raise ValueError("FX response extends beyond the requested end.")

    #modelling convention
    frame["assumed_available_at"] = (frame["reference_date"] + pd.Timedelta(days=1))

    return (frame[["reference_date", "assumed_available_at", "usd_per_gbp"]
        ].sort_values("reference_date").reset_index(drop=True))

def build_gbp_valuation_prices(prices, fx_observations, max_age_days=7):
    #add synthetic GBP open/close prices and return for FX alignment audit
    if (not isinstance(max_age_days, int) or isinstance(max_age_days, bool) or max_age_days < 1):
        raise ValueError("max_age_days must be a positive integer.")

    required_prices = {"product_id", "timestamp", "open", "close"}
    required_fx = {
        "reference_date",
        "assumed_available_at",
        "usd_per_gbp"
    }
    if not required_prices.issubset(prices.columns):
        raise ValueError("Missing required price columns.")
    if not required_fx.issubset(fx_observations.columns):
        raise ValueError("Missing required FX columns.")
    if prices.empty or fx_observations.empty:
        raise ValueError("Prices and FX observations must be non-empty.")

    frame = prices.copy()
    rates = fx_observations[list(sorted(required_fx))].copy()
    if not frame["product_id"].str.endswith("-USD", na=False).all():
        raise ValueError("Expected USD-denominated crypto prices.")
    if frame.duplicated(["product_id", "timestamp"]).any():
        raise ValueError("Duplicate crypto candles.")

    for times in (frame["timestamp"], rates["reference_date"], rates["assumed_available_at"]):
        if (not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != "UTC"):
            raise ValueError("Use timezone-aware UTC timestamps.")
        if times.isna().any() or times.ne(times.dt.normalize()).any():
            raise ValueError("Expected valid UTC midnight timestamps.")

    if rates["reference_date"].duplicated().any():
        raise ValueError("Duplicate FX reference dates.")

    expected_availability = rates["reference_date"] + pd.Timedelta(days=1)
    if not rates["assumed_available_at"].eq(expected_availability).all():
        raise ValueError("FX availability must follow our next-midnight rule.")

    for values in (frame[["open", "close"]].to_numpy(dtype=float), rates["usd_per_gbp"].to_numpy(dtype=float)):
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Prices and FX rates must be finite and positive.")

    #one shared fx observation per valuation time, across all assets
    valuation_times = pd.concat([frame["timestamp"], frame["timestamp"] + pd.Timedelta(days=1)], ignore_index=True)
    grid = valuation_times.drop_duplicates().sort_values().rename("valuation_at").to_frame().reset_index(drop=True)

    fx_audit = pd.merge_asof(grid, rates.sort_values("assumed_available_at"),left_on="valuation_at",
        right_on="assumed_available_at", direction="backward", allow_exact_matches=True)

    fx_audit["reference_age_days"] = (fx_audit["valuation_at"] - fx_audit["reference_date"]).dt.total_seconds() / 86400
    invalid = (fx_audit["usd_per_gbp"].isna() | fx_audit["reference_age_days"].gt(max_age_days) | fx_audit["assumed_available_at"].gt(fx_audit["valuation_at"]))
    if invalid.any():
        examples = fx_audit.loc[invalid].head().to_string(index=False)
        raise ValueError(f"Missing, stale or future FX observations:\n{examples}")

    rate_by_time = fx_audit.set_index("valuation_at")["usd_per_gbp"]
    frame["open_usd_per_gbp"] = frame["timestamp"].map(rate_by_time)
    frame["close_usd_per_gbp"] = (frame["timestamp"] + pd.Timedelta(days=1)).map(rate_by_time)
    frame["open_gbp"] = frame["open"] / frame["open_usd_per_gbp"]
    frame["close_gbp"] = frame["close"] / frame["close_usd_per_gbp"]

    converted = frame[["open_gbp", "close_gbp"]].to_numpy(dtype=float)
    if not np.isfinite(converted).all() or (converted <= 0).any():
        raise ValueError("Invalid converted GBP prices.")

    return frame, fx_audit