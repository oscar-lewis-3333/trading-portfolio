
#small cached feasibility probe for execution shortly after UTC midnight
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


MINUTE = pd.Timedelta(minutes=1)


def probe_execution_window(product_id, day, cache_dir, allow_download=False, window_minutes=15):
    #audit [00:00, 00:15) one-min candles. Existing raw response always reused. Failed responses remain cached for inspection

    if not re.fullmatch(r"[A-Z0-9]+-USD", product_id):
        raise ValueError("Expected a Coinbase USD product ID.")
    start = pd.Timestamp(day)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    if pd.isna(start) or start != start.normalize():
        raise ValueError("day must identify a UTC midnight.")
    if (isinstance(window_minutes, bool) or not isinstance(window_minutes, int) or not 15 <= window_minutes <= 300):
        raise ValueError("window_minutes must be an integer from 15 to 300.")

    end = start + window_minutes * MINUTE
    execution_bar = start + 5 * MINUTE
    if end > pd.Timestamp.now(tz="UTC"):
        raise ValueError("Request completed intervals only.")

    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": 60, "start": start.isoformat(), "end": end.isoformat()}
    specification = {"product_id": product_id, "url": url, "params": params}
    path = (Path(cache_dir) / f"{product_id}_{start:%Y%m%d}_0000_{end:%H%M}_60s.json")
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(f"Missing saved probe: {path}. Set allow_download=True for the initial capture.")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "source": "coinbase_exchange",
            "request": specification,
            "http_status": None,
            "response_text": None,
            "error": None
        }
        time.sleep(1.1)
        try:
            response = requests.get(url, params=params, timeout=30)
            record.update(http_status=response.status_code, response_text=response.text, response_url=response.url)
        except requests.RequestException as error:
            record["error"] = str(error)
        record["received_at_utc"] = datetime.now(timezone.utc).isoformat()
        with path.open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2)

    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("source") != "coinbase_exchange" or record.get("request") != specification:
        raise ValueError(f"Cached probe metadata mismatch: {path}")

    result = {
        "product_id": product_id,
        "day": start,
        "outcome": "ok",
        "http_status": record["http_status"],
        "expected_minutes": window_minutes,
        "execution_bar_start": execution_bar,
        "execution_proxy_available": False,
        "execution_open_usd": np.nan,
        "raw_path": str(path)
    }

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

    candles = pd.DataFrame(payload, columns=["time", "low", "high", "open", "close", "volume"])
    seconds = pd.to_numeric(candles["time"], errors="coerce")

    candles["timestamp"] = pd.to_datetime(seconds, unit="s", utc=True, errors="coerce")
    if candles["timestamp"].isna().any():
        return dict(result, outcome="invalid_timestamps")

    window = candles.loc[candles["timestamp"].ge(start) & candles["timestamp"].lt(end)].copy()
    numeric = ["low", "high", "open", "close", "volume"]
    window[numeric] = window[numeric].apply(pd.to_numeric, errors="coerce")
    expected = pd.date_range(start, end - MINUTE, freq="min")
    observed = pd.DatetimeIndex(window["timestamp"])

    invalid = (
        ~np.isfinite(window[numeric].to_numpy(dtype=float)).all(axis=1)
        | window[["low", "high", "open", "close"]].le(0).any(axis=1)
        | window["volume"].lt(0)
        | window["low"].gt(window["high"])
        | window["open"].lt(window["low"]) | window["open"].gt(window["high"])
        | window["close"].lt(window["low"]) | window["close"].gt(window["high"]))
    
    duplicates = int(observed.duplicated().sum())
    off_minute = int((observed != observed.floor("min")).sum())
    at_execution = window.loc[window["timestamp"].eq(execution_bar)]
    usable = (len(at_execution) == 1 and duplicates == 0 and off_minute == 0 and not invalid.any() and at_execution["volume"].iloc[0] > 0)

    result.update(
        returned_rows=len(candles),
        rows_in_window=len(window),
        missing_minutes=len(expected.difference(observed)),
        duplicate_timestamps=duplicates,
        off_minute_rows=off_minute,
        invalid_ohlcv_rows=int(invalid.sum()),
        zero_volume_rows=int(window["volume"].eq(0).sum()),
        excluded_rows=len(candles) - len(window),
        execution_bar_present=len(at_execution) > 0,
        execution_proxy_available=bool(usable),
        execution_open_usd=float(at_execution["open"].iloc[0]) if usable else np.nan
    )
    if duplicates or off_minute or invalid.any():
        result["outcome"] = "invalid_candles"
    return result


def run_execution_probes(product_ids, days, cache_dir, allow_download=False):
    #run a small grid. cached calls make no network requests or sleeps

    if not product_ids or not days:
        raise ValueError("Provide products and dates.")
    rows = []
    for product in sorted(set(product_ids)):
        for day in sorted(set(days)):
            row = probe_execution_window(product, day, cache_dir, allow_download)
            rows.append(row)
            if row["outcome"] != "ok":
                print(f"Stopped: {product}, {day}, {row['outcome']}. Inspect {row['raw_path']}")
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)

def probe_universe_execution(universe_members, days, cache_dir, allow_download=False):
    #probe only assets belonging to each requested decision date
    dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))

    if dates.empty or dates.hasnans or dates.has_duplicates:
        raise ValueError("Provide valid, unique probe dates.")

    selected = universe_members.loc[universe_members["decision_at"].isin(dates), ["product_id", "decision_at"]].copy()
    if selected.duplicated(["product_id", "decision_at"]).any():
        raise ValueError("Duplicate universe members.")

    missing_dates = dates.difference(pd.DatetimeIndex(selected["decision_at"].unique()))
    if not missing_dates.empty:
        raise ValueError(f"No universe members for: {missing_dates.tolist()}")

    selected = selected.sort_values(["decision_at", "product_id"])
    rows = []

    for item in selected.itertuples(index=False):
        row = probe_execution_window(
            product_id=item.product_id,
            day=item.decision_at,
            cache_dir=cache_dir,
            allow_download=allow_download)
        rows.append(row)

        if row["outcome"] != "ok":
            print(f"Stopped: {item.product_id}, {item.decision_at}, {row['outcome']}. Inspect {row['raw_path']}")
            break

    return pd.DataFrame(rows)

def read_execution_reference(product_id, day, cache_dir, max_age_minutes=5):
    #read cached last-trade mark for 00:05. assume each minute candle is available at its end, with latency unmeasured.

    if not np.isfinite(max_age_minutes) or not 1 <= max_age_minutes <= 5:
        raise ValueError("max_age_minutes must be between 1 and 5 for this probe.")

    #reuse metadata and candle validation. downloads never done here
    audit = probe_execution_window(product_id, day, cache_dir, allow_download=False)
    if audit["outcome"] != "ok":
        raise ValueError(f"Invalid saved probe: {audit['outcome']}: {audit['raw_path']}")

    record = json.loads(Path(audit["raw_path"]).read_text(encoding="utf-8"))
    candles = pd.DataFrame(json.loads(record["response_text"]), columns=["time", "low", "high", "open", "close", "volume"])
    candles["timestamp"] = pd.to_datetime(pd.to_numeric(candles["time"]), unit="s", utc=True)
    for column in ["close", "volume"]:
        candles[column] = pd.to_numeric(candles[column])

    execution_at = audit["execution_bar_start"]
    completed = candles.loc[candles["timestamp"].ge(audit["day"])
        & (candles["timestamp"] + MINUTE).le(execution_at)
        & candles["volume"].gt(0)].sort_values("timestamp")

    result = {
        "product_id": product_id,
        "day": audit["day"],
        "execution_at": execution_at,
        "reference_bar_start": pd.NaT,
        "assumed_available_at": pd.NaT,
        "age_upper_bound_minutes": np.nan,
        "reference_price_usd": np.nan,
        "reference_available": False,
        "status": "no_completed_trade_bar",
        "raw_path": audit["raw_path"]
    }
    if completed.empty:
        return result

    latest = completed.iloc[-1]
    age = (execution_at - latest["timestamp"]) / MINUTE
    usable = bool(age <= max_age_minutes)

    result.update(
        reference_bar_start=latest["timestamp"],
        assumed_available_at=latest["timestamp"] + MINUTE,
        age_upper_bound_minutes=float(age),
        reference_price_usd=float(latest["close"]) if usable else np.nan,
        reference_available=usable,
        status="ok" if usable else "stale_reference"
    )
    return result


def build_reference_request_plan(universe_targets, cache_dir):
    #start from cash and assume scheduled rebalance completed. planned for 00:05
    times = universe_targets.index
    if (universe_targets.empty or not isinstance(times, pd.DatetimeIndex) or times.tz is None or str(times.tz) != "UTC" or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing or not (times == times.normalize()).all()):
        raise ValueError("Provide sorted, unique decision rows at UTC midnight.")
    if universe_targets.columns.has_duplicates or "CASH" not in universe_targets:
        raise ValueError("Provide unique asset columns and a CASH column.")
    weights = universe_targets.to_numpy(dtype=float)
    if (not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Weights must be finite, non-negative and sum to one.")

    active = universe_targets.drop(columns="CASH").gt(0)
    if any(not re.fullmatch(r"[A-Z0-9]+-USD", str(p)) for p in active.columns):
        raise ValueError("Expected Coinbase USD product IDs.")

    rows = []
    previous = set()

    for day, membership in active.iterrows():
        current = set(membership.index[membership])

        for product_id in sorted(current | previous):
            path = (Path(cache_dir) / f"{product_id}_{day:%Y%m%d}_0000_0015_60s.json")
            rows.append({
                "product_id": product_id,
                "day": day,
                "execution_at": day + 5 * MINUTE,
                "current_member": product_id in current,
                "previous_member": product_id in previous,
                "exit_only": product_id in previous - current,
                "cache_exists": path.is_file(),
                "raw_path": str(path)
            })

        previous = current

    return pd.DataFrame(rows, columns=["product_id", "day", "execution_at", "current_member", "previous_member", "exit_only", "cache_exists", "raw_path"])

def capture_execution_references(request_plan, cache_dir, allow_download=False, max_new_requests=200, max_age_minutes=5):
    #capture missing windows and reuse saved responses
    
    if max_new_requests is not None and (isinstance(max_new_requests, bool) or not isinstance(max_new_requests, int) or max_new_requests < 1):
        raise ValueError("max_new_requests must be a positive integer or None.")
    if not np.isfinite(max_age_minutes) or not 1 <= max_age_minutes <= 5:
        raise ValueError("max_age_minutes must be between 1 and 5.")

    plan = request_plan[["product_id", "day"]].copy()
    plan["day"] = pd.to_datetime(plan["day"], utc=True)
    if (plan.isna().any().any() or plan.duplicated(["product_id", "day"]).any() or plan["day"].ne(plan["day"].dt.normalize()).any()
        or not plan["product_id"].map(lambda p: isinstance(p, str) and bool(re.fullmatch(r"[A-Z0-9]+-USD", p))).all()):
        raise ValueError("Provide unique product/UTC-midnight pairs.")

    plan = plan.sort_values(["day", "product_id"])
    rows, new_requests = [], 0
    stop_reason = "complete"
    for item in plan.itertuples(index=False):
        path = (Path(cache_dir) / f"{item.product_id}_{item.day:%Y%m%d}_0000_0015_60s.json")
        cached = path.is_file()  #recheck disk. plan may be stale

        if not cached:
            if not allow_download:
                stop_reason = "missing_cache_downloads_disabled"
                break
            if max_new_requests is not None and new_requests >= max_new_requests:
                stop_reason = "request_limit"
                break
            new_requests += 1

        audit = probe_execution_window(item.product_id, item.day, cache_dir, allow_download=allow_download)
        row = dict(audit, reference_available=False, reference_price_usd=np.nan, status="probe_error")

        if audit["outcome"] == "ok":
            row.update(read_execution_reference(item.product_id, item.day, cache_dir, max_age_minutes=max_age_minutes))
        rows.append(row)

        if audit["outcome"] != "ok":
            stop_reason = "probe_error"
            print(f"Stopped: {audit['outcome']}. Inspect {audit['raw_path']}")
            break

        if len(rows) % 50 == 0:
            print(f"Audited {len(rows)}/{len(plan)}, new requests: {new_requests}", flush=True)

    report = pd.DataFrame(rows)
    summary = pd.Series({"planned_windows": len(plan),
        "audited_windows": len(rows),
        "unprocessed_windows": len(plan) - len(rows),
        "new_requests": new_requests,
        "usable_references": sum(r["reference_available"] for r in rows),
        "flagged_windows": sum(not r["reference_available"] for r in rows),
        "stop_reason": stop_reason
    })
    return report, summary

def inspect_extended_gaps(gaps, cache_dir, allow_download=False):
    #inspect one hour windows without treating later prices as 00:05 fills
    pairs = gaps[["product_id", "day"]].copy()
    if pairs.duplicated().any():
        raise ValueError("Duplicate product/day gaps.")

    rows = []

    def load_bars(path, start, end):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        frame = pd.DataFrame(json.loads(record["response_text"]), columns=["time", "low", "high", "open", "close", "volume"]).astype(float)

        return frame.loc[frame["time"].ge(start.timestamp()) & frame["time"].lt(end.timestamp())].sort_values("time").reset_index(drop=True)

    for item in pairs.sort_values(["day", "product_id"]).itertuples(index=False):
        original = probe_execution_window(item.product_id, item.day, cache_dir, allow_download=False)
        if original["outcome"] != "ok":
            raise ValueError(f"Inspect original probe: {original['raw_path']}")

        extended = probe_execution_window(item.product_id, item.day, cache_dir, allow_download=allow_download, window_minutes=60)
        if extended["outcome"] != "ok":
            raise ValueError(f"Inspect extended probe: {extended['raw_path']}")

        day = original["day"]
        old = load_bars(original["raw_path"], day, day + 15 * MINUTE)
        overlap = load_bars(extended["raw_path"], day, day + 15 * MINUTE)
        hour = load_bars(extended["raw_path"], day, day + 60 * MINUTE)

        later = hour.loc[hour["time"].ge((day + 5 * MINUTE).timestamp()) & hour["volume"].gt(0)]
        first = pd.to_datetime(later.iloc[0]["time"], unit="s", utc=True) if not later.empty else pd.NaT

        rows.append({
            "product_id": item.product_id,
            "day": day,
            "original_window_matches": old.equals(overlap),
            "trade_bars_in_hour": int(hour["volume"].gt(0).sum()),
            "first_later_bar_start": first,
            "earliest_later_reference_at": first + MINUTE,
            "status": "later_reference_found" if not later.empty else "no_later_reference_in_hour",
            "raw_path": extended["raw_path"]
        })

        print(f"Inspected {len(rows)}/{len(pairs)}", flush=True)

    return pd.DataFrame(rows)

def find_common_execution_references(request_plan, cache_dir, window_minutes=15, max_age_minutes=5, allow_download=False):
    #find the first common fresh reference time, checking each minute from 00:05. targets stay frozen at midnight. last-trade marks, not fills
    if window_minutes not in (15, 60):
        raise ValueError("Use a 15- or 60-minute observation window.")
    if not np.isfinite(max_age_minutes) or not 1 <= max_age_minutes <= 5:
        raise ValueError("max_age_minutes must be between 1 and 5.")
    plan = request_plan[["product_id", "day"]].copy()
    plan["day"] = pd.to_datetime(plan["day"], utc=True)
    if plan.empty or plan.isna().any().any() or plan.duplicated().any():
        raise ValueError("Provide nonempty, unique product/day pairs.")

    def load_bars(audit, minutes):
        record = json.loads(Path(audit["raw_path"]).read_text(encoding="utf-8"))
        bars = pd.DataFrame(json.loads(record["response_text"]), columns=["time", "low", "high", "open", "close", "volume"]).astype(float)
        start = audit["day"].timestamp()
        return bars.loc[bars["time"].ge(start) & bars["time"].lt(start + 60 * minutes)].sort_values("time").reset_index(drop=True)

    schedules, references = [], []
    for day, group in plan.groupby("day", sort=True):
        histories = {}
        for product_id in sorted(group["product_id"]):
            audit = probe_execution_window(product_id, day, cache_dir, allow_download=allow_download, window_minutes=window_minutes)
            if audit["outcome"] != "ok":
                raise ValueError(f"Inspect {audit['outcome']}: {audit['raw_path']}")
            bars = load_bars(audit, window_minutes)
            if window_minutes == 60:
                original = probe_execution_window(product_id, day, cache_dir)
                if original["outcome"] != "ok":
                    raise ValueError(f"Inspect original: {original['raw_path']}")
                if not load_bars(original, 15).equals(load_bars(audit, 15)):
                    raise ValueError(f"Overlapping captures differ: {product_id}, {day}")
            histories[product_id] = bars.loc[bars["volume"].gt(0)]

        chosen = []
        execution_at = pd.NaT
        for minute in range(5, window_minutes + 1):
            candidate = day + minute * MINUTE
            chosen = []
            for product_id, bars in histories.items():
                valid = bars.loc[(bars["time"] + 60).le(candidate.timestamp()) & bars["time"].ge(candidate.timestamp() - 60 * max_age_minutes)]
                if valid.empty:
                    chosen = []
                    break
                latest = valid.iloc[-1]
                bar_start = pd.to_datetime(latest["time"], unit="s", utc=True)
                chosen.append({
                    "product_id": product_id,
                    "day": day,
                    "execution_at": candidate,
                    "reference_bar_start": bar_start,
                    "assumed_available_at": bar_start + MINUTE,
                    "age_upper_bound_minutes": float((candidate - bar_start) / MINUTE),
                    "reference_price_usd": float(latest["close"])
                })
            if chosen:
                execution_at = candidate
                break

        references.extend(chosen)
        schedules.append({
            "day": day,
            "required_assets": len(histories),
            "execution_at": execution_at,
            "delay_after_0005_minutes": float((execution_at - day) / MINUTE - 5) if pd.notna(execution_at) else np.nan,
            "status": "ready" if chosen else "no_common_time_in_window"
        })

    return pd.DataFrame(schedules), pd.DataFrame(references)