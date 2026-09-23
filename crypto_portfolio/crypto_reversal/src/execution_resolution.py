#resolve development execution marks and save a verified, offline data bundle.
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

import intraday


POLICY = {
    "version": "development_execution_v1",
    "earliest_minute": 5,
    "latest_minute": 60,
    "max_age_minutes_from_bar_start": 5,
    "reference": "Latest completed positive-volume minute candle close",
    "availability_assumption": "Minute candles available at their end; feed latency unmeasured",
    "common_time": "First qualifying minute for the union of required assets across all configurations and benchmarks",
    "targets": "Fixed at midnight; do not rerank while waiting",
    "missing_reference": "Stop for investigation; never drop assets or fabricate prices",
    "capture_overlap": "One-hour captures must agree exactly with saved first-15-minute candles",
    "limitation": "Trade marks are not executable quotes or evidence of Revolut X fills"
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _validate_plan(plan, protocol):
    plan = plan.copy()
    plan["day"] = pd.to_datetime(plan["day"], utc=True)
    plan["earliest_execution_at"] = pd.to_datetime(plan["earliest_execution_at"], utc=True)
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    holdout = pd.Timestamp(protocol["holdout_start"], tz="UTC")

    if not start < end <= holdout:
        raise ValueError("Invalid development boundaries.")
    if (plan.empty or plan[["day", "product_id"]].isna().any().any() or plan.duplicated(["day", "product_id"]).any() or not plan["day"].between(start, end, inclusive="left").all()
            or not plan["day"].eq(plan["day"].dt.normalize()).all() or not plan["earliest_execution_at"].eq(plan["day"] + pd.Timedelta(minutes=5)).all()
            or not plan["product_id"].str.fullmatch(r"[A-Z0-9]+-USD").all()):
        raise ValueError("Expected unique development asset/day requests at UTC midnight.")
    
    return plan.sort_values(["day", "product_id"]).reset_index(drop=True)


def _plan_fingerprint(plan):
    return _fingerprint([(r.product_id, r.day.isoformat()) for r in plan.itertuples(index=False)])


def _read_window(path, product, day, minutes, manifest):
    #validate the response and return complete, in-window OHLCV bars
    path = Path(path).resolve()
    raw = path.read_bytes()
    record = json.loads(raw)
    expected = {"product_id": product,
        "url": f"https://api.exchange.coinbase.com/products/{product}/candles",
        "params": {"granularity": 60, "start": day.isoformat(), "end": (day + pd.Timedelta(minutes=minutes)).isoformat()}}
    if (record.get("source") != "coinbase_exchange" or record.get("request") != expected or record.get("http_status") != 200 or record.get("error") is not None):
        raise ValueError(f"Invalid cached response or metadata: {path}")
    
    payload = json.loads(record["response_text"])
    if not isinstance(payload, list):
        raise ValueError(f"Expected candle list: {path}")
    
    start = int(day.timestamp())
    bars = {}
    for row in payload:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError(f"Invalid candle shape: {path}")
        if any(isinstance(x, bool) for x in row):
            raise ValueError(f"Boolean candle value: {path}")
        
        values = tuple(float(x) for x in row)
        second, low, high, opening, close, volume = values
        if not math.isfinite(second):
            raise ValueError(f"Invalid timestamp: {path}")
        if not start <= second < start + minutes * 60:
            continue  #API boundary/extra rows are not part of this window
        if (not all(math.isfinite(x) for x in values) or second % 60 or second in bars or min(low, high, opening, close) <= 0
                or volume < 0 or not low <= opening <= high or not low <= close <= high):
            raise ValueError(f"Invalid or duplicate candle: {path}")
        
        bars[int(second)] = values
    manifest[str(path)] = hashlib.sha256(raw).hexdigest()
    return dict(sorted(bars.items()))


def _latest(bars, at):
    valid = [t for t, row in bars.items() if at - 300 <= t <= at - 60 and row[5] > 0]
    return max(valid) if valid else None


def _common(histories, day, limit):
    for minute in range(5, limit + 1):
        at = int(day.timestamp()) + minute * 60
        chosen = {p: _latest(bars, at) for p, bars in histories.items()}
        if all(t is not None for t in chosen.values()):
            return at, chosen
    return None, {}


def build_execution_data(request_plan, protocol, cache_dir, reuse_cache_dirs=(), allow_download=False, max_new_requests=0):
    #use saved 15 minute windows first
    plan = _validate_plan(request_plan, protocol)
    if (not isinstance(allow_download, bool) or isinstance(max_new_requests, bool) or not isinstance(max_new_requests, int) or max_new_requests < 0):
        raise ValueError("Provide a boolean download flag and non-negative integer budget.")
    
    locations = [Path(cache_dir).resolve()] + [Path(p).resolve() for p in reuse_cache_dirs]
    references, schedules, audits, manifest = [], [], [], {}
    new_requests = 0
    for number, (day, group) in enumerate(plan.groupby("day", sort=True), 1):
        histories, paths = {}, {}
        for row in group.itertuples(index=False):
            expected = f"{row.product_id}_{day:%Y%m%d}_0000_0015_60s.json"
            candidates = [Path(row.download_path)]
            if pd.notna(row.cache_path):
                candidates.append(Path(row.cache_path))
            if any(p.name != expected for p in candidates):
                raise ValueError("Original cache path does not match asset/day.")
            
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Original capture missing: {expected}")
            
            histories[row.product_id] = _read_window(path, row.product_id, day, 15, manifest)
            paths[row.product_id] = str(path.resolve())
            available = _latest(histories[row.product_id], int(day.timestamp()) + 300) is not None
            audits.append({"product_id": row.product_id, "day": day, "original_reference_available": available,
                           "original_status": "ok" if available else "no_completed_trade_bar", "raw_path": str(path.resolve())})

        at, chosen = _common(histories, day, 15)
        used_hour = at is None
        if used_hour:
            print(f"Extending {day.date()}: {len(group)} required assets", flush=True)
            for product in histories:
                filename = f"{product}_{day:%Y%m%d}_0000_0100_60s.json"
                path = next((p / filename for p in locations if (p / filename).is_file()), None)
                if path is None:
                    if not allow_download or new_requests >= max_new_requests:
                        raise FileNotFoundError(f"Missing hour capture or exhausted request budget: {filename}")
                    
                    new_requests += 1
                    audit = intraday.probe_execution_window(
                        product, day, locations[0], allow_download=True, window_minutes=60)
                    if audit["outcome"] != "ok":
                        raise ValueError(f"Inspect extended response: {audit['raw_path']}")
                    
                    path = Path(audit["raw_path"])
                extended = _read_window(path, product, day, 60, manifest)
                overlap = {t: v for t, v in extended.items() if t < int(day.timestamp()) + 900}
                if overlap != histories[product]:
                    raise ValueError(f"Overlapping captures differ: {product}, {day}")
                
                histories[product], paths[product] = extended, str(path.resolve())
            at, chosen = _common(histories, day, 60)
        if at is None:
            raise ValueError(f"No common fresh reference by 01:00: {day}; investigate, do not drop.")
        
        execution_at = pd.to_datetime(at, unit="s", utc=True)
        schedules.append({"day": day, "required_assets": len(group), "execution_at": execution_at,
                          "delay_after_0005_minutes": (at - int(day.timestamp())) // 60 - 5, "used_hour_window": used_hour, "status": "ready"})
        for product, second in chosen.items():
            references.append({
                "product_id": product, "day": day, "execution_at": execution_at,
                "reference_bar_start": pd.to_datetime(second, unit="s", utc=True),
                "assumed_available_at": pd.to_datetime(second + 60, unit="s", utc=True),
                "age_upper_bound_minutes": (at - second) / 60,
                "reference_price_usd": histories[product][second][4], "raw_path": paths[product]
            })
        if number % 100 == 0:
            print(f"Resolved {number} dates; new hour requests: {new_requests}", flush=True)

    references, schedule, audit = map(pd.DataFrame, (references, schedules, audits))
    _validate_result(plan, references, schedule, audit)
    summary = _summary(references, schedule, audit)
    summary["new_hour_requests"] = new_requests
    return {"references": references, "schedule": schedule, "audit": audit, "summary": summary, "raw_sha256": manifest}


def _validate_result(plan, references, schedule, audit):
    expected = set(zip(plan["product_id"], plan["day"]))
    for frame in (references, audit):
        if frame.duplicated(["product_id", "day"]).any() or set(zip(frame["product_id"], frame["day"])) != expected:
            raise ValueError("Execution data do not cover exactly the planned asset/dates.")
    if schedule["day"].duplicated().any() or set(schedule["day"]) != set(plan["day"]):
        raise ValueError("Execution schedule does not cover the planned dates.")
    r = references
    ages = (r["execution_at"] - r["reference_bar_start"]) / pd.Timedelta(minutes=1)
    delays = (r["execution_at"] - r["day"]) / pd.Timedelta(minutes=1)
    if (not ages.between(1, 5).all() or not ages.eq(r["age_upper_bound_minutes"]).all()
            or not delays.between(5, 60).all()
            or not r["assumed_available_at"].eq(r["reference_bar_start"] + pd.Timedelta(minutes=1)).all()
            or not r["assumed_available_at"].le(r["execution_at"]).all()
            or not r["reference_price_usd"].map(lambda x: math.isfinite(x) and x > 0).all()
            or not r.groupby("day")["execution_at"].nunique().eq(1).all()):
        raise ValueError("Invalid reference timing, age or price.")
    joined = schedule.set_index("day").join(r.groupby("day").agg(
        actual_assets=("product_id", "size"), actual_at=("execution_at", "first")))
    if (not joined["required_assets"].eq(joined["actual_assets"]).all()
            or not joined["execution_at"].eq(joined["actual_at"]).all()
            or not joined["status"].eq("ready").all()):
        raise ValueError("Schedule and references disagree.")


def _summary(references, schedule, audit):
    return pd.Series({
        "reference_rows": len(references), "execution_dates": len(schedule),
        "original_missing_references": int((~audit["original_reference_available"]).sum()),
        "ordinary_0005_dates": int(schedule["delay_after_0005_minutes"].eq(0).sum()),
        "delayed_dates": int(schedule["delay_after_0005_minutes"].gt(0).sum()),
        "hour_window_dates": int(schedule["used_hour_window"].sum()),
        "max_delay_after_0005_minutes": int(schedule["delay_after_0005_minutes"].max()),
        "unresolved_references": 0
    })


def save_execution_data(bundle, request_plan, protocol, directory):
    #create new version
    plan = _validate_plan(request_plan, protocol)
    _validate_result(plan, bundle["references"], bundle["schedule"], bundle["audit"])
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Preserve the existing bundle: {directory}")
    
    directory.mkdir(parents=True)
    exports = {}
    for name in ("references", "schedule", "audit"):
        path = directory / f"{name}.csv"
        bundle[name].to_csv(path, index=False)
        exports[path.name] = _sha(path)
    metadata = {"policy": POLICY, "protocol_sha256": _fingerprint(protocol),
                "plan_sha256": _plan_fingerprint(plan), "exports_sha256": exports,
                "raw_sha256": bundle["raw_sha256"],
                "capture_summary": {k: int(v) for k, v in bundle["summary"].items()}}
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_execution_data(request_plan, protocol, directory):
    #offline replay
    plan = _validate_plan(request_plan, protocol)
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if (metadata["policy"] != POLICY or metadata["protocol_sha256"] != _fingerprint(protocol) or metadata["plan_sha256"] != _plan_fingerprint(plan)):
        raise ValueError("Saved execution data do not match the current plan/protocol/policy.")
    
    for filename, checksum in metadata["exports_sha256"].items():
        if _sha(directory / filename) != checksum:
            raise ValueError(f"Saved output changed: {filename}")
        
    for filename, checksum in metadata["raw_sha256"].items():
        if _sha(filename) != checksum:
            raise ValueError(f"Saved raw capture changed: {filename}")
        
    bundle = {}
    for name in ("references", "schedule", "audit"):
        frame = pd.read_csv(directory / f"{name}.csv")
        for column in ("day", "execution_at", "reference_bar_start", "assumed_available_at"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        bundle[name] = frame
    _validate_result(plan, bundle["references"], bundle["schedule"], bundle["audit"])
    bundle["summary"] = _summary(bundle["references"], bundle["schedule"], bundle["audit"])
    return bundle
