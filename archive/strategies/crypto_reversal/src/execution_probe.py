from pathlib import Path

import pandas as pd

import intraday


def probe_execution_sample(request_plan, days, protocol, allow_download=False, max_new_requests=60):
    #audit every required asset on chosen dates.
    
    dates = pd.DatetimeIndex(pd.to_datetime(days, utc=True))
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    holdout = pd.Timestamp(protocol["holdout_start"], tz="UTC")
    if not start < end <= holdout:
        raise ValueError("Invalid development boundaries.")
    if (dates.empty or dates.hasnans or dates.has_duplicates or not (dates == dates.normalize()).all() or not ((dates >= start) & (dates < end)).all()):
        raise ValueError("Choose unique UTC-midnight development dates.")
    if (isinstance(max_new_requests, bool) or not isinstance(max_new_requests, int) or max_new_requests < 0):
        raise ValueError("max_new_requests must be a non-negative integer.")
    if not isinstance(allow_download, bool):
        raise ValueError("allow_download must be a boolean.")

    selected = request_plan.loc[request_plan["day"].isin(dates)].copy()
    if selected.duplicated(["product_id", "day"]).any():
        raise ValueError("Duplicate asset/date requests.")
    if not dates.difference(pd.DatetimeIndex(selected["day"].unique())).empty:
        raise ValueError("A chosen date has no planned price requests.")

    #validate paths before making requests
    locations = []
    for row in selected.sort_values(["day", "product_id"]).itertuples(index=False):
        filename = f"{row.product_id}_{row.day:%Y%m%d}_0000_0015_60s.json"
        destination = Path(row.download_path)
        cached = Path(row.cache_path) if pd.notna(row.cache_path) else None
        if destination.name != filename or (cached is not None and cached.name != filename):
            raise ValueError("Cache filename does not match the asset/date.")
        if row.earliest_execution_at != row.day + pd.Timedelta(minutes=5):
            raise ValueError("This probe expects an earliest execution time of 00:05.")
        
        locations.append((row, destination, cached))

    rows = []
    new_requests = 0
    for row, destination, cached in locations:
        path = next((p for p in (destination, cached) if p is not None and p.is_file()), None)
        new_request = path is None
        if new_request and (not allow_download or new_requests >= max_new_requests):
            rows.append({
                "product_id": row.product_id, "day": row.day,
                "outcome": "not_captured", "status": "missing_cache",
                "reference_available": False, "new_request": False,
                "raw_path": str(destination)
            })
            continue
        if new_request:
            path = destination
            new_requests += 1

        audit = intraday.probe_execution_window(row.product_id, row.day, path.parent, allow_download=bool(new_request and allow_download))
        result = dict(audit, new_request=new_request, reference_available=False, status=audit["outcome"])
        if audit["outcome"] == "ok":
            result.update(intraday.read_execution_reference(row.product_id, row.day, path.parent, max_age_minutes=5))
        rows.append(result)
        if audit["outcome"] != "ok":
            #preserve the response and stop on API/data errors
            break

    report = pd.DataFrame(rows)
    summary = pd.Series({
        "planned_windows": len(selected),
        "reported_windows": len(report),
        "unprocessed_windows": len(selected) - len(report),
        "new_requests": new_requests,
        "missing_cached_windows": int(report["outcome"].eq("not_captured").sum()),
        "usable_references": int(report["reference_available"].sum()),
        "flagged_captured_windows": int((report["outcome"].ne("not_captured") & ~report["reference_available"]).sum())
    })
    return report, summary
