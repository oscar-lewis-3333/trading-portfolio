import re
from pathlib import Path

import numpy as np
import pandas as pd


def build_execution_request_plan(sweep_targets, development_grid, protocol, cache_dir, reuse_cache_dirs=()):
    #union held/targeted assets across all strategies and matched benchmarks

    if (development_grid.empty or not development_grid.index.is_unique or development_grid.index.hasnans or set(sweep_targets) != set(development_grid.index)):
        raise ValueError("Supply targets for every uniquely identified configuration.")
    
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    if not start < end <= pd.Timestamp(protocol["holdout_start"], tz="UTC"):
        raise ValueError("Invalid development/holdout boundaries.")
    if start != start.normalize() or end != end.normalize():
        raise ValueError("Development boundaries must be UTC midnight.")
    
    destination = Path(cache_dir).expanduser().resolve()
    locations = [destination] + [Path(p).expanduser().resolve() for p in reuse_cache_dirs]
    requirements = {}
    all_decisions = set()

    for config_id, config in development_grid.iterrows():
        interval = config["rebalance_days"]
        if (isinstance(interval, (bool, np.bool_)) or not np.isfinite(interval) or interval < 1 or int(interval) != interval):
            raise ValueError("Rebalance intervals must be positive integer day counts.")
        expected = pd.date_range(start, end, freq=f"{int(interval)}D", inclusive="left")
        pair = sweep_targets[config_id]

        if not {"strategy", "benchmark"}.issubset(pair):
            raise ValueError(f"Missing strategy or benchmark: {config_id}")
        for role in ("strategy", "benchmark"):
            weights = pair[role]
            if (not isinstance(weights, pd.DataFrame) or not weights.index.equals(expected) or weights.columns.has_duplicates or "CASH" not in weights):
                raise ValueError(f"Invalid development target calendar or columns: {config_id}, {role}")
            
            values = weights.to_numpy(dtype=float)
            if (not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-12)):
                raise ValueError("Target weights must be non-negative and sum to one.")
            
            assets = weights.drop(columns="CASH")
            if not all(isinstance(p, str) and re.fullmatch(r"[A-Z0-9]+-USD", p) for p in assets.columns):
                raise ValueError("Expected Coinbase USD product identifiers.")

            previous = set()
            for day, positive in assets.gt(0).iterrows():
                all_decisions.add(day)
                current = set(positive.index[positive])
                for asset in current | previous:
                    flags = requirements.setdefault((day, asset), [False, False])
                    flags[0] = flags[0] or asset in current
                    flags[1] = flags[1] or asset in previous
                previous = current

    rows = []
    for (day, asset), (current, previous) in sorted(requirements.items()):
        filename = f"{asset}_{day:%Y%m%d}_0000_0015_60s.json"
        cached = next((p / filename for p in locations if (p / filename).is_file()), None)
        rows.append({
            "product_id": asset,
            "day": day,
            "earliest_execution_at": day + pd.Timedelta(minutes=5),
            "current_member": current,
            "previous_member": previous,
            "exit_only": previous and not current,
            "cache_exists": cached is not None,
            "cache_path": str(cached) if cached is not None else None,
            "download_path": str(destination / filename)
        })

    columns = ["product_id", "day", "earliest_execution_at", "current_member",
            "previous_member", "exit_only", "cache_exists", "cache_path", "download_path"]
    plan = pd.DataFrame(rows, columns=columns)
    for column in ["day", "earliest_execution_at"]:
        plan[column] = pd.to_datetime(plan[column], utc=True)
    for column in ["current_member", "previous_member", "exit_only", "cache_exists"]:
        plan[column] = plan[column].astype(bool)

    summary = pd.Series({
        "configurations": len(development_grid),
        "decision_dates": len(all_decisions),
        "dates_requiring_prices": plan["day"].nunique(),
        "unique_product_day_windows": len(plan),
        "cached_windows": int(plan["cache_exists"].sum()),
        "missing_windows": int((~plan["cache_exists"]).sum()),
        "exit_only_windows": int(plan["exit_only"].sum())
    })
    return plan, summary
