#development parameter grid
from itertools import product

import numpy as np
import pandas as pd


LOOKBACK_DAYS = (1, 5, 10, 30, 90)
TOP_FRACTIONS = (0.10, 0.20, 0.30)
REBALANCE_INTERVALS = (1, 2, 3, 7)


def build_development_grid():
    #every combination included. stable identifier for later audits

    rows = []
    for lookback, fraction, interval in product(LOOKBACK_DAYS, TOP_FRACTIONS, REBALANCE_INTERVALS):
        rows.append({
            "config_id": f"return_{lookback}d_bottom_{round(100 * fraction)}pct_rebalance_{interval}d",
            "lookback_days": lookback,
            "top_fraction": fraction,
            "rebalance_days": interval
        })
    return pd.DataFrame(rows).set_index("config_id")


def build_development_schedules(universe_coverage, protocol, intervals=REBALANCE_INTERVALS):
    #all intervals share first development date as their fixed anchor

    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    holdout_start = pd.Timestamp(protocol["holdout_start"], tz="UTC")
    if not start < end <= holdout_start:
        raise ValueError("Development must end on or before the holdout starts.")
    if any(t != t.normalize() for t in (start, end, holdout_start)):
        raise ValueError("Research boundaries must be UTC midnight.")
    intervals = tuple(intervals)
    if (not intervals or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in intervals) or len(set(intervals)) != len(intervals)):
        raise ValueError("Intervals must be distinct positive integer day counts.")

    dates = universe_coverage.index
    expected = pd.date_range(start, end, freq="D", inclusive="left", name="decision_at")
    if (not isinstance(dates, pd.DatetimeIndex) or str(dates.tz) != "UTC" or not dates.equals(expected)):
        raise ValueError("Coverage must contain every development day in order, including inactive days.")
    if not {"active", "universe_size"}.issubset(universe_coverage.columns):
        raise ValueError("Coverage needs active and universe_size columns.")
    
    active = universe_coverage["active"]
    sizes = universe_coverage["universe_size"]
    if not pd.api.types.is_bool_dtype(active) or active.isna().any():
        raise ValueError("active must contain non-null booleans.")
    if (not pd.api.types.is_integer_dtype(sizes) or sizes.isna().any() or sizes.lt(0).any()):
        raise ValueError("universe_size must contain non-negative integers.")
    
    minimum = protocol["universe"]["min_assets"]
    maximum = protocol["universe"]["max_assets"]
    if (not sizes.loc[active].between(minimum, maximum).all() or not sizes.loc[~active].eq(0).all()):
        raise ValueError("Universe size disagrees with the activity rules.")

    frames = []
    for interval in intervals:
        decision_dates = pd.date_range(start, end, freq=f"{interval}D", inclusive="left")
        frame = universe_coverage.loc[decision_dates, ["active", "universe_size"]].copy()
        frame.index.name = "decision_at"
        frame = frame.reset_index()
        frame.insert(0, "rebalance_days", interval)
        frame["rebalance_number"] = np.arange(len(frame))
        frame["anchor_at"] = start
        frame["signal_candle_start"] = frame["decision_at"] - pd.Timedelta(days=1)
        #keep insufficient-breadth dates.
        frame["status"] = np.where(frame["active"], "universe_ready", "insufficient_breadth")
        frames.append(frame)

    schedules = pd.concat(frames, ignore_index=True)
    audit = schedules.groupby("rebalance_days", sort=True).agg(scheduled_decisions=("decision_at", "size"),
        active_decisions=("active", "sum"), first_decision=("decision_at", "min"), last_decision=("decision_at", "max"))
    audit["inactive_decisions"] = audit["scheduled_decisions"] - audit["active_decisions"]
    return schedules, audit[["scheduled_decisions", "active_decisions", "inactive_decisions", "first_decision", "last_decision"]]
