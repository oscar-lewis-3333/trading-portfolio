import numpy as np
import pandas as pd


def build_execution_schedule(targets, gbp_prices):
    #build monday targets for Tuesday, preserving out-of-range decisions
    decisions = targets.index

    if targets.empty:
        raise ValueError("No targets supplied.")
    if not isinstance(decisions, pd.DatetimeIndex):
        raise ValueError("Targets must have a DatetimeIndex.")
    if decisions.tz is None or str(decisions.tz) != "UTC":
        raise ValueError("Decision times must use UTC.")
    if decisions.hasnans or decisions.has_duplicates:
        raise ValueError("Decision times must be valid and unique.")
    if not decisions.is_monotonic_increasing:
        raise ValueError("Decision times must be sorted.")
    if not ((decisions.dayofweek == 0) & (decisions == decisions.normalize())).all():
        raise ValueError("Expected Monday decisions at UTC midnight.")

    if targets.columns.has_duplicates or "CASH" not in targets.columns:
        raise ValueError("Expected unique asset columns and a CASH column.")

    weights = targets.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and non-negative.")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Each target portfolio must sum to one.")

    candle_times = gbp_prices["timestamp"]
    if candle_times.empty or candle_times.isna().any():
        raise ValueError("Missing price-history timestamps.")
    if (not isinstance(candle_times.dtype, pd.DatetimeTZDtype) or str(candle_times.dt.tz) != "UTC"):
        raise ValueError("Price timestamps must use UTC.")
    if candle_times.ne(candle_times.dt.normalize()).any():
        raise ValueError("Expected daily candles at UTC midnight.")

    schedule = pd.DataFrame({
        "decision_at": decisions,
        "execution_at": decisions + pd.Timedelta(days=1)
    })

    first_candle = candle_times.min()
    last_candle = candle_times.max()

    schedule["status"] = "within_history"
    schedule.loc[schedule["execution_at"].lt(first_candle), "status"] = "before_history"
    schedule.loc[schedule["execution_at"].gt(last_candle), "status"] = "after_history"

    #change time index, never the weights
    planned_targets = targets.copy()
    planned_targets.index = pd.DatetimeIndex(schedule["execution_at"], name="execution_at")

    return planned_targets, schedule

