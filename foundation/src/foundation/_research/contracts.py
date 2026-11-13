import numpy as np
import pandas as pd

PANEL_COLUMNS = ("product_id", "timestamp", "close", "eligible")
CASH = "CASH"
WEIGHT_TOLERANCE = 1e-9


def validate_panel(panel):
    #one row per asset per completed day
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("The panel must be a DataFrame.")
    missing = set(PANEL_COLUMNS).difference(panel.columns)
    if missing:
        raise ValueError(f"Missing panel columns: {sorted(missing)}")
    if panel.empty:
        raise ValueError("The panel is empty.")

    times = panel["timestamp"]
    if not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != "UTC":
        raise ValueError("Panel timestamps must be UTC timezone-aware.")
    if times.isna().any() or times.ne(times.dt.normalize()).any():
        raise ValueError("Panel timestamps must be valid UTC midnights.")

    if panel["product_id"].isna().any():
        raise ValueError("Missing product IDs.")
    if panel["product_id"].eq(CASH).any():
        raise ValueError(f"{CASH} is reserved and cannot be a product ID.")
    if panel.duplicated(["product_id", "timestamp"]).any():
        raise ValueError("Duplicate product/timestamp observations.")

    closes = panel["close"].to_numpy(dtype=float)
    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("Closes must be finite and positive.")

    eligible = panel["eligible"]
    if not pd.api.types.is_bool_dtype(eligible) or eligible.isna().any():
        raise ValueError("eligible must contain non-null booleans.")

    return panel


def validate_schedule(schedule):
    #decisions timings for each signal

    if not isinstance(schedule, pd.DatetimeIndex):
        raise ValueError("The schedule must be a DatetimeIndex.")
    if schedule.empty:
        raise ValueError("The schedule is empty.")
    if schedule.tz is None or str(schedule.tz) != "UTC":
        raise ValueError("Decision stamps must use UTC.")
    if schedule.hasnans or schedule.has_duplicates:
        raise ValueError("Decision stamps must be valid and unique.")
    if not schedule.is_monotonic_increasing:
        raise ValueError("Decision stamps must be sorted.")
    if (schedule != schedule.normalize()).any():
        raise ValueError("Decision stamps must be UTC midnights.")

    return schedule


def validate_targets(targets, schedule=None):
    #the things every signal must satisfy
    if not isinstance(targets, pd.DataFrame):
        raise ValueError("Targets must be a DataFrame.")
    if not isinstance(targets.index, pd.DatetimeIndex):
        raise ValueError("Targets must be indexed by decision stamp.")

    validate_schedule(targets.index)

    if schedule is not None and not targets.index.equals(validate_schedule(schedule)):
        raise ValueError("Targets must cover exactly the supplied schedule.")
    if targets.columns.has_duplicates:
        raise ValueError("Duplicate asset columns.")
    if CASH not in targets.columns:
        raise ValueError(f"Targets must include a {CASH} column.")

    weights = targets.to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("Weights must be finite.")
    if (weights < 0).any():
        raise ValueError("Weights must be non-negative; this venue is long-only.")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=WEIGHT_TOLERANCE):
        raise ValueError("Each target row must sum to one.")

    return targets


def all_cash(schedule, assets):
    #position when a signal doesn't act

    validate_schedule(schedule)
    columns = [a for a in sorted(set(assets)) if a != CASH]
    targets = pd.DataFrame(0.0, index=schedule, columns=columns)
    targets[CASH] = 1.0

    return validate_targets(targets, schedule)


def normalise_targets(weights, schedule, assets):
    #close cash residual and check result
    
    validate_schedule(schedule)
    columns = [a for a in sorted(set(assets)) if a != CASH]
    frame = pd.DataFrame(weights, index=schedule, columns=columns).astype(float)
    frame = frame.reindex(index=schedule, columns=columns).fillna(0.0)

    invested = frame.to_numpy(dtype=float)
    if not np.isfinite(invested).all() or (invested < 0).any():
        raise ValueError("Signal weights must be finite and non-negative.")

    totals = invested.sum(axis=1)
    if (totals > 1.0 + WEIGHT_TOLERANCE).any():
        raise ValueError("Signal weights cannot exceed one; leverage is unavailable.")

    frame[CASH] = np.clip(1.0 - totals, 0.0, 1.0)

    return validate_targets(frame, schedule)
