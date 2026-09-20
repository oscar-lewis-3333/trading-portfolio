import numpy as np

from preparation import add_history_segments

def add_momentum_features(daily_panel, lookback_days=30):
    #calculate close to close momentum.
    if (not isinstance(lookback_days, int) or isinstance(lookback_days, bool) or lookback_days < 1):
        raise ValueError("lookback_days must be a positive integer.")

    required = {"product_id", "timestamp", "close"}
    missing = required.difference(daily_panel.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    #recompute segments and signal timing while preserving eligibility flags
    frame = add_history_segments(daily_panel)
    closes = frame["close"].to_numpy(dtype=float)
    if not np.isfinite(closes).all() or (closes <= 0).any():
        raise ValueError("Close prices must be finite and positive.")

    frame["momentum_reference_close"] = frame.groupby(["product_id", "segment_id"])["close"].shift(lookback_days)
    frame["momentum_return"] = (frame["close"] / frame["momentum_reference_close"] - 1.0)
    frame["momentum_ready"] = frame["momentum_reference_close"].notna()
    frame["momentum_lookback_days"] = lookback_days
    available = frame.loc[frame["momentum_ready"], "momentum_return"].to_numpy(dtype=float)

    if not np.isfinite(available).all():
        raise ValueError("Non-finite momentum calculations.")

    return frame