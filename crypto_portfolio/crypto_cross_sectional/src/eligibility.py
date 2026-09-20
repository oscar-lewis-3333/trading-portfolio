import numpy as np

from preparation import add_history_segments


def add_research_eligibility(prices, history_days=180, liquidity_days=30, min_median_dollar_volume=1_000_000.0):
    #add history and liquidity screens using completed daily candles

    for name, value in (("history_days", history_days), ("liquidity_days", liquidity_days)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if (not np.isfinite(min_median_dollar_volume) or min_median_dollar_volume <= 0):
        raise ValueError("The dollar-volume threshold must be positive.")

    #recompute continuity so supplied segment labels can't be stale
    frame = add_history_segments(prices)
    if not frame["product_id"].str.endswith("-USD").all():
        raise ValueError("This liquidity measure expects USD products.")

    values = frame[["close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Resolve non-finite close/volume observations.")
    if frame["close"].le(0).any() or frame["volume"].lt(0).any():
        raise ValueError("Invalid close prices or volumes.")

    frame["dollar_volume_proxy"] = frame["close"] * frame["volume"]
    if not np.isfinite(frame["dollar_volume_proxy"]).all():
        raise ValueError("Non-finite dollar-volume calculations.")

    frame["median_dollar_volume"] = (frame.groupby(["product_id", "segment_id"])["dollar_volume_proxy"]
        .transform(lambda values: values.rolling(liquidity_days, min_periods=liquidity_days).median()))

    frame["history_ready"] = (frame["contiguous_observations"].ge(history_days + 1))
    frame["liquidity_ready"] = (frame["median_dollar_volume"].ge(min_median_dollar_volume))
    frame["research_candidate"] = (frame["history_ready"] & frame["liquidity_ready"])

    return frame