import pandas as pd
import numpy as np

from features import add_momentum_features


def build_bitcoin_filter(daily_prices, decision_index, lookback_days):
    """Bitcoin trend known at each weekly decision."""
    decisions = pd.DatetimeIndex(decision_index, name="decision_at")

    if (decisions.empty or str(decisions.tz) != "UTC" or decisions.hasnans or decisions.has_duplicates
        or not decisions.is_monotonic_increasing or not (decisions == decisions.normalize()).all() or not (decisions.dayofweek == 0).all()):
        raise ValueError("Provide unique, sorted Monday UTC midnights.")

    btc = daily_prices.loc[daily_prices["product_id"].eq("BTC-USD")].copy()
    if btc.empty:
        raise ValueError("BTC-USD history is missing.")

    featured = add_momentum_features(btc, lookback_days=lookback_days)
    audit = (featured.set_index("signal_available_at")[["close", "momentum_reference_close", "momentum_return"]]
        .reindex(decisions).rename(columns={
            "close": "btc_close_usd",
            "momentum_reference_close": "btc_reference_close_usd",
            "momentum_return": "btc_return"}))
    if audit.isna().any().any():
        raise ValueError("Missing BTC candle or insufficient uninterrupted history.")

    audit["btc_lookback_days"] = lookback_days
    audit["risk_on"] = audit["btc_return"].gt(0.0)
    return audit

def apply_bitcoin_filter(targets, bitcoin_audit):
    #keep ordinary approach when BTC positive, else cash
    if targets.empty or targets.columns.has_duplicates or "CASH" not in targets:
        raise ValueError("Provide nonempty targets with unique columns and CASH.")
    if targets.index.has_duplicates or not targets.index.equals(bitcoin_audit.index):
        raise ValueError("Targets and Bitcoin filter must have identical dates.")

    risk_on = bitcoin_audit["risk_on"]
    if not pd.api.types.is_bool_dtype(risk_on) or risk_on.isna().any():
        raise ValueError("risk_on must contain non-null booleans.")

    values = targets.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Targets must be finite, nonnegative and sum to one.")

    filtered = targets.copy()
    filtered.loc[~risk_on, :] = 0.0
    filtered.loc[~risk_on, "CASH"] = 1.0

    return filtered