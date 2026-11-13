
#crypto trend builder, without registry side effects

import numpy as np
import pandas as pd
from foundation._research.benchmark import eligible_universe
from foundation._research.contracts import normalise_targets
from foundation._research.panel import daily_returns, pivot_closes, realised_volatility, trailing_return

def volatility_scaled_trend(panel, schedule, lookback_days=30, vol_window_days=30, target_volatility=0.30):
    #same rule, but each position is scaled by its realised volatility

    if not np.isfinite(target_volatility) or target_volatility <= 0:
        raise ValueError("target_volatility must be positive and finite.")

    closes = pivot_closes(panel)
    returns = daily_returns(closes)
    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        if not members:
            continue

        trailing = trailing_return(closes, stamp, lookback_days).reindex(members)
        volatility = realised_volatility(returns, stamp, vol_window_days).reindex(members)
        usable = trailing.gt(0.0) & volatility.gt(0.0)
        usable = usable & np.isfinite(trailing.to_numpy(dtype=float)) & np.isfinite(volatility.to_numpy(dtype=float))
        if not usable.any():
            continue

        scaled = (target_volatility / volatility[usable]).clip(upper=1.0) / len(members)
        weights.loc[stamp, list(scaled.index)] = scaled.to_numpy(dtype=float)

    return normalise_targets(weights, schedule, assets)


def trend_agreement(panel, schedule, lookbacks=(7, 14, 30, 60, 90)):
    #exposure equals the fraction of lookbacks that agree

    windows = tuple(lookbacks)
    if not windows:
        raise ValueError("Supply at least one lookback.")
    for window in windows:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("Every lookback must be a positive integer.")

    closes = pivot_closes(panel)
    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        if not members:
            continue

        votes = pd.Series(0.0, index=members, dtype=float)
        counted = pd.Series(0.0, index=members, dtype=float)
        for window in windows:
            trailing = trailing_return(closes, stamp, window).reindex(members)
            observed = pd.Series(np.isfinite(trailing.to_numpy(dtype=float)), index=members)
            votes = votes.add(trailing.gt(0.0).where(observed, False).astype(float), fill_value=0.0)
            counted = counted.add(observed.astype(float), fill_value=0.0)

        agreement = (votes / counted.replace(0.0, np.nan)).fillna(0.0)
        weights.loc[stamp, members] = (agreement / len(members)).to_numpy(dtype=float)

    return normalise_targets(weights, schedule, assets)


def trend_agreement_vol(panel, schedule, lookbacks=(7, 14, 30, 60, 90, 180), vol_window_days=30, target_volatility=0.50, no_trade_band=0.0):
    #each assets slot is scaled by the fraction of lookbacks that agree, then by inverse volatility. changes inside the band are not traded

    windows = tuple(lookbacks)
    if not windows:
        raise ValueError("Supply at least one lookback.")
    for window in windows:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("Every lookback must be a positive integer.")
    if not np.isfinite(target_volatility) or target_volatility <= 0:
        raise ValueError("target_volatility must be positive and finite.")
    if not np.isfinite(no_trade_band) or not 0 <= no_trade_band < 1:
        raise ValueError("no_trade_band must lie in [0, 1).")

    closes = pivot_closes(panel)
    returns = daily_returns(closes)
    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)
    previous = pd.Series(0.0, index=assets, dtype=float)

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        desired = pd.Series(0.0, index=assets, dtype=float)

        if members:
            votes = pd.Series(0.0, index=members, dtype=float)
            counted = pd.Series(0.0, index=members, dtype=float)
            for window in windows:
                trailing = trailing_return(closes, stamp, window).reindex(members)
                observed = pd.Series(np.isfinite(trailing.to_numpy(dtype=float)), index=members)
                votes = votes.add(trailing.gt(0.0).where(observed, False).astype(float), fill_value=0.0)
                counted = counted.add(observed.astype(float), fill_value=0.0)
            agreement = (votes / counted.replace(0.0, np.nan)).fillna(0.0)

            #if no volatility estimate, then no position
            volatility = realised_volatility(returns, stamp, vol_window_days).reindex(members)
            scale = (target_volatility / volatility).clip(upper=1.0)
            scale = scale.where(np.isfinite(scale.to_numpy(dtype=float)) & volatility.gt(0.0), 0.0)

            desired.loc[members] = (agreement * scale / len(members)).to_numpy(dtype=float)

        #if weight change within band, then don't change the weight we currently have
        keep = (desired - previous).abs().lt(no_trade_band) & desired.gt(0.0)
        chosen = desired.where(~keep, previous)
        if chosen.sum() > 1.0:
            chosen = desired

        weights.loc[stamp] = chosen.to_numpy(dtype=float)
        previous = chosen

    return normalise_targets(weights, schedule, assets)


#offline readers needed by crypto system
from foundation.data.crypto_reference import  load_daily_history, load_monday_minutes, minute_fills #noqa: E402,F401
