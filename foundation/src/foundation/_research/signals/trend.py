import numpy as np
import pandas as pd

from ..benchmark import eligible_universe
from ..contracts import normalise_targets
from ..panel import daily_returns, pivot_closes, realised_volatility, trailing_return
from ..registry import register

#a variety of trend following signals, which are all long-only, and generally combined with other signals.

@register("trend_absolute_slow", {"lookback_days": 90})
@register("trend_absolute_fast", {"lookback_days": 14})
@register("trend_absolute_30d", {"lookback_days": 30})
def absolute_momentum(panel, schedule, lookback_days=30):

    #hold asset only when its return over lookback_days is positive, else cash

    closes = pivot_closes(panel)
    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        if not members:
            continue

        trailing = trailing_return(closes, stamp, lookback_days).reindex(members)
        positive = trailing.gt(0.0) & np.isfinite(trailing.to_numpy(dtype=float))
        if not positive.any():
            continue

        #if one asset has negative trend, the others do not receive a larger share, instead cash is held in place
        weights.loc[stamp, list(positive.index[positive])] = 1.0 / len(members)

    return normalise_targets(weights, schedule, assets)


@register("trend_vol_scaled", {"lookback_days": 30, "vol_window_days": 30, "target_volatility": 0.30})
def volatility_scaled_trend(panel, schedule, lookback_days=30, vol_window_days=30, target_volatility=0.30):

    #same trend rule as above, each position is not fixed but scaled by volatility.
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


@register("trend_agreement", {"lookbacks": (7, 14, 30, 60, 90)})
def trend_agreement(panel, schedule, lookbacks=(7, 14, 30, 60, 90)):
    #same trend, but the weighting is the fraction of the lookbacks which agree

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


@register("trend_agreement_vol", {"lookbacks": (7, 14, 30, 60, 90, 180), "vol_window_days": 30, "target_volatility": 0.50, "no_trade_band": 0.0})
def trend_agreement_vol(panel, schedule, lookbacks=(7, 14, 30, 60, 90, 180), vol_window_days=30, target_volatility=0.50, no_trade_band=0.0):
    #same trend, effectively combining the two above. that is, scaling by the amount of lookbacks which agree, then scaling by volatility. the no_trade_band is a fraction of the total weight, i.e 0.1 means that if the change is less than 10% of the total weight, it is not traded.

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

            #if no volatility estimate, then no position,
            volatility = realised_volatility(returns, stamp, vol_window_days).reindex(members)
            scale = (target_volatility / volatility).clip(upper=1.0)
            scale = scale.where(np.isfinite(scale.to_numpy(dtype=float)) & volatility.gt(0.0), 0.0)

            desired.loc[members] = (agreement * scale / len(members)).to_numpy(dtype=float)

        #if change is inside the band, then keep previous weight and avoid rebalance
        keep = (desired - previous).abs().lt(no_trade_band) & desired.gt(0.0)
        chosen = desired.where(~keep, previous)
        if chosen.sum() > 1.0:
            chosen = desired
        weights.loc[stamp] = chosen.to_numpy(dtype=float)
        previous = chosen

    return normalise_targets(weights, schedule, assets)
