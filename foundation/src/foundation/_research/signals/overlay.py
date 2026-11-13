import numpy as np
import pandas as pd

from ..benchmark import eligible_universe
from ..contracts import normalise_targets
from ..panel import daily_returns, pivot_closes, realised_volatility
from ..registry import register

#portfolio/book level overlays

@register("vol_target", {"vol_window_days": 30, "target_volatility": 0.30, "max_exposure": 1.0})
def volatility_target(panel, schedule, vol_window_days=30, target_volatility=0.30, max_exposure=1.0):

    #scale whole book by forecast volatility
    if not np.isfinite(target_volatility) or target_volatility <= 0:
        raise ValueError("target_volatility must be positive and finite.")
    if not np.isfinite(max_exposure) or not 0 < max_exposure <= 1:
        raise ValueError("max_exposure must lie in (0, 1]; leverage is unavailable.")

    closes = pivot_closes(panel)
    returns = daily_returns(closes)
    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        if not members:
            continue

        #forecast equal-weight portfolio volatility, as opposed to each individual asset's
        window = returns.loc[returns.index < stamp, members].tail(vol_window_days)
        if len(window) < vol_window_days or window.isna().any().any():
            continue

        portfolio = window.mean(axis=1)
        forecast = float(portfolio.std(ddof=1) * np.sqrt(365))
        if not np.isfinite(forecast) or forecast <= 0:
            continue

        exposure = min(max_exposure, target_volatility / forecast)
        weights.loc[stamp, members] = exposure / len(members)

    return normalise_targets(weights, schedule, assets)


def apply_exposure(targets, exposure):
    #compose each signal with a single exposure series, separating sizing from selection

    from ..contracts import CASH, validate_targets

    validate_targets(targets)
    scale = pd.Series(exposure, dtype=float).reindex(targets.index)
    if scale.isna().any():
        raise ValueError("The exposure series must cover every decision.")
    if not np.isfinite(scale.to_numpy(dtype=float)).all() or ((scale < 0) | (scale > 1)).any():
        raise ValueError("Exposure must lie in [0, 1].")

    assets = [column for column in targets.columns if column != CASH]
    scaled = targets[assets].mul(scale, axis=0)

    return normalise_targets(scaled, targets.index, assets)


def forecast_volatility(panel, schedule, window_days=30):
    #exposed on its own since several signals use the same forecast

    closes = pivot_closes(panel)
    returns = daily_returns(closes)
    rows = {}

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        if not members:
            rows[stamp] = np.nan
            continue
        estimate = realised_volatility(returns[members].mean(axis=1).to_frame("portfolio"), stamp, window_days)
        rows[stamp] = float(estimate.iloc[0])

    return pd.Series(rows, name="forecast_volatility").reindex(schedule)
