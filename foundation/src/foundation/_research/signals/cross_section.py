import numpy as np
import pandas as pd

from ..benchmark import eligible_universe
from ..contracts import normalise_targets
from ..panel import daily_returns, pivot_closes, trailing_return
from ..registry import register

#function designed specifically for any cross-sectional signal. rank scores by given metric, take top fraction, tie-break on product id. _select does this.


def _select(scores, top_fraction, ascending):
    observed = scores.dropna()
    observed = observed[np.isfinite(observed.to_numpy(dtype=float))]
    if observed.empty:
        return []

    direction = 1.0 if ascending else -1.0
    ordered = sorted(observed.index, key=lambda name: (direction * float(observed[name]), name))
    count = int(np.ceil(top_fraction * len(observed)))

    return ordered[:count]


@register("xs_momentum", {"lookback_days": 90, "top_fraction": 0.10})
def cross_sectional_momentum(panel, schedule, lookback_days=90, top_fraction=0.10):

    #specifically for momentum. rank cumulative returns over lookback_days, take top_fraction, tie-break as above.

    if isinstance(top_fraction, bool) or not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0, 1].")

    closes = pivot_closes(panel)
    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        if not members:
            continue

        trailing = trailing_return(closes, stamp, lookback_days).reindex(members)
        chosen = _select(trailing, top_fraction, ascending=False)
        if chosen:
            weights.loc[stamp, chosen] = 1.0 / len(chosen)

    return normalise_targets(weights, schedule, assets)


@register("xs_reversal", {"lookback_days": 1, "top_fraction": 0.20})
def short_term_reversal(panel, schedule, lookback_days=1, top_fraction=0.20):

    #for reversal signals, rank candidates by negative returns over lookback_days, take top_fraction, tie-break as above.

    if isinstance(top_fraction, bool) or not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0, 1].")

    closes = pivot_closes(panel)
    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        if not members:
            continue

        trailing = trailing_return(closes, stamp, lookback_days).reindex(members)
        chosen = _select(trailing, top_fraction, ascending=True)
        if chosen:
            weights.loc[stamp, chosen] = 1.0 / len(chosen)

    return normalise_targets(weights, schedule, assets)


@register("xs_residual_momentum", {"lookback_days": 90, "top_fraction": 0.10, "reference": "BTC-GBP", "min_observations": 40})
def residual_momentum(panel, schedule, lookback_days=90, top_fraction=0.10, reference="BTC-GBP", min_observations=40):

    #the same for residual momentum, rank cumulative residual return against the market reference. idea is to remove the beta which influences the original ranking.

    if isinstance(top_fraction, bool) or not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0, 1].")
    if isinstance(min_observations, bool) or not isinstance(min_observations, int) or min_observations < 10:
        raise ValueError("min_observations must be an integer of at least ten.")

    closes = pivot_closes(panel)
    if reference not in closes.columns:
        raise ValueError(f"'{reference}' is not in the panel.")

    returns = daily_returns(closes)
    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)

    for stamp in schedule:
        members = eligible_universe(panel, stamp)
        if not members:
            continue

        window = returns.loc[returns.index < stamp].tail(lookback_days)
        market = window[reference]
        market_variance = float(market.var(ddof=1)) if market.notna().sum() >= min_observations else np.nan
        if not np.isfinite(market_variance) or market_variance <= 0:
            continue

        scores = {}
        for name in members:
            paired = pd.concat([window[name].rename("asset"), market.rename("market")], axis=1).dropna()
            if len(paired) < min_observations:
                continue

            beta = float(paired.cov().loc["asset", "market"] / paired["market"].var(ddof=1))
            residual = paired["asset"] - beta * paired["market"]
            scores[name] = float(residual.sum())

        chosen = _select(pd.Series(scores, dtype=float), top_fraction, ascending=False)
        if chosen:
            weights.loc[stamp, chosen] = 1.0 / len(chosen)

    return normalise_targets(weights, schedule, assets)
