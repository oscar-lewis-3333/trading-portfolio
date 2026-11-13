import numpy as np
import pandas as pd

from .contracts import normalise_targets, validate_panel, validate_schedule
from .panel import eligible_at, pivot_closes, pivot_eligible

MIN_UNIVERSE = 1
MAX_UNIVERSE = 30


def eligible_universe(panel, decision_at, min_assets=MIN_UNIVERSE, max_assets=MAX_UNIVERSE):
    #shared universe, with no signal defining its own

    for name, value in (("min_assets", min_assets), ("max_assets", max_assets)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if min_assets > max_assets:
        raise ValueError("min_assets cannot exceed max_assets.")

    eligible = pivot_eligible(panel)
    members = eligible_at(eligible, decision_at)
    names = sorted(members.index[members])

    #if breadth too small, then hold cash
    if len(names) < min_assets:
        return []

    if len(names) > max_assets:
        raise ValueError(f"{len(names)} assets are eligible at {decision_at}, above the cap of {max_assets}. Set the panel's eligible flag to the top assets by trailing liquidity before building signals.")

    return names


def equal_weight_targets(panel, schedule, min_assets=MIN_UNIVERSE, max_assets=MAX_UNIVERSE):
    #common benchmark for measuring signals against

    validate_panel(panel)
    validate_schedule(schedule)

    assets = sorted(panel["product_id"].unique())
    weights = pd.DataFrame(0.0, index=schedule, columns=assets)

    for stamp in schedule:
        members = eligible_universe(panel, stamp, min_assets, max_assets)
        if members:
            weights.loc[stamp, members] = 1.0 / len(members)

    return normalise_targets(weights, schedule, assets)


def benchmark_returns(panel, schedule, returns, **kwargs):
    #convenience wrapper, so callers never rebuild the benchmark each time its called
    from .evaluate import run_portfolio

    targets = equal_weight_targets(panel, schedule)

    return run_portfolio(targets, returns, **kwargs)["net_return"]


def market_factor(panel, reference="BTC-GBP"):
    #single asset market proxy. used by residual signal as opposed to the benchmark
    closes = pivot_closes(panel)
    if reference not in closes.columns:
        raise ValueError(f"'{reference}' is not in the panel.")

    series = closes[reference].pct_change(fill_method=None)

    return series.rename("market_return")


def universe_breadth(panel, schedule, min_assets=MIN_UNIVERSE, max_assets=MAX_UNIVERSE):
    #the breadth of the qualifying assets at each decision
    
    validate_schedule(schedule)
    counts = [len(eligible_universe(panel, stamp, min_assets, max_assets)) for stamp in schedule]
    frame = pd.DataFrame({"eligible_assets": counts}, index=schedule)
    frame["active"] = frame["eligible_assets"].gt(0)

    if not np.isfinite(frame["eligible_assets"].to_numpy(dtype=float)).all():
        raise ValueError("Universe breadth must be finite.")

    return frame
