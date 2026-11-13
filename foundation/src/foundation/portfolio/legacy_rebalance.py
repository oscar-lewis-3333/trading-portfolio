
#used by quality project

import numpy as np
import pandas as pd


def rebalance_with_costs(holdings_gbp, cash_gbp, target_weights, cost_per_side_bps, frozen=None):
    #funding trades with costs, while keep positions which cannot be traded
    if (not holdings_gbp.index.is_unique or not holdings_gbp.index.equals(target_weights.index)):
        raise ValueError("Holdings and weights must have matching unique tickers.")

    holdings = holdings_gbp.astype(float)
    weights = target_weights.astype(float)
    if (not np.isfinite(holdings).all() or not np.isfinite(weights).all() or holdings.lt(0).any() or weights.lt(0).any() or not np.isfinite(cash_gbp) or cash_gbp < 0):
        raise ValueError("Expected finite, non-negative holdings, cash and weights.")

    total_weight = weights.sum()
    if total_weight > 1 + 1e-12:
        raise ValueError("Target weights cannot exceed 100%.")
    if total_weight > 1:
        weights = weights / total_weight

    if frozen is None:
        frozen = pd.Series(False, index=holdings.index)
    if (not frozen.index.equals(holdings.index) or frozen.isna().any() or not frozen.isin([True, False]).all()):
        raise ValueError("Frozen-position flags must match the holdings index.")
    frozen = frozen.astype(bool)

    cost_rate = float(cost_per_side_bps) / 10_000
    if not np.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("Invalid transaction cost.")

    equity_before = float(holdings.sum() + cash_gbp)
    fixed_values = holdings.where(frozen, 0.)
    fixed_total = float(fixed_values.sum())
    tradable_weights = weights.where(~frozen, 0.)
    tradable_weight_sum = float(tradable_weights.sum())

    def allocation(equity):
        available = max(0., equity - fixed_total)
        desired = equity * tradable_weight_sum
        scale = min(1., available / desired) if desired > 0 else 1.
        return fixed_values + tradable_weights * equity * scale

    lower, upper = fixed_total, equity_before
    for _ in range(64):
        candidate_equity = (lower + upper) / 2
        candidate_trades = allocation(candidate_equity) - holdings
        candidate_cost = candidate_trades.abs().sum() * cost_rate
        if candidate_equity + candidate_cost > equity_before:
            upper = candidate_equity
        else:
            lower = candidate_equity

    holdings_after = allocation((lower + upper) / 2)
    trades = holdings_after - holdings
    costs = trades.abs() * cost_rate
    total_cost = float(costs.sum())
    cash_after = float(cash_gbp - trades.sum() - total_cost)
    if cash_after < -1e-10 * max(1., equity_before):
        raise ValueError("Rebalance spends more cash than available.")

    return {"holdings_gbp": holdings_after,
        "trades_gbp": trades,
        "costs_gbp": costs,
        "cash_gbp": max(0., cash_after),
        "total_cost_gbp": total_cost,
        "equity_before_gbp": equity_before,
        "equity_after_gbp": equity_before - total_cost}
