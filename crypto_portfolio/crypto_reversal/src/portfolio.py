import numpy as np
import pandas as pd


def rebalance_gbp(units, cash_gbp, prices_gbp, target_weights, fee_bps=9.0, other_cost_bps=3.5):
    #long-only fractional-unit rebalance, with costs paid in GBP
    held = pd.Series(units, dtype=float).copy()
    prices = pd.Series(prices_gbp, dtype=float).copy()
    weights = pd.Series(target_weights, dtype=float).copy()

    if not all(s.index.is_unique for s in (held, prices, weights)):
        raise ValueError("Duplicate asset identifiers.")
    if "CASH" in held.index or "CASH" not in weights.index:
        raise ValueError("Supply cash separately and include CASH in targets.")

    for values in (held.to_numpy(), weights.to_numpy()):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Units and weights must be finite and non-negative.")

    if not np.isfinite([cash_gbp, fee_bps, other_cost_bps]).all():
        raise ValueError("Cash and cost assumptions must be finite.")
    if min(cash_gbp, fee_bps, other_cost_bps) < 0:
        raise ValueError("Cash and costs cannot be negative.")
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Target weights must sum to one.")

    #remove only floating point sum error
    weights = weights / weights.sum()

    fee_rate = fee_bps / 10000
    other_rate = other_cost_bps / 10000
    cost_rate = fee_rate + other_rate
    if cost_rate >= 1:
        raise ValueError("Total cost per side must be less than 100%.")

    #include departing holdings even if omitted from new targets
    assets = held.index.union(weights.drop("CASH").index).sort_values()
    held = held.reindex(assets, fill_value=0.0)
    asset_weights = weights.reindex(assets, fill_value=0.0)

    #prices are required only for held/targeted assets
    needed = held.gt(0) | asset_weights.gt(0)
    p = prices.reindex(assets[needed])
    if not np.isfinite(p.to_numpy()).all() or p.le(0).any():
        raise ValueError("A held or targeted asset lacks a valid GBP price.")

    old_values = held.loc[needed] * p
    equity_before = float(cash_gbp + old_values.sum())
    if not np.isfinite(equity_before) or equity_before <= 0:
        raise ValueError("Portfolio equity must be positive and finite.")

    #solve: post cost equity + costs on actual trade = pre trade equity
    lower, upper = 0.0, equity_before
    w = asset_weights.loc[needed]

    for _ in range(64):
        candidate = (lower + upper) / 2
        cost = cost_rate * (w * candidate - old_values).abs().sum()

        if candidate + cost <= equity_before:
            lower = candidate
        else:
            upper = candidate

    new_values = w * lower
    signed = new_values - old_values
    fees = fee_rate * signed.abs()
    other = other_rate * signed.abs()
    total_cost = float(fees.sum() + other.sum())
    new_cash = float(cash_gbp - signed.sum() - total_cost)
    tolerance = 1e-10 * max(1.0, equity_before)

    if new_cash < -tolerance:
        raise RuntimeError("Rebalance would borrow cash.")
    new_cash = max(new_cash, 0.0)
    new_units = pd.Series(0.0, index=assets)
    new_units.loc[needed] = new_values / p

    equity_after = float(new_cash + (new_units.loc[needed] * p).sum())
    error = equity_after + total_cost - equity_before
    if abs(error) > tolerance:
        raise RuntimeError("Rebalance accounting does not reconcile.")
    if abs(new_cash - weights["CASH"] * equity_after) > tolerance:
        raise RuntimeError("Post-cost cash allocation does not match target.")

    trades = pd.DataFrame({"reference_price_gbp": p,
        "units_before": held.loc[needed],
        "units_after": new_units.loc[needed],
        "signed_notional_gbp": signed,
        "fee_gbp": fees,
        "other_cost_gbp": other
    }).rename_axis("product_id")

    audit = pd.Series({
        "equity_before_gbp": equity_before,
        "equity_after_gbp": equity_after,
        "traded_notional_gbp": float(signed.abs().sum()),
        "total_cost_gbp": total_cost,
        "cash_after_gbp": new_cash,
        "accounting_error_gbp": error
    })

    return new_units, new_cash, trades, audit