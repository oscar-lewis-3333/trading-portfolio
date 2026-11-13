import numpy as np

#revX 0bp/9bp maker/taker. spread, slippage are estimates.
COST_MODES = {"optimistic": {"commission_bps": 0.0, "slippage_bps": 3.5},
    "pessimistic": {"commission_bps": 9.0, "slippage_bps": 3.5},
    "stress": {"commission_bps": 9.0, "slippage_bps": 21.0}}

DEFAULT_MODE = "pessimistic"


def per_side_rate(mode=DEFAULT_MODE):
    #costs charged on one side of the trade (decimal). only applies to net traded assets
    if mode not in COST_MODES:
        raise ValueError(f"Unknown cost mode '{mode}'. Choose from {sorted(COST_MODES)}.")

    assumption = COST_MODES[mode]
    rate = (assumption["commission_bps"] + assumption["slippage_bps"]) / 10000.0
    if not 0 <= rate < 1:
        raise ValueError("The per-side cost rate must lie in [0, 1).")

    return rate


def round_trip_bps(mode=DEFAULT_MODE):
    return round(per_side_rate(mode) * 2 * 10000, 2)


def daily_cash_rate(annual_rate, periods_per_year=365):
    #apply interest to held cash

    if isinstance(annual_rate, bool) or not np.isfinite(annual_rate):
        raise ValueError("annual_rate must be a finite number.")
    if annual_rate < -1:
        raise ValueError("annual_rate cannot imply a loss of more than the balance.")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year < 1:
        raise ValueError("periods_per_year must be a positive integer.")

    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def load_cost_table(path):
    #table of costs measured on revx per pair, per side. 

    import pandas as pd
    from pathlib import Path

    path = Path(path)
    table = pd.read_csv(path)
    required = {"product_id", "per_side_bps"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Cost table lacks columns: {sorted(missing)}")
    if table["product_id"].duplicated().any():
        raise ValueError("Cost table has duplicate products.")

    bps = table.set_index("product_id")["per_side_bps"].astype(float)
    if not np.isfinite(bps).all() or (bps < 0).any():
        raise ValueError("Per-side costs must be finite and non-negative.")
    
    rates = bps / 10000.0
    if (rates >= 1).any():
        raise ValueError("A per-side cost of 100% or more cannot be traded.")
    
    rates.name = "per_side_rate"
    rates.attrs["source"] = path.name

    return rates


def per_asset_rates(assets, cost_table, fallback_mode=None):
    #per side, decimal costs for each traded asset. measuered costs preferred, but if mising then a flat fallback is used
    import pandas as pd

    rates = cost_table.reindex(assets)
    if rates.isna().any():
        if fallback_mode is None:
            raise ValueError(f"No measured cost for: {sorted(rates.index[rates.isna()])}")
        rates = rates.fillna(per_side_rate(fallback_mode))

    return pd.Series(rates.to_numpy(dtype=float), index=assets)
