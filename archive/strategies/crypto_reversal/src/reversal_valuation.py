
import numpy as np
import pandas as pd

from fx import build_gbp_valuation_prices


def build_development_gbp_inputs(prices, references, fx_observations, protocol):
    #return daily prices in GBP, along with execution GBP marks and a daily FX audit
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    if not start < end <= pd.Timestamp(protocol["holdout_start"], tz="UTC"):
        raise ValueError("Invalid development/holdout boundaries.")
    if references.empty or references.duplicated(["product_id", "day"]).any():
        raise ValueError("Expected unique, non-empty execution references.")
    
    for column in ("day", "execution_at"):
        times = references[column]
        if (not isinstance(times.dtype, pd.DatetimeTZDtype) or str(times.dt.tz) != "UTC" or times.isna().any() or not times.between(start, end, inclusive="left").all()):
            raise ValueError("Execution references must remain inside development.")
    if (not references["day"].eq(references["day"].dt.normalize()).all() or not references["execution_at"].dt.normalize().eq(references["day"]).all()):
        raise ValueError("Execution day must match its UTC decision date.")
    if not references["product_id"].str.endswith("-USD", na=False).all():
        raise ValueError("Expected USD-denominated execution references.")

    daily = prices.loc[prices["timestamp"].ge(start) & prices["timestamp"].lt(end)].copy()
    daily_gbp, fx_audit = build_gbp_valuation_prices(daily, fx_observations, max_age_days=7)
    execution_gbp = convert_execution_references_to_gbp(references, fx_observations, max_age_days=7)
    return daily_gbp, execution_gbp, fx_audit


def convert_execution_references_to_gbp(references, fx_observations, max_age_days=7):
    #use latest FX observation assumed available at each execution time
    if references.empty or not np.isfinite(max_age_days) or max_age_days <= 0:
        raise ValueError("Provide references and a positive FX age limit.")
    rates = fx_observations[["reference_date", "assumed_available_at", "usd_per_gbp"]].copy()

    if rates.empty:
        raise ValueError("Missing FX history.")
    for times in [references["execution_at"], rates["reference_date"], rates["assumed_available_at"]]:
        if str(times.dtype) != "datetime64[ns, UTC]" or times.isna().any():
            raise ValueError("Use non-null UTC timestamps.")
        
    if (rates["reference_date"].duplicated().any() or not rates["reference_date"].eq(rates["reference_date"].dt.normalize()).all() or not rates["assumed_available_at"].eq(rates["reference_date"] + pd.Timedelta(days=1)).all()):
        raise ValueError("FX dates must follow the existing next-midnight convention.")
    
    for values in [rates["usd_per_gbp"], references["reference_price_usd"]]:
        numeric = values.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric <= 0).any():
            raise ValueError("Prices and FX rates must be finite and positive.")
        
    rates = rates.rename(columns={"reference_date": "fx_reference_date", "assumed_available_at": "fx_assumed_available_at"})

    converted = pd.merge_asof(references.sort_values("execution_at"), rates.sort_values("fx_assumed_available_at"), left_on="execution_at", right_on="fx_assumed_available_at", direction="backward", allow_exact_matches=True)
    converted["fx_reference_age_days"] = (converted["execution_at"] - converted["fx_reference_date"]).dt.total_seconds() / 86400
    if (converted["usd_per_gbp"].isna().any() or not converted["fx_reference_age_days"].between(0, max_age_days).all() or converted["fx_assumed_available_at"].gt(converted["execution_at"]).any()):
        raise ValueError("Missing, stale or future FX observations.")
    
    converted["reference_price_gbp"] = converted["reference_price_usd"] / converted["usd_per_gbp"]
    if not np.isfinite(converted["reference_price_gbp"]).all() or converted["reference_price_gbp"].le(0).any():
        raise ValueError("Invalid converted GBP prices.")
    
    return converted.sort_values(["day", "product_id"]).reset_index(drop=True)
