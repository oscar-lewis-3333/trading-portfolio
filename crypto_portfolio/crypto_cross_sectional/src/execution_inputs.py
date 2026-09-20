#assemble audited execution references
import numpy as np
import pandas as pd


REFERENCE_COLUMNS = ["product_id", "day", "execution_at", "reference_bar_start", "assumed_available_at", "age_upper_bound_minutes", "reference_price_usd"]


def _keys(frame):
    if frame[["product_id", "day"]].isna().any().any():
        raise ValueError("Missing product/day keys.")
    
    keys = pd.MultiIndex.from_frame(frame[["product_id", "day"]])
    if keys.has_duplicates:
        raise ValueError("Duplicate product/day keys.")
    return keys


def _same_keys(left, right, label):
    a, b = _keys(left), _keys(right)
    if len(a.difference(b)) or len(b.difference(a)):
        raise ValueError(f"{label}: missing or unexpected product/day pairs.")


def assemble_execution_inputs(decision_targets, request_plan, reference_report, delayed_schedule, delayed_references, max_age_minutes=5):
    #replace whole delayed dates, check coverage and retain cash decisions
    if not np.isfinite(max_age_minutes) or not 1 <= max_age_minutes <= 5:
        raise ValueError("max_age_minutes must be between 1 and 5.")
    decisions = decision_targets.index
    if (decision_targets.empty or not isinstance(decisions, pd.DatetimeIndex) or str(decisions.tz) != "UTC" or decisions.hasnans or decisions.has_duplicates or not decisions.is_monotonic_increasing or not (decisions == decisions.normalize()).all() or decision_targets.columns.has_duplicates or "CASH" not in decision_targets):
        raise ValueError("Expected unique, sorted UTC-midnight target rows and CASH.")
    weights = decision_targets.to_numpy(dtype=float)
    if (not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Invalid target weights.")

    #independently require current members and potential departing holdings
    active = decision_targets.drop(columns="CASH").gt(0)
    required = active | active.shift(1, fill_value=False)
    expected = required.rename_axis(index="day", columns="product_id").stack()
    expected = expected.loc[expected].rename("required").reset_index()
    _same_keys(request_plan, expected, "Request plan")
    _same_keys(reference_report, request_plan, "Captured report")
    if not reference_report["outcome"].eq("ok").all():
        raise ValueError("Resolve failed captures before assembly.")

    delayed = delayed_schedule.copy()
    if (delayed["day"].isna().any() or delayed["day"].duplicated().any() or not delayed["day"].isin(request_plan["day"]).all() or not delayed["status"].eq("ready").all()):
        raise ValueError("Invalid or unresolved delayed schedule.")
    delayed_days = delayed["day"]
    delayed_plan = request_plan.loc[request_plan["day"].isin(delayed_days)]
    _same_keys(delayed_references, delayed_plan, "Delayed references")

    ordinary = reference_report.loc[~reference_report["day"].isin(delayed_days)]
    if not ordinary["reference_available"].eq(True).all():
        raise ValueError("An ordinary date still has missing references.")
    references = pd.concat([ordinary[REFERENCE_COLUMNS], delayed_references[REFERENCE_COLUMNS]],
        ignore_index=True).sort_values(["day", "product_id"]).reset_index(drop=True) 
    
    _same_keys(references, request_plan, "Assembled references")

    schedule = pd.DataFrame({"day": decisions})
    overrides = delayed.set_index("day")["execution_at"]
    schedule["execution_at"] = schedule["day"].map(overrides).fillna(schedule["day"] + pd.Timedelta(minutes=5))
    for column in ["day", "execution_at"]:
        times = schedule[column]
        if str(times.dtype) != "datetime64[ns, UTC]" or times.isna().any():
            raise ValueError("Schedule timestamps must be non-null UTC times.")
    minutes = (schedule["execution_at"] - schedule["day"]).dt.total_seconds() / 60
    if not minutes.between(5, 60).all() or not minutes.mod(1).eq(0).all():
        raise ValueError("Execution must be on a minute boundary from 00:05 to 01:00.")
    
    schedule["required_assets"] = schedule["day"].map(request_plan.groupby("day").size()).fillna(0).astype(int)
    schedule["delay_after_0005_minutes"] = minutes - 5
    schedule["status"] = np.where(schedule["required_assets"].gt(0), "ready", "cash_only")

    for column in ["day", "execution_at", "reference_bar_start", "assumed_available_at"]:
        times = references[column]
        if str(times.dtype) != "datetime64[ns, UTC]" or times.isna().any():
            raise ValueError(f"{column} must contain non-null UTC times.")
    expected_times = references["day"].map(schedule.set_index("day")["execution_at"])
    if not references["execution_at"].eq(expected_times).all():
        raise ValueError("All references on a date must use its common execution time.")
    
    starts = references["reference_bar_start"]
    availability = starts + pd.Timedelta(minutes=1)
    ages = (references["execution_at"] - starts).dt.total_seconds() / 60
    if (not starts.eq(starts.dt.floor("min")).all() or not starts.ge(references["day"]).all() or not references["assumed_available_at"].eq(availability).all() or not availability.le(references["execution_at"]).all() or not ages.between(1, max_age_minutes).all() or not np.allclose(ages, references["age_upper_bound_minutes"], rtol=0, atol=1e-12)):
        raise ValueError("Future, stale or inconsistent trade references.")
    values = references["reference_price_usd"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("References must have finite positive USD prices.")
    return schedule, references


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
    rates = rates.rename(columns={"reference_date": "fx_reference_date",
        "assumed_available_at": "fx_assumed_available_at"})

    converted = pd.merge_asof(references.sort_values("execution_at"), rates.sort_values("fx_assumed_available_at"), left_on="execution_at", right_on="fx_assumed_available_at", direction="backward", allow_exact_matches=True)
    converted["fx_reference_age_days"] = (converted["execution_at"] - converted["fx_reference_date"]).dt.total_seconds() / 86400
    if (converted["usd_per_gbp"].isna().any() or not converted["fx_reference_age_days"].between(0, max_age_days).all() or converted["fx_assumed_available_at"].gt(converted["execution_at"]).any()):
        raise ValueError("Missing, stale or future FX observations.")
    converted["reference_price_gbp"] = converted["reference_price_usd"] / converted["usd_per_gbp"]
    if not np.isfinite(converted["reference_price_gbp"]).all() or converted["reference_price_gbp"].le(0).any():
        raise ValueError("Invalid converted GBP prices.")
    return converted.sort_values(["day", "product_id"]).reset_index(drop=True)