from copy import deepcopy

import numpy as np
import pandas as pd

from reversal_signals import rank_reversal_candidates


def _positive_day_count(value, name):
    #pandas grid row can represent integer columns as floats
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    
    number = float(value)
    if not np.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    
    return int(number)


def build_configuration_targets(eligibility_panel, daily_universe, rebalance_schedules, protocol, configuration):
    #return strategy targets, matched benchmark, rankings and a weight audit

    if protocol["universe"]["min_assets"] != 1:
        raise ValueError("Load the revised protocol with min_assets=1 first.")
    
    lookback = _positive_day_count(configuration["lookback_days"], "lookback_days")
    interval = _positive_day_count(configuration["rebalance_days"], "rebalance_days")
    fraction = float(configuration["top_fraction"])
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")

    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    holdout_start = pd.Timestamp(protocol["holdout_start"], tz="UTC")
    if not start < end <= holdout_start:
        raise ValueError("Invalid development/holdout boundaries.")
    
    dates = pd.date_range(start, end, freq=f"{interval}D", inclusive="left", name="decision_at")
    schedule = rebalance_schedules.loc[rebalance_schedules["rebalance_days"].eq(interval)].copy()
    if not pd.DatetimeIndex(schedule["decision_at"]).equals(dates):
        raise ValueError("Supply the full ordered development schedule for this interval.")
    if (schedule["universe_size"].isna().any() or schedule["universe_size"].lt(0).any() or not schedule["active"].eq(schedule["universe_size"].gt(0)).all()):
        raise ValueError("Schedule activity must agree with whether the eligible universe is non-empty.")

    members = daily_universe.loc[daily_universe["decision_at"].isin(dates)].copy()
    counts = members.groupby("decision_at").size().reindex(dates, fill_value=0)
    if not np.array_equal(counts.to_numpy(), schedule["universe_size"].to_numpy()):
        raise ValueError("Membership counts disagree with the schedule; rebuild both under the revised protocol.")

    settings = deepcopy(protocol)
    settings["signal"]["formation_days"] = lookback
    settings["signal"]["selection_fraction"] = fraction
    rankings = rank_reversal_candidates(eligibility_panel, members, settings)
    #all development members get a column. departing assets have zero targets
    assets = sorted(daily_universe["product_id"].unique())
    if "CASH" in assets:
        raise ValueError("CASH is reserved for the cash-weight column.")

    def weight_matrix(column):
        weights = rankings.pivot(index="decision_at", columns="product_id", values=column)
        weights = weights.reindex(index=dates, columns=assets).fillna(0.0)
        weights["CASH"] = counts.eq(0).astype(float)
        if (not np.isfinite(weights.to_numpy()).all() or weights.lt(0).any().any() or not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("Target portfolios must have non-negative weights summing to one.")
        return weights

    strategy = weight_matrix("reversal_weight")
    benchmark = weight_matrix("benchmark_weight")
    
    audit = pd.DataFrame(index=dates)
    audit["universe_size"] = counts
    audit["selected_assets"] = strategy.drop(columns="CASH").gt(0).sum(axis=1)
    audit["expected_selected"] = np.ceil(fraction * counts).astype("int64")
    audit["strategy_weight_sum"] = strategy.sum(axis=1)
    audit["benchmark_weight_sum"] = benchmark.sum(axis=1)
    if not audit["selected_assets"].eq(audit["expected_selected"]).all():
        raise ValueError("Selected-asset counts disagree with the ceiling rule.")
    
    return strategy, benchmark, rankings, audit

def build_sweep_targets(eligibility_panel, daily_universe, rebalance_schedules, protocol, development_grid):
    #build each config independently and retain only target weights
    required = ["lookback_days", "top_fraction", "rebalance_days"]
    if (development_grid.empty or not development_grid.index.is_unique or development_grid.index.hasnans or not set(required).issubset(development_grid.columns)):
        raise ValueError("Supply a non-empty grid with unique configuration IDs and all parameter columns.")
    if development_grid[required].duplicated().any():
        raise ValueError("The grid contains duplicate parameter combinations.")

    targets, summaries = {}, []
    for config_id, configuration in development_grid.iterrows():
        strategy, benchmark, _, audit = build_configuration_targets(eligibility_panel, daily_universe, rebalance_schedules, protocol, configuration)
        targets[config_id] = {"strategy": strategy, "benchmark": benchmark}
        summaries.append({"config_id": config_id,
            "lookback_days": int(configuration["lookback_days"]),
            "top_fraction": float(configuration["top_fraction"]),
            "rebalance_days": int(configuration["rebalance_days"]),
            "scheduled_decisions": len(audit),
            "cash_decisions": int(strategy["CASH"].eq(1.0).sum()),
            "min_universe_size": int(audit["universe_size"].min()),
            "max_universe_size": int(audit["universe_size"].max()),
            "min_selected_assets": int(audit["selected_assets"].min()),
            "max_selected_assets": int(audit["selected_assets"].max()),
            "max_weight_sum_error": float(max(audit["strategy_weight_sum"].sub(1).abs().max(),audit["benchmark_weight_sum"].sub(1).abs().max())),
            "first_decision": audit.index[0],
            "last_decision": audit.index[-1]
        })

    return targets, pd.DataFrame(summaries).set_index("config_id")
