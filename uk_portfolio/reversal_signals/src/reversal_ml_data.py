#frozen inputs and causal features for ML extension

import json
from pathlib import Path

import numpy as np
import pandas as pd

import reversal_features
import reversal_validation


SNAPSHOT = "data/uk_extension_2023_2025_v1/prepared_2015_2025_v1.pkl"
SPEC = "data/uk_holdout_2021_2023_v1/frozen_spec.json"
FEATURE_COLUMNS = ["raw_reversal_score", "return_1", "return_20", "volatility_20", "relative_volume_20", "universe_return_1"]
MAX_TARGET_WEIGHT = 0.05
OUTCOME_COLUMNS = ["entry_time", "exit_time", "entry_open", "exit_open", "dividends_earned", "forward_return", "outcome_observed"]


def build_features(prices, schedule, rules):
    #build same eligibility mask for all four formation windows

    panels = {lookback: reversal_features.build_raw_reversal_features(prices, schedule, lookback=lookback).set_index("ticker", append=True).sort_index()
            for lookback in (1, 3, 5, 10)}
    
    activity = prices[["ticker", "Close", "Volume"]].copy()
    activity["traded_value_proxy_gbp"] = activity["Close"] * activity["Volume"]
    activity["reported_zero_volume"] = activity["Volume"].eq(0).astype(float).where(activity["Volume"].notna())

    liquidity = reversal_features.build_liquidity_features(activity, schedule, lookback=rules["liquidity_lookback_sessions"]).set_index("ticker", append=True).sort_index()
    keyed = prices.set_index("ticker", append=True).sort_index()
    valid_close = np.isfinite(keyed["adj_close"]) & keyed["adj_close"].gt(0)

    window = max(panels) + 1
    complete = valid_close.groupby(level="ticker", sort=False).transform(lambda values: values.rolling(window, min_periods=window).sum().eq(window))
    eligible = (liquidity["history_ready"] & liquidity["median_traded_value_gbp"].ge(rules["min_median_traded_value_gbp"]) & liquidity["zero_volume_fraction"].le(rules["max_zero_volume_fraction"]))

    if rules["require_complete_formation_window"]:
        eligible &= complete
    if rules["require_positive_signal_day_volume"]:
        eligible &= np.isfinite(keyed["Volume"]) & keyed["Volume"].gt(0)

    for panel in panels.values():
        eligible &= np.isfinite(panel["raw_reversal_score"])

    eligible = eligible.rename("eligible")
    for panel in panels.values():
        panel["eligible"] = eligible

    return panels, eligible, build_context_features(keyed, eligible)


def build_context_features(keyed_prices, eligible):
    #past returns, realised volatility and relative volume, at close

    close = keyed_prices["adj_close"].unstack("ticker")
    volume = keyed_prices["Volume"].unstack("ticker")
    daily = close.pct_change(fill_method=None)
    volatility = daily.rolling(20, min_periods=20).std(ddof=1)

    trailing_return = close.div(close.shift(20)).sub(1).where(volatility.notna())
    prior_volume = volume.shift(1).rolling(20, min_periods=20).mean()
    relative_volume = volume.div(prior_volume.where(prior_volume.gt(0)))
    universe_return = daily.where(eligible.unstack("ticker")).mean(axis=1)

    context = pd.DataFrame({
        "return_1": daily.stack(future_stack=True),
        "return_20": trailing_return.stack(future_stack=True),
        "volatility_20": volatility.stack(future_stack=True),
        "relative_volume_20": relative_volume.stack(future_stack=True)}).sort_index()
    
    context["universe_return_1"] = universe_return.reindex(context.index.get_level_values("Date")).to_numpy()
    return context


def build_candidates(prices, schedule, panels, grid, spec):
    #select on past information before attaching future outcome labels

    outcomes = {int(holding): reversal_features.build_forward_outcomes(prices, schedule, horizon=int(holding)).set_index("ticker", append=True).sort_index()
        for holding in sorted(grid["holding_sessions"].unique())}
    parts, calendars = {}, {}
    positions = np.arange(len(schedule))

    for config in grid.itertuples():
        config_id, holding = int(config.Index), int(config.holding_sessions)
        calendar = reversal_validation.build_rebalance_calendar(schedule, anchor=pd.Timestamp(spec["rebalance_anchor"]), rebalance_sessions=int(config.rebalance_sessions))
        dates = schedule.index[calendar["is_rebalance"].to_numpy() & (positions + 1 + holding < len(schedule))]

        calendars[config_id] = dates  #includes dates without selected stocks
        features = panels[int(config.formation_sessions)]
        ranked = reversal_validation.select_reversal_candidates(features.loc[features.index.get_level_values("Date").isin(dates)],
            score_column="raw_reversal_score", top_frac=float(config.top_frac), min_eligible=int(spec["minimum_eligible"]))
        
        selected = ranked.loc[ranked["selected"]].copy()
        selected["uncapped_target_weight"] = 1.0 / selected["target_slots"]
        selected["base_target_weight"] = selected["uncapped_target_weight"].clip(upper=MAX_TARGET_WEIGHT)
        selected["top_frac"] = float(config.top_frac)
        selected["holding_sessions"] = holding
        selected["rebalance_sessions"] = int(config.rebalance_sessions)

        parts[config_id] = selected.join(outcomes[holding][OUTCOME_COLUMNS], how="left", validate="one_to_one")
    return pd.concat(parts, names=["configuration_id"]).sort_index(), calendars


def prepare_inputs(project):
    #load local snapshot only. never download or repair historical inputs

    project = Path(project)
    for relative in (SNAPSHOT, SPEC):
        if not (project / relative).is_file():
            raise FileNotFoundError(f"Required frozen research input: {project / relative}")
        
    snapshot = pd.read_pickle(project / SNAPSHOT)
    spec = json.loads((project / SPEC).read_text())
    grid = pd.DataFrame(spec["configuration_grid"]).set_index("configuration_id").sort_index()

    prices, schedule, market = (snapshot[name] for name in ("prices", "schedule", "market"))
    panels, eligible, context = build_features(prices, schedule, spec["eligibility"])
    candidates, calendars = build_candidates(prices, schedule, panels, grid, spec)

    model_data = candidates.join(context, on=["Date", "ticker"], how="left", validate="many_to_one").sort_index()
    model_data["features_ready"] = np.isfinite(model_data[FEATURE_COLUMNS]).all(axis=1)
    if not model_data.index.is_unique:
        raise ValueError("Repeated candidate rows.")
    
    coverage = model_data.groupby(level="configuration_id").agg(candidate_trades=("features_ready", "size"), complete_features=("features_ready", "sum"),
            observed_outcomes=("outcome_observed", "sum"))
    period_rows = []
    for name, start, end in (("Development", "2017", "2020"), ("Assessment", "2020", "2022"), ("Historical extension", "2022", "2026")):
        dates = schedule.index[(schedule.index >= pd.Timestamp(start))& (schedule.index < pd.Timestamp(end))]
        period_rows.append({"period": name, "first_session": dates[0], "last_session": dates[-1], "sessions": len(dates)})
        
    return {"spec": spec, "grid": grid, "schedule": schedule, "market": market,
            "feature_panels": panels, "eligible": eligible, "model_data": model_data,
            "decision_dates": calendars, "coverage": coverage,
            "periods": pd.DataFrame(period_rows).set_index("period")}
