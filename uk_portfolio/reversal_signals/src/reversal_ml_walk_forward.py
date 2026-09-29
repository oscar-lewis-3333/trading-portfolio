#continuous 63 trading dat ML refits and config selection. interest is included, as is target cap. these do not affect selection decisions

import numpy as np
import pandas as pd

import ml_trade_filter as ml
import reversal_backtest
import reversal_walk_forward as walk_forward
from reversal_ml_data import FEATURE_COLUMNS, MAX_TARGET_WEIGHT
from reversal_risk.execution import run_cash_interest_backtest


CASH_AER = 0.038
SCORE_COLUMNS = {"Unfiltered": "uncapped_target_weight", "XGBoost": "xgb_uncapped_weight"}


def build_folds(inputs):
    schedule, spec = inputs["schedule"], inputs["spec"]
    rules = spec["walk_forward"]
    dates = schedule.loc[pd.Timestamp(spec["rebalance_anchor"]):].index[1:]
    refits = walk_forward.build_walk_forward_folds(pd.DataFrame({"calendar_marker": 0.0}, index=dates), schedule,
        initial_train_sessions=rules["initial_train_sessions"], test_sessions=rules["test_sessions"])
    
    choices = refits.copy()
    choices["train_start"] = refits.iloc[0]["test_start"]
    choices["train_sessions"] -= rules["initial_train_sessions"]
    choices = choices.loc[choices["train_sessions"] >= rules["initial_train_sessions"]].copy()

    return refits, choices


def forecast_folds(inputs, refits, *, progress=False):
    #use outcomes observable by each selection close, with complete coverage
    data, schedule, grid = (inputs[k] for k in ("model_data", "schedule", "grid"))
    parts, audit, split_rows = {}, [], []
    coverage = pd.Series(0, index=data.index, dtype="int16")

    for number, fold in enumerate(refits.itertuples(), start=1):
        end = schedule.at[fold.test_end, "market_close"]
        split = ml.observable_split(data, fold.selection_time, end, include_exit_at_start=True)
        training = data.loc[split["train_mask"]]

        candidates = data.loc[split["candidate_mask"]]
        candidate_configs = candidates.index.get_level_values("configuration_id")
        coverage += split["candidate_mask"].astype("int16")
        predictions = pd.DataFrame(np.nan, index=candidates.index, columns=["xgb_prediction"])

        counts = training.groupby(level="configuration_id").size().reindex(grid.index, fill_value=0)
        if counts.eq(0).any():
            raise ValueError(f"Empty configuration training set in fold {fold.Index}.")
        
        for config_id in grid.index:
            forecast, fit = ml.fit_predict(training.xs(config_id, level="configuration_id"),
                candidates.loc[candidate_configs == config_id], FEATURE_COLUMNS, family="xgb")
            
            predictions.loc[forecast.index, "xgb_prediction"] = forecast
            audit.append({"fold": fold.Index, "configuration_id": config_id, **fit})

        parts[fold.Index] = predictions
        split_rows.append({"fold": fold.Index, "selection_date": fold.selection_date,
                           "last_training_exit": training["exit_time"].max(),
                           "training_rows": len(training),
                           "smallest_configuration_training_set": int(counts.min()),
                           "candidate_rows": len(candidates),
                           "exits_after_block": int(candidates["exit_time"].gt(end).sum())})
        
        if progress:
            print(f"ML refit {number}/{len(refits)} complete.", flush=True)

    expected = data["signal_time"].ge(refits.iloc[0]["selection_time"]) & data["signal_time"].lt(
        schedule.at[refits.iloc[-1]["test_end"], "market_close"])
    
    np.testing.assert_array_equal(coverage.to_numpy(), expected.to_numpy(dtype="int16"))
    predictions = pd.concat(parts, names=["fold"]).sort_index().droplevel("fold").sort_index()
    pd.testing.assert_index_equal(predictions.index, data.index[expected])

    if not predictions.index.is_unique:
        raise ValueError("Candidate forecasts are duplicated across refits.")
    
    return predictions, pd.DataFrame(audit).set_index(["fold", "configuration_id"]).sort_index(), \
        pd.DataFrame(split_rows).set_index("fold")


def build_allocations(model_data, predictions):
    columns = ["signal_time", "uncapped_target_weight", "base_target_weight", "features_ready", "holding_sessions"]
    allocations = model_data.loc[predictions.index, columns].join(predictions, validate="one_to_one")
    forecast, ready = allocations["xgb_prediction"], allocations["features_ready"]
    allocations["xgb_keep"] = forecast.gt(0) | ~ready

    for base, filtered in (("uncapped_target_weight", "xgb_uncapped_weight"), ("base_target_weight", "xgb_target_weight")):
        allocations[filtered] = ml.filter_weights(allocations[base], forecast, ready)

    np.testing.assert_allclose(allocations["xgb_uncapped_weight"].clip(upper=MAX_TARGET_WEIGHT), allocations["xgb_target_weight"], rtol=1e-12, atol=1e-12)
    weights = allocations[["uncapped_target_weight", "xgb_uncapped_weight", "base_target_weight", "xgb_target_weight"]]
    if not weights.groupby(level=["configuration_id", "Date"]).sum().le(1 + 1e-12).all().all():
        raise ValueError("Target exposure exceeds available capital.")
    
    summary = allocations.groupby(level="configuration_id").agg(candidate_trades=("xgb_keep", "size"), retained_trades=("xgb_keep", "sum"))
    summary["retained_pct"] = 100 * summary["retained_trades"] / summary["candidate_trades"]

    return allocations, summary


def matched_benchmark(decisions, eligible):
    #equal weight the eligible universe at strategy's planned exposure

    eligible = eligible.reindex(decisions.index)
    if eligible.isna().any():
        raise ValueError("Missing benchmark eligibility.")
    
    counts = eligible.groupby(level="Date").transform("sum")
    exposure = decisions["target_weight"].groupby(level="Date").transform("sum")
    if (exposure.gt(0) & counts.eq(0)).any():
        raise ValueError("Positive exposure with an empty benchmark universe.")
    
    benchmark = decisions[["signal_time"]].copy()
    benchmark["target_weight"] = eligible.astype(float).mul(exposure).div(counts.where(counts.gt(0))).fillna(0.0)
    np.testing.assert_allclose(benchmark["target_weight"].groupby(level="Date").sum(), decisions["target_weight"].groupby(level="Date").sum(), rtol=1e-12, atol=1e-12)

    return benchmark


def score_configurations(inputs, refits, allocations, *, progress=False):
    #run each forward predicted candidate and its benchmark

    grid, spec = inputs["grid"], inputs["spec"]
    cash_date = refits.iloc[0]["selection_date"]
    schedule, market = inputs["schedule"].loc[cash_date:], inputs["market"].loc[cash_date:]
    tickers = market.index.get_level_values("ticker").unique().sort_values()

    daily_results = {}
    for number, config in enumerate(grid.itertuples(), start=1):
        config_id, holding = int(config.Index), int(config.holding_sessions)
        dates = inputs["decision_dates"][config_id]
        dates = dates[dates >= cash_date].union(pd.DatetimeIndex([cash_date]))

        index = pd.MultiIndex.from_product([dates, tickers], names=["Date", "ticker"])
        block = allocations.xs(config_id, level="configuration_id")
        if not block.index.isin(index).all():
            raise ValueError("An allocation falls outside its configuration calendar.")
        
        for variant, column in SCORE_COLUMNS.items():
            decisions = block[[column]].rename(columns={column: "target_weight"}).reindex(index, fill_value=0.0)
            decisions["signal_time"] = decisions.index.get_level_values("Date").map(schedule["market_close"])
            benchmark = matched_benchmark(decisions, inputs["eligible"])

            for role, targets in (("Strategy", decisions), ("Benchmark", benchmark)):
                result = reversal_backtest.run_execution_backtest(decisions=targets, market=market, schedule=schedule,
                    initial_capital_gbp=float(spec["initial_capital_gbp"]), holding_sessions=holding,
                    cost_per_side_bps=spec["walk_forward"]["cost_per_side_bps"], net_orders=spec["net_orders"])
                
                validate_daily(result, schedule, float(spec["initial_capital_gbp"]))
                daily_results[(config_id, variant, role)] = result["daily"]

            if progress:
                print(f"Scoring {number}/{len(grid)}: configuration {config_id}, {variant} complete.", flush=True)

    scoring_returns = {}
    for variant in SCORE_COLUMNS:
        for role in ("Strategy", "Benchmark"):
            returns = pd.DataFrame({int(config): daily_results[(int(config), variant, role)]["equity_return"].iloc[1:] for config in grid.index}).rename_axis(columns="configuration_id")
            pd.testing.assert_index_equal(returns.index, schedule.index[1:])
            pd.testing.assert_index_equal(returns.columns, grid.index)

            if not np.isfinite(returns).all().all():
                raise ValueError("Missing scoring returns.")
            
            scoring_returns[(variant, role)] = returns
    excess = {variant: scoring_returns[(variant, "Strategy")] - scoring_returns[(variant, "Benchmark")] for variant in SCORE_COLUMNS}
    return daily_results, scoring_returns, excess


def validate_daily(result, schedule, initial):
    daily = result["daily"]
    pd.testing.assert_index_equal(daily.index, schedule.index)
    if (not np.isfinite(daily["equity_gbp"]).all() or not np.isfinite(daily["equity_return"].iloc[1:]).all()
        or not np.isclose(daily["equity_gbp"].iloc[0], initial) or result["final_positions"]):
        raise ValueError("Unresolved continuous portfolio values or positions.")


def select_configurations(inputs, choice_folds, excess):
    choices = {variant: walk_forward.select_walk_forward_configurations(net_returns=returns, folds=choice_folds, grid=inputs["grid"], rules=inputs["spec"]["walk_forward"])
        for variant, returns in excess.items()}
    
    comparison = choice_folds[["selection_date", "test_start", "test_end"]].copy()
    for variant, selected in choices.items():
        pd.testing.assert_index_equal(selected.index, choice_folds.index)
        comparison[f"{variant}_configuration"] = selected["configuration_id"]

    return choices, comparison


def selected_targets(inputs, choices, allocations):
    grid = inputs["grid"]
    calendars = {int(holding): inputs["decision_dates"][int(grid.index[grid["holding_sessions"].eq(holding)][0])] for holding in sorted(grid["holding_sessions"].unique())}
    decisions_by_variant, lengths, rows = {}, {}, []

    for variant, selected in choices.items():
        decisions = walk_forward.build_walk_forward_decisions(choices=selected, feature_panels=inputs["feature_panels"],
            decision_dates_by_holding=calendars, min_eligible=int(inputs["spec"]["minimum_eligible"]))
        decisions = decisions[["signal_time", "eligible", "target_weight", "holding_sessions", "configuration_id", "wf_fold"]].copy()
        decisions["target_weight"] = decisions["target_weight"].clip(upper=MAX_TARGET_WEIGHT)

        candidates = decisions["target_weight"].gt(0)
        proposed = decisions.loc[candidates]
        index = pd.MultiIndex.from_arrays([proposed["configuration_id"].to_numpy(dtype="int64"),
                proposed.index.get_level_values("Date"), proposed.index.get_level_values("ticker")], names=allocations.index.names)
        
        allocated = allocations.reindex(index)
        if allocated[["base_target_weight", "xgb_target_weight"]].isna().any().any():
            raise ValueError(f"Missing selected-trade allocations: {variant}")
        
        np.testing.assert_allclose(proposed["target_weight"].to_numpy(), allocated["base_target_weight"].to_numpy(), rtol=1e-12, atol=1e-12)
        if variant == "XGBoost":
            decisions.loc[candidates, "target_weight"] = allocated["xgb_target_weight"].to_numpy()

        if (not decisions.index.is_unique or not np.isfinite(decisions["target_weight"]).all()
                or not decisions["target_weight"].between(0, MAX_TARGET_WEIGHT).all()
                or not decisions["target_weight"].groupby(level="Date").sum().le(1 + 1e-12).all()
                or not decisions["holding_sessions"].groupby(level="Date").nunique().eq(1).all()):
            raise ValueError("Invalid selected portfolio targets.")
        
        decisions_by_variant[variant] = decisions
        lengths[variant] = decisions["holding_sessions"].groupby(level="Date").first().astype("int64")
        rows.append({"variant": variant, "decision_dates": len(lengths[variant]),
                     "candidate_trades": int(candidates.sum()),
                     "positive_targets": int(decisions["target_weight"].gt(0).sum()),
                     "max_target_weight_pct": 100 * decisions["target_weight"].max()})
        
    return decisions_by_variant, lengths, pd.DataFrame(rows).set_index("variant")


def evaluate_selected(inputs, choice_folds, decisions_by_variant, lengths, *, progress=False):
    cash_date = choice_folds.iloc[0]["selection_date"]
    schedule, market = inputs["schedule"].loc[cash_date:], inputs["market"].loc[cash_date:]
    initial = float(inputs["spec"]["initial_capital_gbp"])
    daily_runs, rows = {}, []

    for variant, decisions in decisions_by_variant.items():
        benchmark = matched_benchmark(decisions, decisions["eligible"])
        for portfolio, targets in (("Strategy", decisions), ("Benchmark", benchmark)):
            result = run_cash_interest_backtest(decisions=targets, market=market, schedule=schedule, initial_capital_gbp=initial, holding_sessions_by_date=lengths[variant],
                cost_per_side_bps=float(inputs["spec"]["walk_forward"]["cost_per_side_bps"]), net_orders=True, cash_aer=CASH_AER)
            validate_daily(result, schedule, initial)
            daily = result["daily"]
            daily_runs[(variant, portfolio)] = daily

            rows.append({"variant": variant, "portfolio": portfolio,
                         "ending_equity_gbp": daily["equity_gbp"].iloc[-1],
                         "net_total_return_pct": 100 * (daily["equity_gbp"].iloc[-1] / initial - 1),
                         "max_drawdown_pct": 100 * (daily["equity_gbp"] / daily["equity_gbp"].cummax() - 1).min(),
                         "total_cost_gbp": daily["cost_gbp"].sum(),
                         "cash_interest_gbp": daily["cash_interest_gbp"].sum()})
            if progress:
                print(f"Continuous portfolio complete: {variant} / {portfolio}", flush=True)

    return daily_runs, pd.DataFrame(rows).set_index(["variant", "portfolio"])


def run_walk_forward(inputs, *, progress=False):
    refits, folds = build_folds(inputs)
    predictions, fits, splits = forecast_folds(inputs, refits, progress=progress)
    allocations, filter_summary = build_allocations(inputs["model_data"], predictions)
    scoring_daily, scoring_returns, excess = score_configurations(inputs, refits, allocations, progress=progress)

    choices, comparison = select_configurations(inputs, folds, excess)
    decisions, lengths, selected_summary = selected_targets(inputs, choices, allocations)
    daily, summary = evaluate_selected(inputs, folds, decisions, lengths, progress=progress)
    
    return {"refit_folds": refits, "choice_folds": folds, "predictions": predictions,
            "fit_summary": fits, "split_summary": splits, "allocations": allocations,
            "filter_summary": filter_summary, "scoring_daily": scoring_daily,
            "scoring_returns": scoring_returns, "net_excess": excess, "choices": choices,
            "choice_comparison": comparison, "selected_decisions": decisions,
            "selected_summary": selected_summary, "daily": daily, "summary": summary}
