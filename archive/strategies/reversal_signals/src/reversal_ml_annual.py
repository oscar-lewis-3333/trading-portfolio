
#annual refits and separate yearly backtests for ML experiment

import numpy as np
import pandas as pd

import ml_trade_filter as ml
import reversal_backtest
from reversal_ml_data import FEATURE_COLUMNS


WEIGHT_COLUMNS = {"Unfiltered": "base_target_weight", "Ridge": "ridge_target_weight",
                  "XGBoost": "xgb_target_weight", "Ridge control": "ridge_control_weight",
                  "XGBoost control": "xgb_control_weight"}
FAMILIES = {"Ridge": "ridge", "XGBoost": "xgb"}
PRIMARY_VARIANTS = ["Unfiltered", "Ridge", "XGBoost"]
COMPARISON_VARIANTS = ["Unfiltered", "XGBoost", "XGBoost control"]


def forecast_year(model_data, year, cases):
    #refit only requested families. candidates who have missing outcomes are preserved

    config_ids = sorted({config for config, _ in cases})
    data = model_data.loc[model_data.index.get_level_values("configuration_id").isin(config_ids)]
    split = ml.observable_split(data, pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), require_exit_before_end=True)

    candidates = data.loc[split["candidate_mask"]]
    training = data.loc[split["train_mask"]]
    predictions = pd.DataFrame(index=candidates.index, columns=["mean_prediction"], dtype=float)
    audit = []
    
    for config_id in config_ids:
        rows = candidates.index.get_level_values("configuration_id") == config_id
        train = training.xs(config_id, level="configuration_id")
        predictions.loc[rows, "mean_prediction"] = train["forward_return"].mean()

        for variant, family in FAMILIES.items():
            if (config_id, variant) not in cases:
                continue

            forecast, counts = ml.fit_predict(train, candidates.loc[rows], FEATURE_COLUMNS, family=family)
            predictions.loc[rows, f"{family}_prediction"] = forecast
            audit.append({"year": year, "configuration_id": config_id, "variant": variant, **counts})

    allocations = candidates[["signal_time", "base_target_weight", "features_ready"]].join(predictions, validate="one_to_one")
    
    for config_id, variant in cases:
        if variant not in FAMILIES:
            continue

        family = FAMILIES[variant]
        rows = allocations.index.get_level_values("configuration_id") == config_id
        block = allocations.loc[rows]
        base = block["base_target_weight"]

        filtered = ml.filter_weights(base, block[f"{family}_prediction"], block["features_ready"])
        allocations.loc[rows, f"{family}_target_weight"] = filtered
        allocations.loc[rows, f"{family}_control_weight"] = ml.exposure_matched_control(base, filtered, group_levels=["configuration_id", "Date"])
    return predictions, allocations, pd.DataFrame(audit), split, data


def prediction_skill(data, predictions, split):
    #2019 squared error comparison against training period mean

    rows = []
    for config_id in predictions.index.get_level_values("configuration_id").unique():
        mask = data.index.get_level_values("configuration_id") == config_id
        scored = data.loc[split["score_mask"] & mask]
        actual = scored["forward_return"]
        forecast = predictions.loc[scored.index]
        mean_mse = (actual - forecast["mean_prediction"]).pow(2).mean()

        row = {"configuration_id": config_id, "training_rows": int((split["train_mask"] & mask).sum()), "validation_rows": len(scored), "mean_rmse_bps": np.sqrt(mean_mse) * 10_000}

        for family in ("ridge", "xgb"):
            mse = (actual - forecast[f"{family}_prediction"]).pow(2).mean()
            row[f"{family}_rmse_bps"] = np.sqrt(mse) * 10_000
            row[f"{family}_mse_skill_vs_mean"] = 1 - mse / mean_mse if mean_mse > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("configuration_id")


def run_annual_cases(inputs, years, cases, *, progress=False):
    #reset each year to cash. ensure this is not a continuous test

    schedule, market, grid, spec = (inputs[k] for k in ("schedule", "market", "grid", "spec"))
    initial = float(spec["initial_capital_gbp"])
    tickers = market.index.get_level_values("ticker").unique().sort_values()
    daily_runs, rows, fit_parts, forecast_parts = {}, [], [], {}
    skill = None

    for year in years:
        predictions, allocations, audit, split, data = forecast_year(inputs["model_data"], year, cases)
        fit_parts.append(audit)
        forecast_parts[year] = predictions
        if year == 2019 and set(PRIMARY_VARIANTS).issubset({variant for _, variant in cases}):
            skill = prediction_skill(data, predictions, split)

        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        cash_date = schedule.index[schedule.index < start][-1]
        year_schedule = schedule.loc[(schedule.index >= cash_date) & (schedule.index < end)]
        year_market = market.loc[market.index.get_level_values("Date").isin(year_schedule.index)]

        for config_id, variant in cases:
            holding = int(grid.loc[config_id, "holding_sessions"])
            dates = inputs["decision_dates"][config_id]
            positions = year_schedule.index.get_indexer(dates)

            dates = dates[(positions > 0) & (positions + 1 + holding < len(year_schedule))]
            dates = dates.union(pd.DatetimeIndex([cash_date]))
            index = pd.MultiIndex.from_product([dates, tickers], names=["Date", "ticker"])

            block = allocations.xs(config_id, level="configuration_id")
            if not block.index.isin(index).all():
                raise ValueError("Candidate decisions fall outside the annual calendar.")
            
            column = WEIGHT_COLUMNS[variant]
            if not np.isfinite(block[column]).all():
                raise ValueError(f"Invalid allocations: {year}, {config_id}, {variant}")
            
            decisions = block[[column]].rename(columns={column: "target_weight"}).reindex(index, fill_value=0.0)
            decisions["signal_time"] = decisions.index.get_level_values("Date").map(year_schedule["market_close"])
            result = reversal_backtest.run_execution_backtest(decisions=decisions, market=year_market, schedule=year_schedule, initial_capital_gbp=initial,
                holding_sessions=holding, cost_per_side_bps=float(spec["walk_forward"]["cost_per_side_bps"]), net_orders=spec["net_orders"])
            daily = result["daily"]
            pd.testing.assert_index_equal(daily.index, year_schedule.index)

            equity, returns = daily["equity_gbp"], daily["equity_return"].iloc[1:]
            if not np.isfinite(equity).all() or not np.isfinite(returns).all() or result["final_positions"]:
                raise ValueError(f"Unresolved annual portfolio: {year}, {config_id}, {variant}")
            
            daily_runs[(year, config_id, variant)] = daily
            volatility = returns.std()
            rows.append({"year": year, "configuration_id": config_id, "variant": variant,
                         "net_return_pct": 100 * (equity.iloc[-1] / initial - 1),
                         "net_sharpe": np.sqrt(252) * returns.mean() / volatility if volatility > 0 else np.nan,
                         "max_drawdown_pct": 100 * (equity / equity.cummax() - 1).min(),
                         "mean_stock_exposure_pct": 100 * (daily["stock_value_gbp"] / equity).iloc[1:].mean(),
                         "costs_gbp": daily["cost_gbp"].sum()})
            
        if progress:
            print(f"{year}: {len(cases)} annual portfolio cases complete.", flush=True)
    returns = pd.concat({key: daily["equity_return"].iloc[1:] for key, daily in daily_runs.items()}, names=["validation_year", "configuration_id", "variant"])
    statistics = returns.groupby(level=["configuration_id", "variant"]).agg(["mean", "std", "count"])
    statistics["net_sharpe"] = np.sqrt(252) * statistics["mean"] / statistics["std"].where(statistics["std"].gt(0))

    return {"daily": daily_runs, "summary": pd.DataFrame(rows).set_index(
            ["configuration_id", "variant", "year"]).sort_index(),
            "statistics": statistics, "prediction_skill": skill,
            "fit_summary": pd.concat(fit_parts, ignore_index=True),
            "predictions": pd.concat(forecast_parts, names=["year"]).sort_index()}


def select_finalists(development, grid):
    #keep orignal Sharpe ranking, before recording 12/28 as sensitivity cases

    selection = development["statistics"].reset_index()
    selection = selection.loc[selection["variant"].isin(PRIMARY_VARIANTS) & np.isfinite(selection["net_sharpe"])]
    finalists = selection.sort_values(["variant", "net_sharpe", "configuration_id"], ascending=[True, False, True]).drop_duplicates("variant")
    finalists = finalists.set_index("variant").reindex(PRIMARY_VARIANTS)
    configurations = finalists["configuration_id"].astype(int).to_dict()

    assessment = pd.DataFrame([(variant, configurations[variant], "primary") for variant in PRIMARY_VARIANTS] + [("XGBoost", 12, "sensitivity"),
                             ("XGBoost", 28, "sensitivity")], columns=["variant", "configuration_id", "role"])
    assessment = assessment.join(development["statistics"]["net_sharpe"].rename("development_net_sharpe"), on=["configuration_id", "variant"],
            validate="many_to_one").join(grid[["formation_sessions", "top_frac", "holding_sessions"]],on="configuration_id", validate="many_to_one")
    cases = set()

    for row in assessment.itertuples(index=False):
        cases.add((int(row.configuration_id), row.variant))
        if row.variant != "Unfiltered":
            cases.update(((int(row.configuration_id), "Unfiltered"), (int(row.configuration_id), f"{row.variant} control")))

    return finalists, assessment, sorted(cases)


def link_fixed_configuration(stages, schedule, *, configuration_id=28):
    #compound annual simulations. year-end gaps are kept.
    columns = {}
    for variant in COMPARISON_VARIANTS:
        annual = [daily["equity_return"].iloc[1:] for stage in stages for (year, config_id, name), daily in stage["daily"].items() if config_id == configuration_id and name == variant]
        columns[variant] = pd.concat(annual).sort_index()

    returns = pd.DataFrame(columns)
    expected = schedule.index[(schedule.index >= pd.Timestamp("2017-01-01")) & (schedule.index < pd.Timestamp("2026-01-01"))]
    pd.testing.assert_index_equal(returns.index, expected)
    if not returns.index.is_unique or not np.isfinite(returns).all().all():
        raise ValueError("Invalid linked annual return history.")
    
    start = schedule.index[schedule.index < returns.index[0]][-1]
    equity = pd.concat([pd.DataFrame(100.0, index=pd.DatetimeIndex([start], name="Date"), columns=returns.columns), 100 * (1 + returns).cumprod()])
    drawdown = 100 * (equity / equity.cummax() - 1)
    
    summary = pd.DataFrame({"linked_return_pct": equity.iloc[-1] - 100,
                            "net_sharpe": np.sqrt(252) * returns.mean() / returns.std(),
                            "annual_volatility_pct": 100 * np.sqrt(252) * returns.std(),
                            "max_drawdown_pct": drawdown.min()})
    return {"returns": returns, "equity": equity, "drawdown": drawdown, "summary": summary}
