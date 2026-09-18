import numpy as np
import pandas as pd

import etf_backtest
import etf_features
import etf_data

def run_lookback_sweep(monthly_prices, open_prices, close_prices, decision_schedule, *, initial_capital_gbp, cost_per_side_bps, cash_rate_annual=0.0, lookbacks=(3, 6, 9, 12)):
    #compare lookback columns using the same decision and trading dates

    lookbacks = tuple(lookbacks)
    if not lookbacks or len(set(lookbacks)) != len(lookbacks):
        raise ValueError("Provide a nonempty collection of distinct lookbacks.")
    if decision_schedule.empty or not decision_schedule.index.is_unique:
        raise ValueError("Provide a nonempty schedule with unique decision dates.")
    if not monthly_prices.columns.equals(open_prices.columns):
        raise ValueError("Monthly and daily prices must contain the same assets.")
    execution_dates = pd.DatetimeIndex(decision_schedule["execution_date"], name="Date")

    if (execution_dates.hasnans or not execution_dates.is_unique or (execution_dates <= decision_schedule.index).any()):
        raise ValueError("Each decision must have a unique, later execution date.")

    records = []
    results = {}

    for lookback in lookbacks:
        _, signal = etf_features.build_trend_features(monthly_prices, lookback_months=lookback)
        weights = etf_backtest.build_target_weights(signal)
        weights = weights.reindex(decision_schedule.index).copy()

        #every config must be ready on same decision dates
        if weights.isna().any().any():
            raise ValueError(f"{lookback} months: incomplete signals on the common dates.")
        weights.index = execution_dates
        result = etf_backtest.run_backtest(
            open_prices=open_prices,
            close_prices=close_prices,
            execution_weights=weights,
            initial_capital_gbp=initial_capital_gbp,
            cost_per_side_bps=cost_per_side_bps,
            cash_rate_annual=cash_rate_annual)

        daily = result["daily"]
        nav = daily["nav_close"]
        returns = daily["net_return"]
        years = ((nav.index[-1] - nav.index[0]).days + 1) / 365.25
        peaks = nav.cummax().clip(lower=initial_capital_gbp)
        drawdown = nav / peaks - 1

        records.append({
            "lookback_months": lookback,
            "final_value_gbp": nav.iloc[-1],
            "cagr_pct": ((nav.iloc[-1] / initial_capital_gbp) ** (1 / years) - 1) * 100,
            "annualised_vol_pct": returns.std(ddof=1) * np.sqrt(252) * 100,
            "max_drawdown_pct": drawdown.min() * 100,
            "mean_cash_pct": (daily["cash"] / nav).mean() * 100,
            "total_cost_gbp": daily["cost"].sum(),
            "rebalances": int(daily["rebalanced"].sum())
        })

        results[lookback] = result

    summary = pd.DataFrame(records).set_index("lookback_months")
    return summary, results

def run_frozen_evaluation(monthly_prices, open_prices, close_prices, decision_schedule, excluded_quotes, spec):
    #evaluate one fixed strategy and its fixed cash benchmark
    start = pd.Timestamp(spec["holdout_start"])
    end = pd.Timestamp(spec["holdout_end_exclusive"])

    if (open_prices.empty or open_prices.index.min() < start or open_prices.index.max() >= end):
        raise ValueError("Daily prices must lie within the frozen evaluation period.")

    fraction = float(spec["passive_fraction"])
    if not 0 <= fraction <= 1:
        raise ValueError("The passive allocation must lie between zero and one.")
    execution_dates = pd.DatetimeIndex(decision_schedule["execution_date"], name="Date")

    # Check valuation substitutions against the actual execution schedule.
    #check valuation subs against the execution schedule
    marked_open, marked_close, stale_quotes = etf_data.prepare_valuation_marks(
        open_prices,
        close_prices,
        excluded_quotes,
        execution_dates=execution_dates
    )

    settings = {key: spec[key] for key in (
            "initial_capital_gbp",
            "cost_per_side_bps",
            "cash_rate_annual")}

    #reuse existing runner with one fixed lookback
    _, trend_runs = run_lookback_sweep(
        monthly_prices.loc[monthly_prices.index < end],
        marked_open,
        marked_close,
        decision_schedule,
        lookbacks=(spec["lookback_months"],),
        **settings
    )

    passive_weights = pd.DataFrame(fraction / len(open_prices.columns), index=execution_dates, columns=open_prices.columns)
    passive_weights["CASH"] = 1 - fraction

    results = {
        "trend": trend_runs[spec["lookback_months"]],
        "passive_cash": etf_backtest.run_backtest(
            marked_open,
            marked_close,
            passive_weights,
            **settings
        )}

    for result in results.values():
        result["stale_quotes"] = stale_quotes.copy()
        held_assets = result["holdings"].drop(columns="CASH").gt(0)
        result["daily"]["stale_valuation"] = (stale_quotes & held_assets).any(axis=1)

    return results

def bootstrap_mean_excess(monthly_excess, *, block_months=3, n_bootstrap=10_000, confidence_level=0.95, seed=42):
    #estimate uncertainty around mean monthly excess returns
    if not isinstance(monthly_excess.index, pd.PeriodIndex):
        raise ValueError("Use a monthly PeriodIndex.")

    expected_months = pd.period_range(monthly_excess.index.min(), monthly_excess.index.max(), freq="M")
    if not monthly_excess.index.equals(expected_months):
        raise ValueError("Months must be consecutive, ordered and unique.")

    values = monthly_excess.to_numpy(dtype=float)
    n_months = len(values)

    for value in (block_months, n_bootstrap):
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError("Block length and sample count must be positive integers.")

    if n_months < 2 * block_months:
        raise ValueError("Provide at least two blocks of observations.")
    if not np.isfinite(values).all():
        raise ValueError("Monthly excess returns must be finite.")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie between zero and one.")

    observed_mean = values.mean()
    centred_values = values - observed_mean

    rng = np.random.default_rng(seed)
    blocks_per_sample = (n_months + block_months - 1) // block_months

    starts = rng.integers(0, n_months, size=(n_bootstrap, blocks_per_sample))
    indices = (starts[:, :, None] + np.arange(block_months)) % n_months
    indices = indices.reshape(n_bootstrap, -1)[:, :n_months]

    #centreing impose 0 mean for null distrubution
    bootstrap_errors = centred_values[indices].mean(axis=1)

    alpha = 1 - confidence_level
    lower_error, upper_error = np.quantile(bootstrap_errors, [alpha / 2, 1 - alpha / 2])
    #standard, two-sided confidence interval
    lower_bound = observed_mean - upper_error
    upper_bound = observed_mean - lower_error

    #one sided hypothesis test. h_0: mean_excess <= 0, h_1: mean_excess > 0
    p_value = (1 + np.count_nonzero(bootstrap_errors >= observed_mean)) / (n_bootstrap + 1)

    return pd.Series({
        "months": n_months,
        "mean_monthly_excess_bps": observed_mean * 10_000,
        "lower_ci_bps": lower_bound * 10_000,
        "upper_ci_bps": upper_bound * 10_000,
        "one_sided_p_value": p_value,
    })

def run_cost_sensitivity(monthly_prices, open_prices, close_prices, decision_schedule, excluded_quotes, spec, evaluation_settings):
    #rerun frozen comparison under a range of transaction costs

    summaries = {}
    runs = {}

    for cost in evaluation_settings["cost_scenarios_bps"]:
        scenario_spec = {**spec, "cost_per_side_bps": float(cost)}
        results = run_frozen_evaluation(
            monthly_prices,
            open_prices,
            close_prices,
            decision_schedule,
            excluded_quotes,
            scenario_spec
        )

        daily_returns = pd.DataFrame({name: result["daily"]["net_return"] for name, result in results.items()})
        monthly_returns = (1 + daily_returns).groupby(daily_returns.index.to_period("M")).prod()- 1
        inference = bootstrap_mean_excess(
            monthly_returns["trend"] - monthly_returns["passive_cash"],
            block_months=evaluation_settings["bootstrap_block_months"],
            n_bootstrap=evaluation_settings["bootstrap_samples"],
            confidence_level=evaluation_settings["confidence_level"],
            seed=evaluation_settings["seed"]
        )

        trend_daily = results["trend"]["daily"]
        passive_daily = results["passive_cash"]["daily"]

        summaries[float(cost)] = {
            "trend_final_gbp": trend_daily["nav_close"].iloc[-1],
            "benchmark_final_gbp": passive_daily["nav_close"].iloc[-1],
            "trend_cost_gbp": trend_daily["cost"].sum(),
            "benchmark_cost_gbp": passive_daily["cost"].sum(),
            **inference.drop("months").to_dict()
        }
        runs[float(cost)] = results
    summary = pd.DataFrame.from_dict(summaries, orient="index").rename_axis("cost_per_side_bps")

    return summary, runs