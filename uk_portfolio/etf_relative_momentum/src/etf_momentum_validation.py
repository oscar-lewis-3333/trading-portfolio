from itertools import product

import numpy as np
import pandas as pd

import etf_momentum_backtest
import etf_momentum_features
import etf_research


def run_parameter_sweep(monthly_prices, open_prices, close_prices, decision_schedule, common_eligibility, *, initial_capital_gbp, cost_per_side_bps, cash_rate_annual=0.0, lookbacks=(3, 6, 9, 12), top_ns=(1, 3, 5, 8)):
    lookbacks, top_ns = tuple(lookbacks), tuple(top_ns)

    for values in (lookbacks, top_ns):
        if (not values or len(set(values)) != len(values) or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in values)):
            raise ValueError("Grid values must be distinct positive integers.")

    if not monthly_prices.columns.equals(open_prices.columns):
        raise ValueError("Monthly and daily prices must contain the same assets.")

    if decision_schedule.empty or not decision_schedule.index.is_unique:
        raise ValueError("Provide a nonempty schedule with unique decision dates.")

    execution_dates = pd.DatetimeIndex(decision_schedule["execution_date"], name="Date")
    if (execution_dates.hasnans or not (execution_dates > decision_schedule.index).all()):
        raise ValueError("Execution must occur after each decision.")

    eligible = common_eligibility.reindex(index=decision_schedule.index, columns=monthly_prices.columns)
    if (eligible.isna().any().any() or not eligible.isin([True, False]).all().all()):
        raise ValueError("Provide complete boolean eligibility on the common dates.")

    eligible = eligible.astype(bool)

    #calculate each formation window once
    score_cache = {}
    for lookback in lookbacks:
        scores = etf_momentum_features.build_momentum_scores(monthly_prices, lookback_months=int(lookback)).loc[decision_schedule.index]
        if (eligible & scores.isna()).any().any():
            raise ValueError(f"{lookback} months: missing scores for eligible funds.")

        score_cache[lookback] = scores.where(eligible)

    runs, records = {}, []
    configurations = list(product(lookbacks, top_ns))

    for config_id, (lookback, top_n) in enumerate(configurations):
        print(f"{config_id + 1}/{len(configurations)}: {lookback} months, top {top_n}", flush=True)

        weights = etf_momentum_backtest.build_target_weights(score_cache[lookback], top_n=int(top_n))
        weights.index = execution_dates
        result = etf_momentum_backtest.run_backtest(
            open_prices,
            close_prices,
            weights,
            initial_capital_gbp=initial_capital_gbp,
            cost_per_side_bps=cost_per_side_bps,
            cash_rate_annual=cash_rate_annual
        )

        runs[config_id] = result
        returns = result["daily"]["net_return"]
        volatility = returns.std(ddof=1)

        records.append({
            "configuration_id": config_id,
            "lookback_months": lookback,
            "top_n": top_n,
            "sharpe_zero_rf": np.sqrt(252) * returns.mean() / volatility if volatility > 0 else np.nan})

    parameters = pd.DataFrame(records).set_index("configuration_id")
    metrics = etf_research.summarise_portfolios(runs, initial_capital_gbp)

    return parameters.join(metrics), runs