import numpy as np
import pandas as pd

import momentum_backtest


def run_momentum_sweep(market, feature_panels, parameter_grid, rebalance_calendar, sessions, *, initial_capital_gbp, cost_per_side_bps, max_missing_valuation_sessions=0):
    #run each config and one shared eligible universe benchmark
    if parameter_grid.empty or not parameter_grid.index.is_unique:
        raise ValueError("Expected a non-empty grid with unique configuration IDs.")

    pairs = list(parameter_grid[["formation_sessions", "skip_sessions"]].drop_duplicates().itertuples(index=False, name=None))
    reference = feature_panels[pairs[0]]

    #comparisons require identical available stocks on each signal date
    for pair in pairs:
        panel = feature_panels[pair]
        if (not panel.index.equals(reference.index) or not panel["eligible"].equals(reference["eligible"])):
            raise ValueError("Feature panels must share the same eligibility.")

    def run_one(panel, top_frac):
        targets, selection = momentum_backtest.build_momentum_targets(panel, rebalance_calendar, top_frac=top_frac)
        result = momentum_backtest.run_momentum_backtest(
            market=market,
            targets=targets,
            sessions=sessions,
            initial_capital_gbp=initial_capital_gbp,
            cost_per_side_bps=cost_per_side_bps,
            max_missing_valuation_sessions=max_missing_valuation_sessions)

        result["targets"] = targets
        result["selection"] = selection
        return result

    print("Running shared benchmark...", flush=True)
    benchmark = run_one(reference, top_frac=1.0)
    backtests = {}
    columns = ["formation_sessions", "skip_sessions", "top_frac"]

    for number, row in enumerate(parameter_grid[columns].itertuples(name=None),start=1):
        configuration_id, formation, skip, top_frac = row
        print(f"Running configuration {configuration_id} ({number}/{len(parameter_grid)})...", flush=True)

        try:
            backtests[configuration_id] = run_one(feature_panels[(formation, skip)], top_frac=top_frac)
        except ValueError as error:
            raise ValueError(f"Configuration {configuration_id}: {error}") from error

    return {
        "backtests": backtests,
        "benchmark": benchmark,
        "parameter_grid": parameter_grid.copy()
    }

def summarise_momentum_backtest(daily, *, initial_capital_gbp, sessions_per_year=252):
    #summarise a net equity curve and its trading activity
    if len(daily) < 2:
        raise ValueError("At least two sessions are required.")

    if (not np.isfinite(initial_capital_gbp) or initial_capital_gbp <= 0 or not np.isfinite(sessions_per_year) or sessions_per_year <= 0):
        raise ValueError("Capital and annualisation factor must be positive.")

    columns = [
        "equity_gbp",
        "net_return",
        "traded_value_gbp",
        "total_cost_gbp",
        "cash_weight"
    ]
    if not np.isfinite(daily[columns].to_numpy()).all():
        raise ValueError("Performance inputs contain missing or invalid values.")

    equity = daily["equity_gbp"]
    returns = daily["net_return"]
    if equity.le(0).any():
        raise ValueError("Equity must remain positive.")

    previous_equity = equity.shift(1, fill_value=initial_capital_gbp)
    if not np.allclose(returns, equity / previous_equity - 1, rtol=1e-10, atol=1e-12):
        raise ValueError("Returns do not reconcile with equity.")

    years = len(daily) / sessions_per_year
    growth = equity.iloc[-1] / initial_capital_gbp
    daily_volatility = returns.std(ddof=1)

    #include starting capital when identifying earlier equity peaks.
    running_peak = equity.cummax().clip(lower=initial_capital_gbp)
    drawdown = equity / running_peak - 1

    #count both purchases and sales, relative to previous closing equity
    turnover = daily["traded_value_gbp"] / previous_equity

    return pd.Series({
        "final_equity_gbp": equity.iloc[-1],
        "net_cagr_pct": 100 * (growth ** (1 / years) - 1),
        "volatility_pct": (100 * daily_volatility * np.sqrt(sessions_per_year)),
        "sharpe_zero_rf": (np.sqrt(sessions_per_year) * returns.mean() / daily_volatility if daily_volatility > 0 else np.nan),
        "max_drawdown_pct": 100 * drawdown.min(),
        "annual_two_way_turnover": turnover.sum() / years,
        "total_cost_gbp": daily["total_cost_gbp"].sum(),
        "mean_cash_pct": 100 * daily["cash_weight"].mean()
    })

def annual_net_returns(daily):
    #compound daily net returns within each calendar year
    returns = daily["net_return"]
    if (returns.empty or not isinstance(returns.index, pd.DatetimeIndex) or not returns.index.is_unique or not returns.index.is_monotonic_increasing):
        raise ValueError("Expected returns with unique, sorted dates.")
    if (not np.isfinite(returns.to_numpy()).all() or returns.le(-1).any()):
        raise ValueError("Returns must be finite and greater than -100%.")

    return (1 + returns).groupby(returns.index.year).prod().sub(1).rename_axis("year")