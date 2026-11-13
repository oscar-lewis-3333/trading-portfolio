import numpy as np
import pandas as pd

from rebalance_schedule import build_development_grid
from reversal_backtest import run_development_ledger


def summarise_development_ledger(ledger, protocol):
    #include every development day and the initial capital in drawdown
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    if not start < end <= pd.Timestamp(protocol["holdout_start"], tz="UTC"):
        raise ValueError("Invalid development boundaries.")
    
    expected = pd.date_range(start, end, freq="D", inclusive="left")
    if len(expected) < 2 or not ledger.index.equals(expected):
        raise ValueError("Ledger must contain every development day exactly once.")
    if not ledger["valuation_at"].eq(expected + pd.Timedelta(days=1)).all():
        raise ValueError("Daily returns must be available at the following midnight.")
    
    returns = ledger["net_return"].to_numpy(dtype=float)
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Invalid daily returns.")
    
    growth = np.r_[1., np.cumprod(1. + returns)]
    if not np.isfinite(growth).all() or (growth <= 0).any():
        raise ValueError("Invalid compounded wealth.")
    
    notional = ledger["traded_notional_gbp"]
    costs = ledger[["fee_gbp", "other_cost_gbp"]]
    if (not np.isfinite(notional).all() or notional.lt(0).any() or not np.isfinite(costs.to_numpy()).all() or costs.lt(0).any().any()):
        raise ValueError("Costs and traded notionals must be finite and non-negative.")
    
    traded = ledger.loc[notional.gt(0)]
    equity = traded["equity_before_rebalance_gbp"]
    if not np.isfinite(equity).all() or equity.le(0).any():
        raise ValueError("Turnover requires positive pre-trade equity.")
    
    n = len(returns)
    volatility = returns.std(ddof=1)
    turnover = (traded["traded_notional_gbp"] / equity).sum()
    return {
        "observations": n,
        "total_return": growth[-1] - 1,
        "cagr": growth[-1] ** (365 / n) - 1,
        "annualised_volatility": volatility * np.sqrt(365),
        "sharpe_zero_cash_rate": returns.mean() / volatility * np.sqrt(365) if volatility > 0 else np.nan,
        "max_drawdown": (growth / np.maximum.accumulate(growth) - 1).min(),
        "annualised_one_way_turnover": turnover * 365 / n,
        "total_cost_gbp": costs.to_numpy().sum(),
        "max_rebalance_error_gbp": ledger["accounting_error_gbp"].abs().max(),
        "max_daily_pnl_error_gbp": ledger["daily_pnl_error_gbp"].abs().max()
    }


def run_development_sweep(gbp_prices, sweep_targets, development_grid, execution_schedule, execution_references, protocol, initial_cash=1000., fee_bps=9., other_cost_bps=3.5, progress=True):
    #run all 60 configs plus 4 matched frequency benchmarks
    expected_grid = build_development_grid()
    columns = ["lookback_days", "top_fraction", "rebalance_days"]
    if (not development_grid.index.is_unique or set(development_grid.index) != set(expected_grid.index)
        or set(sweep_targets) != set(expected_grid.index) or not set(columns).issubset(development_grid.columns)):
        raise ValueError("Supply the complete 60-configuration grid and targets.")
    
    grid = development_grid.loc[expected_grid.index, columns]
    if not np.array_equal(grid.to_numpy(dtype=float), expected_grid[columns].to_numpy(dtype=float)):
        raise ValueError("Configuration IDs and parameters disagree with the fixed grid.")

    #check all benchmark equivalences before spending time on simulations
    benchmark_targets = {}
    for config_id, config in grid.iterrows():
        pair = sweep_targets[config_id]
        if not {"strategy", "benchmark"}.issubset(pair):
            raise ValueError(f"Missing paired targets: {config_id}")
        
        interval = int(config["rebalance_days"])
        if interval not in benchmark_targets:
            benchmark_targets[interval] = pair["benchmark"]
        elif not pair["benchmark"].equals(benchmark_targets[interval]):
            raise ValueError("Benchmarks differ within a rebalance frequency.")

    def simulate(weights, interval):
        ledger, _, _ = run_development_ledger(
            gbp_prices, weights, execution_schedule, execution_references,
            protocol, rebalance_days=interval, initial_cash=initial_cash,
            fee_bps=fee_bps, other_cost_bps=other_cost_bps)
        return ledger

    benchmark_ledgers, benchmark_metrics = {}, {}
    for interval, weights in sorted(benchmark_targets.items()):
        ledger = simulate(weights, interval)
        benchmark_ledgers[interval] = ledger
        benchmark_metrics[interval] = summarise_development_ledger(ledger, protocol)
    if progress:
        print("Completed 4 matched-frequency benchmarks", flush=True)

    rows, strategy_returns, matched_returns = [], {}, {}
    for number, (config_id, config) in enumerate(grid.iterrows(), 1):
        interval = int(config["rebalance_days"])
        ledger = simulate(sweep_targets[config_id]["strategy"], interval)
        metrics = summarise_development_ledger(ledger, protocol)
        benchmark = benchmark_ledgers[interval]
        bm = benchmark_metrics[interval]
        if not ledger.index.equals(benchmark.index):
            raise ValueError("Strategy and benchmark daily returns are not aligned.")
        
        daily = ledger["net_return"].copy()
        paired = benchmark["net_return"].copy()
        metrics.update({
            "config_id": config_id,
            "lookback_days": int(config["lookback_days"]),
            "top_fraction": float(config["top_fraction"]),
            "rebalance_days": interval,
            "mean_daily_net_excess_bps": 10000 * (daily - paired).mean(),
            "benchmark_total_return": bm["total_return"],
            "benchmark_sharpe": bm["sharpe_zero_cash_rate"],
            "sharpe_difference": metrics["sharpe_zero_cash_rate"] - bm["sharpe_zero_cash_rate"],
        })
        rows.append(metrics)
        strategy_returns[config_id], matched_returns[config_id] = daily, paired
        if progress:
            print(f"Completed {number}/60 development configurations: {config_id}", flush=True)

    summary = pd.DataFrame(rows).set_index("config_id")
    summary.attrs["costs"] = {"fee_bps": fee_bps, "other_cost_bps": other_cost_bps}
    summary.attrs["initial_cash_gbp"] = initial_cash
    return summary, pd.DataFrame(strategy_returns), pd.DataFrame(matched_returns)
