#replay weekly targets at the different cost scenarios

import pandas as pd

from intraday_backtest import run_intraday_ledger
from weekly_sweep import _check_protocol, summarise_training_ledger


def run_cost_stress(gbp_prices, selected_targets, benchmark_targets, execution_schedule, execution_references, protocol):
    #change execution costs only. retain all other variables/elements of the strategy

    _check_protocol(protocol)
    start = pd.Timestamp(protocol["oos_start"], tz="UTC")
    end = pd.Timestamp(protocol["oos_end"], tz="UTC")
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    mondays = pd.date_range(start, end - pd.Timedelta(days=1), freq="W-MON")
    for targets in (selected_targets, benchmark_targets):
        if not targets.index.equals(mondays):
            raise ValueError("Targets must contain all evaluation Mondays exactly once.")

    prices = gbp_prices.loc[gbp_prices["timestamp"].ge(start) & gbp_prices["timestamp"].lt(end)].copy()
    schedule = execution_schedule.loc[execution_schedule["day"].isin(mondays)].copy()
    references = execution_references.loc[execution_references["day"].isin(mondays)].copy()
    fee_bps = protocol["costs"]["fee_bps"]
    base_round_trip_bps = 2 * (fee_bps + protocol["costs"]["other_cost_bps"])
    scenarios = [base_round_trip_bps, *protocol["cost_stress_round_trip_bps"]]

    rows, runs = [], {}
    for round_trip_bps in scenarios:
        metrics = {}
        for name, targets in (("walk_forward", selected_targets), ("benchmark", benchmark_targets)):
            run = run_intraday_ledger(prices, targets, schedule, references, initial_cash=protocol["initial_cash_gbp"],
                fee_bps=fee_bps, other_cost_bps=round_trip_bps / 2 - fee_bps,)
            ledger = run[0]
            if (not ledger.index.equals(days) or not ledger["valuation_at"].eq(ledger.index + pd.Timedelta(days=1)).all()):
                raise ValueError("Ledger must cover the complete evaluation dates and valuations.")
            
            runs[(float(round_trip_bps), name)] = run
            metrics[name] = summarise_training_ledger(ledger, protocol["oos_start"], protocol["oos_end"])

        benchmark_sharpe = metrics["benchmark"]["sharpe_zero_cash_rate"]
        for name, values in metrics.items():
            rows.append({"round_trip_bps": float(round_trip_bps),
                "strategy": name,
                **values,
                "benchmark_sharpe": benchmark_sharpe,
                "sharpe_difference": values["sharpe_zero_cash_rate"] - benchmark_sharpe})

    summary = pd.DataFrame(rows).set_index(["round_trip_bps", "strategy"])
    return summary, runs
