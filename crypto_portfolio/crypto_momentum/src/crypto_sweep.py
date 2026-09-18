import pandas as pd
import math

import crypto_features
import crypto_portfolio
import crypto_backtest
import crypto_reporting


def run_lookback_sweep(prices, symbols, lookbacks, windows, initial_cash=1_000.0, fee_bps=9.0, other_cost_bps=3.5):

    #compare different lookbacks over the same evaluation dates
    lookbacks = tuple(lookbacks)
    symbols = tuple(symbols)

    if (not lookbacks or any(isinstance(days, bool) or not isinstance(days, int) or days < 1 for days in lookbacks) or len(set(lookbacks)) != len(lookbacks)):
        raise ValueError("Lookbacks must be unique positive integers.")
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Provide a nonempty, unique asset universe.")
    if windows.empty:
        raise ValueError("Provide evaluation windows.")

    evaluation_start = windows["test_start"].min()
    evaluation_end = windows["test_end"].max()

    #retain earlier signal history, but exclude later prices
    history = prices.loc[prices["timestamp"] < evaluation_end].copy()

    runs = {}
    summary_rows = []
    annual_rows = []

    for days in lookbacks:
        features = crypto_features.add_momentum_features(prices=history, lookback_days=days)
        targets = crypto_portfolio.build_weekly_targets(features=features, symbols=symbols, execution_end=evaluation_end, asset_weight=1.0 / len(symbols))
        ledger, trades = crypto_backtest.run_daily_backtest(
            prices=history,
            targets=targets,
            symbols=symbols,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            other_cost_bps=other_cost_bps)

        summary = crypto_reporting.summarise_backtest_period(ledger=ledger, start=evaluation_start, end=evaluation_end)
        summary_rows.append({"lookback_days": days, **summary.to_dict()})
        for fold, window in windows.iterrows():
            annual = crypto_reporting.summarise_backtest_period(ledger=ledger, start=window["test_start"], end=window["test_end"])
            annual_rows.append({
                "lookback_days": days,
                "fold": fold,
                "year": window["test_start"].year,
                **annual.to_dict()
            })
        runs[days] = {
            "targets": targets,
            "ledger": ledger,
            "trades": trades}

    summary_table = (pd.DataFrame(summary_rows).set_index("lookback_days"))
    annual_table = (pd.DataFrame(annual_rows).set_index(["lookback_days", "year"]))

    return summary_table, annual_table, runs

def run_cost_sensitivity(prices, targets_by_strategy, symbols, windows, round_trip_costs=(25.0, 40.0, 60.0), initial_cash=1_000.0, fee_bps=9.0):
    #rerun fixed target schedules under different transaction costs
    symbols = tuple(symbols)
    costs = tuple(float(value) for value in round_trip_costs)
    fee_bps = float(fee_bps)

    if not math.isfinite(fee_bps) or fee_bps < 0:
        raise ValueError("Commission must be finite and nonnegative.")
    if (not costs or len(set(costs)) != len(costs) or any(not math.isfinite(cost) or cost < 2.0 * fee_bps for cost in costs)):
        raise ValueError("Costs must be unique, finite and cover both commissions.")
    if windows.empty or not targets_by_strategy:
        raise ValueError("Provide evaluation windows and target schedules.")

    start = windows["test_start"].min()
    end = windows["test_end"].max()
    history = prices.loc[prices["timestamp"] < end].copy()
    rows = []
    runs = {}

    for round_trip_bps in costs:
        other_bps = round_trip_bps / 2.0 - fee_bps

        for name, targets in targets_by_strategy.items():
            schedule = targets.loc[targets["execution_at"] < end].copy()
            ledger, trades = crypto_backtest.run_daily_backtest(
                prices=history,
                targets=schedule,
                symbols=symbols,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                other_cost_bps=other_bps
            )

            summary = crypto_reporting.summarise_backtest_period(ledger=ledger, start=start, end=end)
            rows.append({
                "round_trip_bps": round_trip_bps,
                "strategy": name,
                **summary.to_dict()
            })
            runs[(round_trip_bps, name)] = {"ledger": ledger, "trades": trades}

    results = pd.DataFrame(rows).set_index(["round_trip_bps", "strategy"])
    return results, runs