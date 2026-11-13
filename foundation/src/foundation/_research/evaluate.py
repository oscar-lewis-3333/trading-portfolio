import numpy as np
import pandas as pd

from .contracts import CASH, validate_schedule, validate_targets
from .costs import daily_cash_rate, per_asset_rates, per_side_rate

DEFAULT_CASH_RATE = 0.0375


def _execution_plan(executions, decisions, days, assets):
    #execute the trade on the execution day using the information on the decision day.

    if not isinstance(executions, pd.DataFrame) or "execution_day" not in executions.columns:
        raise ValueError("executions must be a DataFrame with an execution_day column.")
    if not executions.index.equals(decisions):
        raise ValueError("executions must be indexed by exactly the target decision stamps.")
    
    missing = set(assets).difference(executions.columns)
    if missing:
        raise ValueError(f"executions lack fill returns for: {sorted(missing)}")

    execution_days = pd.DatetimeIndex(executions["execution_day"])
    if execution_days.tz is None or str(execution_days.tz) != "UTC" or (execution_days != execution_days.normalize()).any():
        raise ValueError("Execution days must be UTC midnights.")
    if (execution_days < decisions).any():
        raise ValueError("A decision cannot execute before it is made.")
    if not execution_days.is_monotonic_increasing or execution_days.has_duplicates:
        raise ValueError("Execution days must be unique and increasing.")
    if len(decisions) > 1 and (execution_days[:-1] >= decisions[1:]).any():
        raise ValueError("Each decision must execute before the next decision is made.")
    if not execution_days.isin(days).all():
        raise ValueError("Every execution day needs a return observation.")

    fills = executions[assets].astype(float)
    return {decision: (day, fills.loc[decision]) for decision, day in zip(decisions, execution_days)}


def run_portfolio(targets, returns, cost_mode="pessimistic", cash_rate=DEFAULT_CASH_RATE, periods_per_year=365, executions=None, cost_table=None):
    #one function that runs the portfolio from start to finish, with inputted weights and identical signals
    
    validate_targets(targets)
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a DataFrame.")
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.tz is None:
        raise ValueError("Return timestamps must be UTC timezone-aware.")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("Return timestamps must be unique and sorted.")

    assets = [column for column in targets.columns if column != CASH]
    unknown = set(assets).difference(returns.columns)
    if unknown:
        raise ValueError(f"Targets reference assets with no return series: {sorted(unknown)}")

    decisions = targets.index
    grid = returns.index[(returns.index >= decisions.min())]
    if grid.empty:
        raise ValueError("No return observations on or after the first decision.")

    #if measured, those costs are used, else the default flat option
    rates = per_asset_rates(assets, cost_table) if cost_table is not None else None
    rate = per_side_rate(cost_mode)
    cash_daily = daily_cash_rate(cash_rate, periods_per_year)

    plan = _execution_plan(executions, decisions, returns.index, assets) if executions is not None else None
    execute_on = {day: decision for decision, (day, _) in plan.items()} if plan is not None else {}

    held = pd.Series(0.0, index=assets + [CASH], dtype=float)
    held[CASH] = 1.0

    net = []
    gross = []
    traded = []
    costs = []

    for stamp in grid:
        day = returns.loc[stamp, assets].astype(float)
        turnover = 0.0
        pre_growth = 1.0

        if plan is None and stamp in decisions:
            target = targets.loc[stamp].reindex(held.index).fillna(0.0)
            trades = (target[assets] - held[assets]).abs()
            turnover = float(trades.sum())
            held = target
            day_from_fill = day

        elif stamp in execute_on:
            _, fill = plan[execute_on[stamp]]
            target = targets.loc[execute_on[stamp]].reindex(held.index).fillna(0.0)

            involved = held[assets].gt(0) | target[assets].gt(0)
            unfilled = involved & ~np.isfinite(fill.to_numpy(dtype=float))
            if unfilled.any():
                raise ValueError(f"Missing fill for traded assets on {stamp}: {sorted(fill.index[unfilled])}")

            #held weights move from the previous close to the fill, then trade
            moved = held[assets] * (1.0 + fill.fillna(0.0))
            pre_growth = float(moved.sum() + held[CASH])
            drifted = moved / pre_growth
            trades = (target[assets] - drifted).abs()
            turnover = float(trades.sum())
            held = target
            day_from_fill = (1.0 + day) / (1.0 + fill) - 1.0
        else:
            day_from_fill = day
            trades = None

        invested = held[assets]

        #missing returns with allocated capital is not a zero, but a data issue
        unusable = invested.gt(0) & ~np.isfinite(day_from_fill.to_numpy(dtype=float))
        if unusable.any():
            raise ValueError(f"Missing return for held assets on {stamp}: {sorted(invested.index[unusable])}")

        post_growth = float((invested * (1.0 + day_from_fill.fillna(0.0))).sum() + held[CASH] * (1.0 + cash_daily))
        if rates is not None and trades is not None:
            cost_fraction = float((trades * rates).sum())
        else:
            cost_fraction = rate * turnover
        gross_return = pre_growth * post_growth - 1.0
        net_return = pre_growth * (1.0 - cost_fraction) * post_growth - 1.0

        #weights are not fixed, and drift between rebalancing
        grown = invested * (1.0 + day_from_fill.fillna(0.0))
        cash_grown = held[CASH] * (1.0 + cash_daily)
        total = float(grown.sum() + cash_grown)
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Portfolio value became non-positive on {stamp}.")
        held = pd.concat([grown / total, pd.Series({CASH: cash_grown / total})])

        net.append(net_return)
        gross.append(gross_return)
        traded.append(turnover)
        costs.append(cost_fraction)

    return pd.DataFrame({"net_return": net,
        "gross_return": gross,
        "one_way_turnover": traded,
        "cost_fraction": costs}, index=grid)


def estimate_beta(signal_returns, benchmark_returns, train_end):
    #fit beta on training data, then fix.

    stamp = pd.Timestamp(train_end)
    if stamp.tzinfo is None:
        raise ValueError("train_end must be timezone-aware.")

    paired = pd.concat([signal_returns.rename("signal"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    training = paired.loc[paired.index < stamp]
    if len(training) < 60:
        raise ValueError("Beta needs at least 60 paired training observations.")

    variance = float(training["benchmark"].var(ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("The benchmark has no usable variance in the training window.")

    return float(training.cov().loc["signal", "benchmark"] / variance)


def active_returns(signal_returns, benchmark_returns, beta):
    #what the portfolio layer optimises over

    if isinstance(beta, bool) or not np.isfinite(beta):
        raise ValueError("beta must be a finite number.")

    paired = pd.concat([signal_returns.rename("signal"), benchmark_returns.rename("benchmark")], axis=1).dropna()
    if paired.empty:
        raise ValueError("No overlapping observations between signal and benchmark.")

    return (paired["signal"] - beta * paired["benchmark"]).rename("active_return")


def evaluate_signal(signal_id, panel, schedule, benchmark_targets, returns, train_end, params=None, cost_mode="pessimistic", cash_rate=DEFAULT_CASH_RATE, periods_per_year=365, cost_table=None):
    #one signal against its benchmark, from start to finish

    from .metrics import summary
    from .registry import build_targets, check_no_lookahead

    validate_schedule(schedule)
    targets = build_targets(signal_id, panel, schedule, params)
    leakage = check_no_lookahead(signal_id, panel, schedule, params)

    signal_ledger = run_portfolio(targets, returns, cost_mode, cash_rate, periods_per_year, cost_table=cost_table)
    benchmark_ledger = run_portfolio(benchmark_targets, returns, cost_mode, cash_rate, periods_per_year, cost_table=cost_table)

    beta = estimate_beta(signal_ledger["net_return"], benchmark_ledger["net_return"], train_end)
    active = active_returns(signal_ledger["net_return"], benchmark_ledger["net_return"], beta)

    return {"signal_id": signal_id,
        "params": dict(params or {}),
        "cost_mode": cost_mode if cost_table is None else f"per_pair:{cost_table.attrs.get('source', 'table')}",
        "beta": beta,
        "targets": targets,
        "ledger": signal_ledger,
        "active_return": active,
        "no_lookahead": leakage,
        "total": summary(signal_ledger["net_return"], cash_rate, periods_per_year),
        "active": summary(active, 0.0, periods_per_year),
        "annualised_one_way_turnover": float(signal_ledger["one_way_turnover"].sum() * periods_per_year / len(signal_ledger))}
