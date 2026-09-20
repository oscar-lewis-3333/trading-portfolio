"""Annual past-only selection followed by one continuous evaluation ledger."""
import numpy as np
import pandas as pd

from features import add_momentum_features
from selection import build_weekly_universe, build_momentum_targets
from intraday_backtest import run_intraday_ledger
from weekly_sweep import _check_protocol, run_training_sweep, rank_training_candidates


def select_annual_parameters(
    policy_panel, gbp_prices, execution_schedule, execution_references,
    protocol, progress=True,
):
    """Fit each annual choice using only its predeclared training window."""
    _check_protocol(protocol)
    rows, training_tables = [], {}
    for fold in protocol["folds"]:
        if progress:
            print(f"Fold {fold['fold']}: training ends {fold['training_end']}", flush=True)
        table, training_runs, benchmark_run = run_training_sweep(
            policy_panel, gbp_prices, execution_schedule, execution_references,
            protocol, training_end=fold["training_end"], progress=progress,
        )
        expected_days = (pd.Timestamp(fold["training_end"]) - pd.Timestamp(fold["training_start"])).days
        expected_grid = {(l, f) for l in protocol["lookback_days"] for f in protocol["top_fractions"]}
        if table.index.has_duplicates or set(table.index) != expected_grid:
            raise ValueError("Training table does not contain the frozen grid exactly once.")
        if not table["observations"].eq(expected_days).all():
            raise ValueError("Training table has the wrong observation count.")
        ranked = rank_training_candidates(table)
        lookback, fraction = ranked.index[0]
        best = ranked.iloc[0]
        rows.append({
            **fold,
            "lookback_days": int(lookback),
            "top_fraction": float(fraction),
            "training_sharpe": float(best["sharpe_zero_cash_rate"]),
            "training_benchmark_sharpe": float(best["benchmark_sharpe"]),
            "training_sharpe_difference": float(best["sharpe_difference"]),
            "training_turnover": float(best["annualised_one_way_turnover"]),
        })
        training_tables[fold["fold"]] = table
        del training_runs, benchmark_run
    choices = pd.DataFrame(rows).set_index("fold")
    for column in ["training_start", "training_end", "test_start", "test_end"]:
        choices[column] = pd.to_datetime(choices[column], utc=True)
    return choices, training_tables


def build_walk_forward_targets(policy_panel, choices, execution_schedule, protocol):
    """Select target rows by fold; retain the ordinary weekly decision calendar.

    A January parameter change applies at the first scheduled Monday. Holdings
    between that Monday and the previous rebalance are handled by the ledger.
    """
    _check_protocol(protocol)
    folds = protocol["folds"]
    if choices.index.has_duplicates or set(choices.index) != {f["fold"] for f in folds}:
        raise ValueError("Provide exactly one choice for every frozen fold.")
    for fold in folds:
        choice = choices.loc[fold["fold"]]
        for column in ["training_start", "training_end", "test_start", "test_end"]:
            if choice[column] != pd.Timestamp(fold[column], tz="UTC"):
                raise ValueError("Choice boundaries do not match the frozen training/test windows.")
        if (choice["lookback_days"] not in protocol["lookback_days"]
                or choice["top_fraction"] not in protocol["top_fractions"]):
            raise ValueError("Choice is outside the frozen parameter grid.")

    start = pd.Timestamp(protocol["oos_start"], tz="UTC")
    end = pd.Timestamp(protocol["oos_end"], tz="UTC")
    schedule = execution_schedule.loc[
        execution_schedule["day"].ge(start) & execution_schedule["day"].lt(end)
    ].copy().sort_values("day")
    decisions = pd.DatetimeIndex(schedule["day"])
    expected = pd.date_range(start, end - pd.Timedelta(days=1), freq="W-MON")
    if not decisions.equals(expected):
        raise ValueError("Evaluation schedule must retain every Monday, including cash decisions.")
    if not schedule["status"].isin(["ready", "cash_only"]).all():
        raise ValueError("Resolve the execution schedule before evaluation.")

    history = policy_panel.loc[policy_panel["timestamp"].lt(end)].copy()
    bank, benchmark = {}, None
    for lookback in sorted(set(choices["lookback_days"])):
        featured = add_momentum_features(history, lookback_days=int(lookback))
        members, summary = build_weekly_universe(featured, **protocol["universe"])
        relevant = members.loc[members["decision_at"].isin(decisions)]
        if not relevant["momentum_ready"].all():
            raise ValueError("Evaluation universe member lacks momentum history.")
        fractions = choices.loc[choices["lookback_days"].eq(lookback), "top_fraction"].unique()
        for fraction in sorted(fractions):
            _, target, matched = build_momentum_targets(members, summary, top_fraction=float(fraction))
            target, matched = target.loc[decisions], matched.loc[decisions]
            if benchmark is None:
                benchmark = matched
            elif not matched.equals(benchmark):
                raise ValueError("The matched benchmark changed across annual choices.")
            bank[(int(lookback), float(fraction))] = target

    selected = pd.DataFrame(0.0, index=decisions, columns=benchmark.columns)
    audit_rows, assigned = [], pd.Series(False, index=decisions)
    for fold in folds:
        choice = choices.loc[fold["fold"]]
        key = (int(choice["lookback_days"]), float(choice["top_fraction"]))
        mask = (decisions >= choice["test_start"]) & (decisions < choice["test_end"])
        if not mask.any() or assigned.loc[decisions[mask]].any():
            raise ValueError("Empty or overlapping fold assignment.")
        selected.loc[decisions[mask]] = bank[key].loc[decisions[mask]]
        assigned.loc[decisions[mask]] = True
        for day in decisions[mask]:
            audit_rows.append({
                "decision_at": day,
                "selection_fold": fold["fold"],
                "selection_at": choice["test_start"],
                "training_end": choice["training_end"],
                "lookback_days": key[0],
                "top_fraction": key[1],
            })
    if not assigned.all() or not np.allclose(selected.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("Target coverage or portfolio weights are incomplete.")
    audit = pd.DataFrame(audit_rows).sort_values("decision_at").reset_index(drop=True)
    audit = audit.merge(schedule[["day", "execution_at"]], left_on="decision_at", right_on="day", validate="one_to_one").drop(columns="day")
    if (audit["training_end"].gt(audit["decision_at"]).any()
            or audit["selection_at"].gt(audit["decision_at"]).any()
            or audit["decision_at"].ge(audit["execution_at"]).any()):
        raise ValueError("Training, selection, decision and execution times are inconsistent.")
    return selected, benchmark, audit


def run_walk_forward(policy_panel, gbp_prices, execution_schedule, execution_references, choices, protocol):
    #run strategy and benchmark once each, with cash/units carried across years

    selected, benchmark, audit = build_walk_forward_targets(policy_panel, choices, execution_schedule, protocol)
    start = pd.Timestamp(protocol["oos_start"], tz="UTC")
    end = pd.Timestamp(protocol["oos_end"], tz="UTC")
    prices = gbp_prices.loc[gbp_prices["timestamp"].ge(start) & gbp_prices["timestamp"].lt(end)].copy()
    if prices.empty or prices["timestamp"].max() != end - pd.Timedelta(days=1):
        raise ValueError("Evaluation history does not reach the frozen end date.")
    
    schedule = execution_schedule.loc[execution_schedule["day"].isin(selected.index)].copy()
    refs = execution_references.loc[execution_references["day"].isin(selected.index)].copy()
    runs = {}
    for name, targets in [("walk_forward", selected), ("benchmark", benchmark)]:
        runs[name] = run_intraday_ledger(prices, targets, schedule, refs, initial_cash=protocol["initial_cash_gbp"], **protocol["costs"])
    
    return runs, audit, selected, benchmark