import numpy as np
import pandas as pd

from bitcoin_sweep import BITCOIN_LOOKBACKS, run_bitcoin_training_sweep
from weekly_sweep import _check_protocol
from bitcoin_filter import build_bitcoin_filter, apply_bitcoin_filter
from intraday_backtest import run_intraday_ledger
from weekly_walk_forward import build_walk_forward_targets as build_coin_walk_forward_targets


def rank_bitcoin_filters(training_table):
    #rank BTC filters by net Sharpe, then turnover, then lookback
    frame = training_table.copy()

    required = {
        "btc_lookback_days",
        "sharpe_zero_cash_rate",
        "annualised_one_way_turnover"
    }

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    expected = {0, *BITCOIN_LOOKBACKS}
    if (frame.index.has_duplicates or frame["btc_lookback_days"].duplicated().any() or set(frame["btc_lookback_days"]) != expected):
        raise ValueError("Expected the complete sweep grid exactly once.")

    #zero is the unfiltered comparison, not a selectable BTC lookback
    frame = frame.loc[frame["btc_lookback_days"].isin(BITCOIN_LOOKBACKS)].copy()

    score_columns = ["sharpe_zero_cash_rate", "annualised_one_way_turnover"]
    valid = np.isfinite(frame[score_columns].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No Bitcoin filter has a valid training score.")

    frame["selection_score"] = frame["sharpe_zero_cash_rate"].round(12)

    return frame.sort_values(["selection_score",
            "annualised_one_way_turnover",
            "btc_lookback_days"
        ], ascending=[False, True, True], kind="mergesort")


def select_annual_bitcoin_filters(policy_panel, gbp_prices, execution_schedule, execution_references, base_protocol):
    #select each years filter using only preceding training period
    _check_protocol(base_protocol)

    rows = []
    training_tables = {}

    for fold in base_protocol["folds"]:
        print(f"\nSelecting for {fold['test_start']}: training ends {fold['training_end']}", flush=True)

        table, training_runs, filter_audits = run_bitcoin_training_sweep(policy_panel,
            gbp_prices,
            execution_schedule,
            execution_references,
            base_protocol,
            training_end=fold["training_end"])

        expected_days = (pd.Timestamp(fold["training_end"]) - pd.Timestamp(fold["training_start"])).days
        if not table["observations"].eq(expected_days).all():
            raise ValueError("Training observation counts are incorrect.")

        ranked = rank_bitcoin_filters(table)
        best = ranked.iloc[0]

        rows.append({**fold,
            "btc_lookback_days": int(best["btc_lookback_days"]),
            "training_sharpe": float(best["sharpe_zero_cash_rate"]),
            "training_return": float(best["total_return"]),
            "training_max_drawdown": float(best["max_drawdown"]),
            "training_turnover": float(best["annualised_one_way_turnover"])
        })

        training_tables[fold["fold"]] = table
        del training_runs, filter_audits

    choices = pd.DataFrame(rows).set_index("fold")
    for column in ("training_start", "training_end", "test_start", "test_end"):
        choices[column] = pd.to_datetime(choices[column], utc=True)

    return choices, training_tables

def build_bitcoin_walk_forward_targets(policy_panel, choices, execution_schedule, base_protocol):
    #apply each annual BTC at the existing monday decisions
    if "btc_lookback_days" not in choices.columns:
        raise ValueError("Annual choices must include btc_lookback_days.")
    if not choices["btc_lookback_days"].isin(BITCOIN_LOOKBACKS).all():
        raise ValueError("Annual choices must use the Bitcoin sweep grid.")

    #keep strategy fixed
    coin_choices = choices.copy()
    coin_choices["lookback_days"] = 90
    coin_choices["top_fraction"] = 0.10
    momentum, equal_weight, decision_audit = build_coin_walk_forward_targets(policy_panel, coin_choices, execution_schedule, base_protocol)

    #preserve warm-up history, but use fold's selected BTC lookback
    audit_parts = []

    for fold in base_protocol["folds"]:
        choice = choices.loc[fold["fold"]]

        decisions = momentum.index[(momentum.index >= choice["test_start"]) & (momentum.index < choice["test_end"])]
        history = policy_panel.loc[policy_panel["timestamp"].lt(choice["test_end"])].copy()
        audit_parts.append(build_bitcoin_filter(history, decisions, lookback_days=int(choice["btc_lookback_days"])))

    bitcoin_audit = pd.concat(audit_parts).sort_index()
    if not bitcoin_audit.index.equals(momentum.index):
        raise ValueError("Bitcoin filters do not cover every decision exactly once.")

    targets = {
        "filtered_momentum": apply_bitcoin_filter(momentum, bitcoin_audit),
        "unfiltered_momentum": momentum,
        "filtered_equal_weight": apply_bitcoin_filter(equal_weight, bitcoin_audit), 
        "unfiltered_equal_weight": equal_weight}
    decision_audit = decision_audit.rename(columns={"lookback_days": "coin_lookback_days"}).merge(bitcoin_audit, left_on="decision_at", right_index=True, validate="one_to_one")

    return targets, decision_audit


def run_bitcoin_walk_forward(policy_panel, gbp_prices, execution_schedule, execution_references, choices, base_protocol):
    #run each portfolio continuously, including transaction costs
    targets, decision_audit = build_bitcoin_walk_forward_targets(policy_panel, choices, execution_schedule, base_protocol)

    start = pd.Timestamp(base_protocol["oos_start"], tz="UTC")
    end = pd.Timestamp(base_protocol["oos_end"], tz="UTC")
    prices = gbp_prices.loc[gbp_prices["timestamp"].ge(start) & gbp_prices["timestamp"].lt(end)].copy()
    if prices.empty or prices["timestamp"].max() != end - pd.Timedelta(days=1):
        raise ValueError("Evaluation prices do not reach the complete end date.")

    decisions = targets["filtered_momentum"].index
    schedule = execution_schedule.loc[execution_schedule["day"].isin(decisions)].copy()
    references = execution_references.loc[execution_references["day"].isin(decisions)].copy()

    runs = {}

    #one ledger per portfolio
    for name, portfolio_targets in targets.items():
        runs[name] = run_intraday_ledger(prices,
            portfolio_targets,
            schedule,
            references,
            initial_cash=base_protocol["initial_cash_gbp"],
            **base_protocol["costs"])

    return runs, decision_audit, targets