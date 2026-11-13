import numpy as np
import pandas as pd
from . import reversal_backtest

def select_reversal_candidates(feature_panel, score_column, top_frac=0.2, min_eligible=10):
    if score_column not in {"raw_reversal_score", "residual_reversal_score"}:
        raise ValueError("Choose a raw or residual reversal score.")
    if (isinstance(top_frac, bool) or not isinstance(top_frac, (int, float)) or not 0 < top_frac <= 1):
        raise ValueError("Fraction must be between zero and one.")

    if (isinstance(min_eligible, bool) or not isinstance(min_eligible, int) or min_eligible < 1):
        raise ValueError("Minimum eligibility must be a positive integer.")
    required = {score_column, "eligible"}
    missing = required.difference(feature_panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if (feature_panel.empty or list(feature_panel.index.names) != ["Date", "ticker"] or not feature_panel.index.is_unique):
        raise ValueError("Provide a nonempty, unique Date-ticker index.")

    #alphabetic ordering gives deterministic tie-breaker handling
    out = feature_panel.reset_index().sort_values(["Date", "ticker"])

    if out[["Date", "ticker"]].isna().to_numpy().any():
        raise ValueError("Dates and ticker labels cannot be missing.")
    if (not pd.api.types.is_bool_dtype(out["eligible"]) or out["eligible"].isna().any()):
        raise ValueError("Eligibility must contain nonmissing booleans.")

    out["selection_score"] = out[score_column].astype(float)
    if not np.isfinite(out.loc[out["eligible"], "selection_score"]).all():
        raise ValueError("Eligible stocks must have finite scores.")

    out["eligible_count"] = out.groupby("Date")["eligible"].transform("sum").astype(int)
    out["target_slots"] = np.ceil(top_frac* out["eligible_count"]).astype(int)
    out.loc[out["eligible_count"] < min_eligible, "target_slots"] = 0


    out["candidate_rank"] = out["selection_score"].where(out["eligible"]).groupby(out["Date"]).rank(method="first", ascending=False)
    out["selected"] = out["eligible"] & out["selection_score"].gt(0) & out["candidate_rank"].le(out["target_slots"])

    return out.set_index(["Date", "ticker"]).sort_index()


def build_rebalance_calendar(schedule, anchor, rebalance_sessions=5):

    if (isinstance(rebalance_sessions, bool) or not isinstance(rebalance_sessions, int) or rebalance_sessions < 1):
        raise ValueError("Rebalance sessions must be a positive integer.")

    sessions = schedule.index
    if (not isinstance(sessions, pd.DatetimeIndex) or sessions.empty or sessions.hasnans or not sessions.is_unique or not sessions.is_monotonic_increasing):
        raise ValueError("Provide a nonempty, valid, sorted session index.")
    anchor = pd.Timestamp(anchor)
    if anchor not in sessions:
        raise ValueError("Anchor must be an exchange session in the schedule.")

    anchor_position = sessions.get_loc(anchor)
    rebalance_dates = sessions[anchor_position::rebalance_sessions]

    return pd.DataFrame({"is_rebalance": sessions.isin(rebalance_dates)}, index=sessions,).rename_axis("Date")


def attach_forward_outcomes(decisions, outcomes, horizon=5):
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")

    if (decisions.empty or list(decisions.index.names) != ["Date", "ticker"] or not decisions.index.is_unique):
        raise ValueError("Provide nonempty, unique Date-ticker decisions.")
    if not {"signal_time", "selected"}.issubset(decisions.columns):
        raise ValueError("Decisions must contain signal_time and selected.")

    outcome_columns = [
        "horizon_sessions",
        "entry_time",
        "exit_time",
        "entry_open",
        "exit_open",
        "dividends_earned",
        "forward_return",
        "outcome_observed",
    ]

    required = {"ticker", "signal_time", *outcome_columns}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"Missing outcome columns: {sorted(missing)}")

    if outcomes.index.name != "Date":
        raise ValueError("Outcomes must have a Date index.")
    if not outcomes["horizon_sessions"].eq(horizon).all():
        raise ValueError("Outcome horizons do not match the requested horizon.")
    outcome_table = outcomes.reset_index().set_index(["Date", "ticker"], verify_integrity=True).sort_index()

    #a label row must exist, even when its future return was not known
    if not decisions.index.isin(outcome_table.index).all():
        raise ValueError("Some decisions have no corresponding outcome row.")
    matched_signal_times = outcome_table["signal_time"].reindex(decisions.index)
    if not decisions["signal_time"].eq(matched_signal_times).all():
        raise ValueError("Decision and outcome signal timestamps differ.")

    return decisions.join(outcome_table[outcome_columns], how='left', validate="one_to_one")


def evaluate_execution_configuration(feature_panel, decision_dates, market, schedule, *, top_frac=0.2, holding_sessions=5, min_eligible=10, initial_capital_gbp=10_000, cost_per_side_bps=10,):
    #evaluating one reversal configuration, with its matched benchmark
    if (not isinstance(decision_dates, pd.DatetimeIndex) or decision_dates.empty or decision_dates.hasnans or not decision_dates.is_unique or not decision_dates.is_monotonic_increasing):
        raise ValueError("Provide nonempty, sorted, unique decision dates.")

    features = feature_panel.loc[feature_panel.index.get_level_values("Date").isin(decision_dates)]
    decisions = select_reversal_candidates(features, score_column="raw_reversal_score", top_frac=top_frac, min_eligible=min_eligible)
    actual_dates = decisions.index.get_level_values("Date").unique()
    if not actual_dates.equals(decision_dates):
        raise ValueError("Features do not cover every requested decision date.")


    #unfilled slots remain in cash
    slots = decisions["target_slots"]
    decisions["target_weight"] = decisions["selected"].astype(float).div(slots.where(slots.gt(0))).fillna(0.0)


    #spread same planned exposure across all eligible stocks
    benchmark = decisions[["signal_time", "eligible"]].copy()
    eligible_count = benchmark["eligible"].groupby(level="Date").transform("sum")
    exposure = decisions["target_weight"].groupby(level="Date").transform("sum")
    benchmark["target_weight"] = benchmark["eligible"].astype(float).mul(exposure).div(eligible_count.where(eligible_count.gt(0))).fillna(0.0)
    expected_dates = schedule.loc[decision_dates[0]:].index
    results = {}

    for name, targets in [("strategy", decisions), ("benchmark", benchmark)]:
        result = reversal_backtest.run_execution_backtest(
            decisions=targets,
            market=market,
            schedule=schedule,
            initial_capital_gbp=initial_capital_gbp,
            holding_sessions=holding_sessions,
            cost_per_side_bps=cost_per_side_bps,
            net_orders=True
        )
        if not result["daily"].index.equals(expected_dates):
            raise ValueError("Backtest changed the expected valuation dates.")

        results[name] = result

    return {
        "decisions": decisions,
        "benchmark_decisions": benchmark,
        "strategy": results["strategy"],
        "benchmark": results["benchmark"]
    }
