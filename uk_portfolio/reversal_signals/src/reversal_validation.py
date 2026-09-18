import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
import reversal_backtest

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

def bootstrap_block_means(data, block_length=5, n_bootstrap=5000, seed=42):

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Provide a nonempty DataFrame.")
    if (not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans or not data.index.is_unique or not data.index.is_monotonic_increasing or not data.columns.is_unique):
        raise ValueError("Provide sorted, unique dates and unique columns.")

    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Resolve missing periods before bootstrapping.")

    n_periods, n_variants = values.shape

    if (isinstance(block_length, bool) or not isinstance(block_length, int) or not 1 <= block_length < n_periods):
        raise ValueError("Block length must be between 1 and n_periods - 1.")
    if (isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 2):
        raise ValueError("Provide at least two bootstrap repetitions.")

    rng = np.random.default_rng(seed)
    n_blocks = (n_periods + block_length - 1) // block_length
    offsets = np.arange(block_length)
    bootstrap_means = np.empty((n_bootstrap, n_variants))

    for repetition in range(n_bootstrap):
        starts = rng.integers(0, n_periods, size=n_blocks)

        #wrap blocks around end, before trimming to sample size
        indices = ((starts[:, None] + offsets) % n_periods).ravel()[:n_periods]

        #both variants receive exactly the same dates
        bootstrap_means[repetition] = values[indices].mean(axis=0)

    return pd.DataFrame(bootstrap_means, columns=data.columns).rename_axis("bootstrap_draw")

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

def infer_sweep_means(net_returns, grid, rules):
    #HAC mean tests and one holm correction for each 'element' of the grid
    if not isinstance(net_returns, pd.DataFrame) or net_returns.empty:
        raise ValueError("Provide a nonempty return matrix.")

    dates = net_returns.index
    if (not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing):
        raise ValueError("Return dates must be valid, unique and sorted.")
    if (not grid.index.is_unique or not net_returns.columns.is_unique or not net_returns.columns.equals(grid.index)):
        raise ValueError("Return columns must match the complete declared grid.")
    if not np.isfinite(net_returns.to_numpy(dtype=float)).all():
        raise ValueError("Resolve unknown returns before inference.")

    lags = rules["hac_lags"]
    alpha = rules["familywise_alpha"]

    if (isinstance(lags, bool) or not isinstance(lags, int) or not 0 <= lags < len(net_returns) - 1):
        raise ValueError("HAC lags must leave sufficient observations.")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1):
        raise ValueError("Alpha must be between zero and one.")
    if rules["alternative"] != "greater" or rules["hac_use_t"] is not False:
        raise ValueError("This helper uses one-sided normal-reference tests.")
    if rules["multiple_testing"] != "holm":
        raise ValueError("This helper requires Holm correction.")

    constant = np.ones((len(net_returns), 1))
    critical_value = norm.ppf(1 - alpha / 2)
    rows = {}

    for configuration_id in net_returns.columns:
        values = net_returns[configuration_id].to_numpy(dtype=float)

        if np.all(values == values[0]):
            raise ValueError(f"Configuration {configuration_id} has constant returns.")

        fit = OLS(values, constant).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": lags, "kernel": rules["hac_kernel"], "use_correction": rules["hac_use_correction"]},
            use_t=False)

        mean = float(fit.params[0])
        standard_error = float(fit.bse[0])

        if not np.isfinite(standard_error) or standard_error <= 0:
            raise ValueError(f"Invalid standard error for configuration {configuration_id}.")
        z_stat = mean / standard_error

        rows[configuration_id] = {
            "periods": len(values),
            "mean_bps": 10_000 * mean,
            "hac_se_bps": 10_000 * standard_error,
            "ci_lower_bps": 10_000 * (mean - critical_value * standard_error),
            "ci_upper_bps": 10_000 * (mean + critical_value * standard_error),
            "z_stat": z_stat,
            "p_one_sided": norm.sf(z_stat)
        }

    out = pd.DataFrame.from_dict(rows, orient="index").rename_axis("configuration_id")

    #holm-correct entire family before displaying any subset/sorting anything
    rejected, adjusted_p, _, _ = multipletests(out["p_one_sided"].to_numpy(), alpha=alpha, method="holm", is_sorted=False)
    out["p_holm"] = adjusted_p
    out["reject_holm"] = rejected

    return out


def attribute_stock_pnl(result, market, initial_capital_gbp):

    #attribute a complete backtest (starting all cash) by ticker
    daily = result["daily"]
    holdings = result["holdings"]
    fills = result["orders"].loc[result["orders"]["status"].eq("assumed_fill")]

    summary = fills.groupby("ticker").agg(net_trade_cash_gbp=("cash_change_gbp", "sum"), cost_gbp=("cost_gbp", "sum"), traded_notional_gbp=("notional_gbp", "sum"))

    #close holdings, including zero holdings on dates outside a position
    units = holdings["units"].unstack("ticker").reindex(index=daily.index, columns=summary.index).fillna(0.0)

    #overnight owners recieve dividends, including exit date
    overnight_units = units.shift(1, fill_value=0.0)
    dividends_per_share = market["dividend_gbp"].unstack("ticker").reindex(index=daily.index, columns=summary.index)

    dividend_cash = (overnight_units * dividends_per_share).where(overnight_units.gt(0), 0.0)

    if not np.isfinite(dividend_cash.to_numpy()).all():
        raise ValueError("Missing dividend information for a held stock.")

    summary["dividends_gbp"] = dividend_cash.sum(axis=0)

    #include positions still held, rather than treating their purchase as a loss
    final_holdings = holdings.loc[holdings.index.get_level_values("Date") == daily.index[-1]]
    summary["ending_value_gbp"] = final_holdings["value_gbp"].groupby(level="ticker").sum().reindex(summary.index, fill_value=0.0)

    summary["net_pnl_gbp"] = summary["net_trade_cash_gbp"] + summary["dividends_gbp"] + summary["ending_value_gbp"]

    expected_pnl = daily["equity_gbp"].iloc[-1] - initial_capital_gbp
    if not np.isclose(summary["net_pnl_gbp"].sum(), expected_pnl, rtol=1e-10, atol=1e-6):
        raise ValueError("Stock contributions do not reconcile to portfolio P&L.")

    return summary.sort_values("net_pnl_gbp", ascending=False)