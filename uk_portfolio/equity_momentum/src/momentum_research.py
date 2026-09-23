"""Compact replay and reporting for the completed raw-momentum notebook.

The existing feature, ledger, selection and inference functions remain the
authority. Candidate histories are computed once; each evaluation starts cash.
"""

from contextlib import redirect_stdout
import io

import pandas as pd

import momentum_backtest
import momentum_features
import momentum_sweep
import momentum_validation
import momentum_walk_forward


def sessions_in_period(schedule, bounds):
    # Keep period boundaries tied to execution dates, with an exclusive end.
    dates = schedule.index
    selected = dates[(dates >= bounds["start"]) & (dates < bounds["end_exclusive"])]
    if selected.empty:
        raise ValueError("The requested period has no exchange sessions.")
    return selected


def build_candidate_history(prepared, protocol, periods, *, progress=True):
    # Replay every fixed candidate once, retaining warm-up data for features.
    if (protocol["signal"] != "raw_momentum_score"
            or protocol["rebalance_frequency"] != "monthly"
            or protocol["cash_interest_annual"] != 0
            or protocol["initial_state"] != "cash"
            or not protocol["carry_holdings_across_years"]):
        raise ValueError("This replay supports the saved raw-momentum rules only.")
    if (protocol["validation"] != periods["validation"]
            or protocol["reserved_holdout"] != periods["holdout"]):
        raise ValueError("Research periods and evaluation protocol disagree.")

    end = periods["holdout"]["end_exclusive"]
    schedule = prepared["schedule"].loc[lambda x: x.index < end]
    prices = prepared["prices"].loc[:schedule.index[-1]]
    grid = pd.DataFrame(protocol["parameter_grid"]).set_index("configuration_id")
    rules = protocol["liquidity"]
    liquidity = momentum_features.build_liquidity_features(
        prices, schedule, lookback=rules["lookback_sessions"])
    liquidity["liquidity_eligible"] = (
        liquidity["liquidity_history_complete"]
        & liquidity["median_traded_value_gbp"].ge(rules["min_median_traded_value_gbp"])
        & liquidity["positive_volume_fraction"].ge(rules["min_positive_volume_fraction"])
        & prices["positive_reported_volume"].eq(True)
        & prices["close_reference_usable"].eq(True)
    )
    panels, common_eligible = momentum_features.build_momentum_sweep_features(
        prices, schedule, liquidity, grid)
    bounds = {"start": periods["development"]["start"], "end_exclusive": end}
    sessions = sessions_in_period(schedule, bounds)
    calendar = momentum_backtest.build_monthly_rebalance_calendar(schedule)
    rebalances = calendar.loc[calendar["execution_date"].isin(sessions)]

    arguments = dict(
        market=prepared["market"].loc[:sessions[-1]], feature_panels=panels,
        parameter_grid=grid, rebalance_calendar=rebalances, sessions=sessions,
        initial_capital_gbp=protocol["initial_capital_gbp"], **protocol["execution"])
    if progress:
        sweep = momentum_sweep.run_momentum_sweep(**arguments)
    else:
        with redirect_stdout(io.StringIO()):
            sweep = momentum_sweep.run_momentum_sweep(**arguments)

    eligible_counts = common_eligible.groupby(level="Date").sum()
    return sweep, rebalances.join(eligible_counts.rename("eligible_stocks"))


def summarise_portfolios(daily_by_name, initial_capital_gbp):
    # All supplied ledgers must start at the stated capital, including entry costs.
    summary = pd.DataFrame({
        name: momentum_sweep.summarise_momentum_backtest(
            daily, initial_capital_gbp=initial_capital_gbp)
        for name, daily in daily_by_name.items()
    }).T
    summary["total_return_pct"] = 100 * (
        summary["final_equity_gbp"] / initial_capital_gbp - 1)
    return summary


def development_summary(sweep, bounds, initial_capital_gbp):
    # Summarise only the development prefix of the continuous candidate ledgers.
    sessions = sessions_in_period(sweep["benchmark"]["daily"], bounds)
    if sessions[0] != sweep["benchmark"]["daily"].index[0]:
        raise ValueError("Development must begin at the candidate histories' start.")
    daily = {key: run["daily"].loc[sessions]
             for key, run in sweep["backtests"].items()}
    benchmark = sweep["benchmark"]["daily"].loc[sessions]
    metrics = summarise_portfolios(daily, initial_capital_gbp)
    benchmark_summary = summarise_portfolios({"Benchmark": benchmark}, initial_capital_gbp)
    metrics["cagr_gap_pp"] = (
        metrics["net_cagr_pct"] - benchmark_summary.loc["Benchmark", "net_cagr_pct"])
    summary = sweep["parameter_grid"].join(metrics, validate="one_to_one")
    returns = pd.DataFrame({key: value["net_return"] for key, value in daily.items()})
    returns = returns.rename_axis(index="Date", columns="configuration_id")
    annual = pd.DataFrame({key: momentum_sweep.annual_net_returns(value)
                           for key, value in daily.items()}).T.mul(100)
    annual_gap = annual.sub(momentum_sweep.annual_net_returns(benchmark).mul(100), axis=1)
    return dict(summary=summary, benchmark_summary=benchmark_summary, daily=daily,
                benchmark_daily=benchmark, net_returns=returns, annual_gap_pp=annual_gap)


def run_walk_forward_period(sweep, market, schedule, protocol, bounds):
    # Select using past returns, then replay targets in one ledger per portfolio.
    dates = sessions_in_period(schedule, bounds)
    grid = pd.DataFrame(protocol["parameter_grid"]).set_index("configuration_id")
    pd.testing.assert_frame_equal(grid, sweep["parameter_grid"])
    returns = pd.DataFrame({key: run["daily"]["net_return"]
                            for key, run in sweep["backtests"].items()})
    returns = returns.rename_axis(index="Date", columns="configuration_id")
    folds = momentum_walk_forward.build_annual_walk_forward_folds(
        returns.loc[:dates[-1]], schedule.loc[:dates[-1]],
        training_years=protocol["walk_forward"]["training_years"])
    folds = folds.loc[(folds["test_start"] >= dates[0]) & (folds["test_end"] <= dates[-1])]
    if (folds.empty or folds["test_start"].min() != dates[0]
            or folds["test_end"].max() != dates[-1]):
        raise ValueError("Evaluation dates must be covered by complete annual folds.")
    choices = momentum_walk_forward.select_walk_forward_configurations(
        returns.loc[:folds["train_end"].max()], folds, grid, protocol["walk_forward"])
    targets, decisions = momentum_walk_forward.build_walk_forward_targets(choices, sweep)
    benchmark_targets = sweep["benchmark"]["targets"].loc[targets.index].copy()
    backtests = {
        name: momentum_backtest.run_momentum_backtest(
            market, weights, dates, initial_capital_gbp=protocol["initial_capital_gbp"],
            **protocol["execution"])
        for name, weights in {"Walk-forward": targets, "Benchmark": benchmark_targets}.items()
    }
    daily = {name: run["daily"] for name, run in backtests.items()}
    annual = pd.DataFrame({name: momentum_sweep.annual_net_returns(value)
                           for name, value in daily.items()}).mul(100)
    annual["gap_pp"] = annual["Walk-forward"] - annual["Benchmark"]
    return dict(
        choices=choices, decisions=decisions, targets=targets, backtests=backtests,
        equity=pd.DataFrame({name: value["equity_gbp"] for name, value in daily.items()}),
        net_returns=pd.DataFrame({name: value["net_return"] for name, value in daily.items()}),
        summary=summarise_portfolios(daily, protocol["initial_capital_gbp"]), annual=annual)


def test_net_excess(net_returns, benchmark_returns, rules):
    # Preserve paired dates and the original test family at every block length.
    if not net_returns.index.equals(benchmark_returns.index):
        raise ValueError("Strategy and benchmark dates must match exactly.")
    excess = net_returns.sub(benchmark_returns, axis="index")
    results = {
        block: momentum_validation.bootstrap_sweep_excess(
            excess, block_length=block, n_bootstrap=rules["n_bootstrap"],
            alpha=rules["alpha"], seed=rules["seed"])
        for block in [rules["primary_block_length"], *rules["sensitivity_block_lengths"]]
    }
    return pd.concat(results, names=["block_length", "configuration_id"]).sort_index()
