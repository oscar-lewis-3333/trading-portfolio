import hashlib
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import crypto_features
import crypto_portfolio
import crypto_backtest
import crypto_reporting
import crypto_inference
import crypto_validation
import crypto_sweep


def run_primary_holdout(prices, protocol_path, prior_ledgers):
    #evaluate frozen primary comparison with continuous holdings
    record = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    protocol = record["protocol"]

    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if digest != record["protocol_sha256"]:
        raise ValueError("The saved protocol hash does not match.")

    start = pd.Timestamp(protocol["holdout_start"], tz="UTC")
    end = pd.Timestamp(protocol["holdout_end_exclusive"], tz="UTC")
    symbols = protocol["symbols"]
    strategy_name = protocol["primary_strategy"]["name"]
    benchmark_name = protocol["primary_benchmark"]["name"]
    history = prices.loc[prices["timestamp"] < end].copy()

    features = crypto_features.add_momentum_features(prices=history, lookback_days=protocol["primary_strategy"]["lookback_days"])
    strategy_targets = crypto_portfolio.build_weekly_targets(
        features=features,
        symbols=symbols,
        execution_end=end,
        asset_weight=protocol["primary_strategy"]["weight_per_positive_asset"])

    #preserve benchmarks original initialisation date
    benchmark_start = prior_ledgers[benchmark_name].index.min()
    benchmark_schedule = strategy_targets.loc[strategy_targets["execution_at"] >= benchmark_start]

    benchmark_targets = crypto_portfolio.build_constant_mix_targets(
        weekly_targets=benchmark_schedule,
        symbols=symbols,
        crypto_weight=1.0 - protocol["primary_benchmark"]["cash_weight"],
    )

    schedules = {
        strategy_name: strategy_targets,
        benchmark_name: benchmark_targets
    }
    fee = protocol["fee_bps_per_side"]
    other = protocol["primary_round_trip_cost_bps"] / 2.0 - fee

    ledgers = {}
    trades = {}
    summaries = {}
    returns = {}
    curves = {}

    for name, targets in schedules.items():
        ledger, trade_log = crypto_backtest.run_daily_backtest(
            prices=history,
            targets=targets,
            symbols=symbols,
            initial_cash=1_000.0,
            fee_bps=fee,
            other_cost_bps=other)

        #extending history must not change development period accounting
        previous = ledger.loc[ledger.index < start]
        reference = prior_ledgers[name].loc[prior_ledgers[name].index < start]
        pd.testing.assert_frame_equal(previous, reference, check_exact=False, rtol=1e-10, atol=1e-10)

        summaries[name] = crypto_reporting.summarise_backtest_period(ledger=ledger, start=start, end=end)
        period = ledger.loc[(ledger.index >= start) & (ledger.index < end)]
        returns[name] = period["net_return"]
        previous_close = ledger.loc[start - pd.Timedelta(days=1), "equity_close"]

        #rebase for presentation only. holdings not reset
        curve = period["equity_close"] / previous_close * 1_000.0
        curve.index = pd.DatetimeIndex(period["valuation_at"])
        curves[name] = pd.concat([pd.Series([1_000.0], index=pd.DatetimeIndex([start])), curve])

        ledgers[name] = ledger
        trades[name] = trade_log

    performance = pd.DataFrame(summaries).T
    paired_returns = pd.DataFrame(returns)
    equity = pd.DataFrame(curves)

    settings = protocol["bootstrap"]
    block_sizes = [settings["block_days"], *settings["sensitivity_block_days"]]
    inference_rows = []
    bootstrap_draws = {}

    for block_days in block_sizes:
        summary, draws = crypto_inference.paired_sharpe_bootstrap(
            strategy_returns=paired_returns[strategy_name],
            benchmark_returns=paired_returns[benchmark_name],
            block_days=block_days,
            repetitions=settings["repetitions"],
            seed=settings["seed"],
            confidence_level=settings["confidence_level"],
            annualisation_days=protocol["sharpe_annualisation_days"])
        inference_rows.append(summary)
        bootstrap_draws[block_days] = draws

    inference = pd.DataFrame(inference_rows).set_index("block_days")
    primary_interval = inference.loc[settings["block_days"]]
    criterion_met = bool(primary_interval["ci_lower"] > 0 and performance.loc[strategy_name, "total_return"] > 0)

    return {
        "protocol": protocol,
        "performance": performance,
        "inference": inference,
        "primary_criterion_met": criterion_met,
        "equity": equity,
        "returns": paired_returns,
        "ledgers": ledgers,
        "trades": trades,
        "bootstrap_draws": bootstrap_draws
    }


def plot_holdout_equity(equity):
    #plot holdout equity and drawdown from holdout peak
    drawdown = equity / equity.cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    equity.plot(ax=axes[0])
    axes[0].set_title("Holdout: 2024–2025")
    axes[0].set_ylabel("Equity rebased to £1,000")

    drawdown.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_xlabel("Valuation time")

    for axis in axes:
        axis.grid(alpha=0.25)

    fig.tight_layout()
    plt.close(fig)
    return fig

def run_secondary_holdout(prices, primary_result, prior_ledgers, prior_selections):
    #run predeclared secondary portfolios and fixed-choice cost stresses
    
    protocol = primary_result["protocol"]
    start = pd.Timestamp(protocol["holdout_start"], tz="UTC")
    end = pd.Timestamp(protocol["holdout_end_exclusive"], tz="UTC")
    symbols = protocol["symbols"]
    strategy_name = protocol["primary_strategy"]["name"]
    benchmark_name = protocol["primary_benchmark"]["name"]
    annual_name = "annual_past_sharpe_selection"
    weekly_name = "weekly_equal_weight"
    fee = protocol["fee_bps_per_side"]
    base_cost = protocol["primary_round_trip_cost_bps"]
    other = base_cost / 2.0 - fee
    history = prices.loc[prices["timestamp"] < end].copy()
    windows = crypto_validation.build_annual_evaluation_windows(
        data_start=history["timestamp"].min(),
        first_test_start=prior_selections["test_start"].min(),
        evaluation_end=end,
    )
    windows["calibration_start"] = pd.Timestamp(protocol["annual_selection"]["training_start"], tz="UTC")
    #candidate histories are used for past-only parameter selection
    candidate_runs = {}

    for days in protocol["annual_selection"]["lookbacks"]:
        features = crypto_features.add_momentum_features(prices=history, lookback_days=days)
        targets = crypto_portfolio.build_weekly_targets(
            features=features,
            symbols=symbols,
            execution_end=end,
            asset_weight=(protocol["primary_strategy"]["weight_per_positive_asset"])
        )
        ledger, trades = crypto_backtest.run_daily_backtest(
            prices=history,
            targets=targets,
            symbols=symbols,
            initial_cash=1_000.0,
            fee_bps=fee,
            other_cost_bps=other
        )
        candidate_runs[days] = {
            "targets": targets,
            "ledger": ledger,
            "trades": trades
        }

    selections, scores = crypto_validation.select_lookbacks_by_past_sharpe(sweep_runs=candidate_runs, windows=windows)

    #extending the data must not alter earlier selections
    pd.testing.assert_frame_equal(
        selections.loc[prior_selections.index],
        prior_selections,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10)

    fixed_targets = candidate_runs[protocol["primary_strategy"]["lookback_days"]]["targets"]
    annual_schedule = fixed_targets.loc[fixed_targets["execution_at"] >= prior_ledgers[annual_name].index.min()]
    annual_targets = (
        crypto_portfolio.build_selected_momentum_targets(
            weekly_targets=annual_schedule,
            symbols=symbols,
            sweep_runs=candidate_runs,
            selections=selections
        ))

    benchmark_schedule = fixed_targets.loc[fixed_targets["execution_at"] >= primary_result["ledgers"][benchmark_name].index.min()]
    benchmark_targets = crypto_portfolio.build_constant_mix_targets(
        weekly_targets=benchmark_schedule,
        symbols=symbols,
        crypto_weight=(1.0 - protocol["primary_benchmark"]["cash_weight"]))

    weekly_schedule = fixed_targets.loc[fixed_targets["execution_at"] >= prior_ledgers[weekly_name].index.min()]
    weekly_targets = crypto_portfolio.build_benchmark_targets(weekly_targets=weekly_schedule, symbols=symbols, mode="weekly")
    targets_by_strategy = {
        strategy_name: fixed_targets,
        benchmark_name: benchmark_targets,
        annual_name: annual_targets,
        weekly_name: weekly_targets
    }

    holdout_windows = windows.loc[windows["test_start"] >= start].copy()
    cost_summary, cost_runs = crypto_sweep.run_cost_sensitivity(
        prices=history,
        targets_by_strategy=targets_by_strategy,
        symbols=symbols,
        windows=holdout_windows,
        round_trip_costs=(base_cost, *protocol["secondary_round_trip_costs_bps"]),
        initial_cash=1_000.0,
        fee_bps=fee)

    #verify primary replay and secondary devlopment histories
    references = {**primary_result["ledgers"], **prior_ledgers}

    for name, reference in references.items():
        actual = cost_runs[(base_cost, name)]["ledger"]
        pd.testing.assert_frame_equal(actual.reindex(reference.index), reference, check_exact=False, rtol=1e-10, atol=1e-10)

    curves = {}
    annual_rows = []

    for name in targets_by_strategy:
        ledger = cost_runs[(base_cost, name)]["ledger"]
        period = ledger.loc[(ledger.index >= start) & (ledger.index < end)]
        base_equity = ledger.loc[start - pd.Timedelta(days=1), "equity_close"]
        curve = period["equity_close"] / base_equity * 1_000.0
        curve.index = pd.DatetimeIndex(period["valuation_at"])
        curves[name] = pd.concat([pd.Series([1_000.0], index=pd.DatetimeIndex([start])), curve])

        for _, window in holdout_windows.iterrows():
            summary = crypto_reporting.summarise_backtest_period(ledger=ledger, start=window["test_start"], end=window["test_end"])
            annual_rows.append({
                "year": window["test_start"].year,
                "strategy": name,
                **summary.to_dict()
            })

    return {
        "holdout_selections": selections.loc[selections["test_start"] >= start],
        "selection_scores": scores,
        "cost_summary": cost_summary,
        "annual_summary": pd.DataFrame(annual_rows).set_index(["year", "strategy"]),
        "equity": pd.DataFrame(curves),
        "targets": targets_by_strategy,
        "runs": cost_runs
    }