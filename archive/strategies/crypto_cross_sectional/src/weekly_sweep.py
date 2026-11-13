"""Frozen weekly search protocol and training-only parameter comparisons.

No downloads. Training functions discard later observations before computing
features or simulating portfolios. They do not construct an out-of-sample curve.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import add_momentum_features
from selection import build_weekly_universe, build_momentum_targets
from intraday_backtest import run_intraday_ledger


def weekly_protocol():
    return {
        "version": "weekly_cross_sectional_v1",
        "lookback_days": [7, 14, 30, 60, 90, 180],
        "top_fractions": [0.10, 0.20, 0.30],
        "universe": {"max_assets": 30, "min_assets": 20},
        "eligibility": {"history_days": 180, "liquidity_days": 30, "min_median_dollar_volume": 1000000.0},
        "scope": "current Revolut UK GBP candidate catalogue; meme_exclusions_v1",
        "signal": "calendar-day close momentum within uninterrupted history segments",
        "weights": "ceil(fraction * universe_size), equal weights; negative winners permitted",
        "benchmark": "equal weight in the same eligible universe; identical decision/execution schedule",
        "cash": "GBP; zero interest; cash-only decisions retained",
        "decision_frequency": "Monday 00:00 UTC",
        "execution": "first common fresh-reference minute from 00:05 through 01:00; fixed midnight targets",
        "execution_max_reference_age_minutes": 5,
        "costs": {"fee_bps": 9.0, "other_cost_bps": 3.5},
        "initial_cash_gbp": 1000.0,
        "simulation_start": "2021-12-20",
        "training_score_start": "2022-01-01",
        "initialisation_note": "December 2021 trades establish holdings; their costs affect wealth but are outside the training return window",
        "annualisation_days": 365,
        "window_convention": "candle_start >= start and < end; final close is available at end",
        "folds": [
            {"fold": 1, "training_start": "2022-01-01", "training_end": "2024-01-01",
             "test_start": "2024-01-01", "test_end": "2025-01-01"},
            {"fold": 2, "training_start": "2022-01-01", "training_end": "2025-01-01",
             "test_start": "2025-01-01", "test_end": "2026-01-01"},
            {"fold": 3, "training_start": "2022-01-01", "training_end": "2026-01-01",
             "test_start": "2026-01-01", "test_end": "2026-09-01"},
        ],
        "selection": "highest after-cost strategy-minus-benchmark Sharpe, rounded to 12 decimals",
        "tie_breaks": ["lower annualised one-way turnover", "shorter lookback", "smaller top fraction"],
        "undefined_sharpe": "ineligible; stop if no configuration is eligible",
        "oos_construction": "select at each test_start; apply at first scheduled Monday; carry units and cash continuously; rerun one ledger, never splice candidate returns",
        "oos_start": "2024-01-01",
        "oos_end": "2026-09-01",
        "oos_initialisation": "strategy and benchmark each start from GBP 1000 cash on 2024-01-01; include entry costs",
        "primary_comparison": "stitched walk-forward strategy versus matched benchmark after costs",
        "inference": {"statistic": "Sharpe difference", "method": "paired circular moving-block bootstrap, percentile interval",
                      "primary_block_days": 28, "sensitivity_block_days": [14, 56],
                      "repetitions": 10000, "seed": 20260919, "confidence_level": 0.95,
                      "positive_result": "primary confidence interval lower endpoint above zero"},
        "cost_stress_round_trip_bps": [40, 60],
        "cost_stress_selection": "retain selections made under base costs",
        "limitations": "current-catalogue survivorship; execution convention informed by availability audits across the sample; Coinbase last-trade prices and ECB FX are proxies for GBP venue execution"
    }


def _check_protocol(protocol):
    if protocol != weekly_protocol():
        raise ValueError("Protocol differs from weekly_cross_sectional_v1; do not silently change the experiment.")


def freeze_protocol(path):
    path = Path(path)
    expected = weekly_protocol()
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        _check_protocol(actual)
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(expected, file, indent=2, sort_keys=True)
        file.write("\n")
    return expected


def summarise_training_ledger(ledger, start, end):
    #score complete daily observations, including initial baseline in drawdown

    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    window = ledger.loc[(ledger.index >= start) & (ledger.index < end)].copy()
    expected = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")

    if len(expected) < 2 or not window.index.equals(expected):
        raise ValueError("Training window must contain every calendar day exactly once.")
    if not window["valuation_at"].eq(window.index + pd.Timedelta(days=1)).all():
        raise ValueError("Daily valuation timing is inconsistent.")
    returns = window["net_return"].to_numpy(dtype=float)

    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Invalid daily returns.")
    growth = np.r_[1.0, np.cumprod(1 + returns)]

    if not np.isfinite(growth).all() or (growth <= 0).any():
        raise ValueError("Invalid compounded wealth.")
    n = len(returns)
    std = returns.std(ddof=1)
    traded = window.loc[window["traded_notional_gbp"].gt(0)]
    denominators = traded["equity_before_rebalance_gbp"].to_numpy(dtype=float)

    if not np.isfinite(denominators).all() or (denominators <= 0).any():
        raise ValueError("Traded notional requires positive pre-trade equity.")
    turnover = (traded["traded_notional_gbp"] / traded["equity_before_rebalance_gbp"]).sum()

    return {
        "observations": n,
        "total_return": growth[-1] - 1,
        "cagr": growth[-1] ** (365 / n) - 1,
        "annualised_volatility": std * np.sqrt(365),
        "sharpe_zero_cash_rate": returns.mean() / std * np.sqrt(365) if std > 0 else np.nan,
        "max_drawdown": (growth / np.maximum.accumulate(growth) - 1).min(),
        "mean_crypto_weight": window["crypto_weight_close"].mean(),
        "annualised_one_way_turnover": float(turnover * 365 / n),
        "total_cost_gbp": (window["fee_gbp"] + window["other_cost_gbp"]).sum()
    }


def rank_training_candidates(summary):
    #fixed selection rule.

    frame = summary.reset_index().copy()
    needed = ["sharpe_difference", "annualised_one_way_turnover"]
    frame = frame.loc[np.isfinite(frame[needed].to_numpy(dtype=float)).all(axis=1)].copy()

    if frame.empty:
        raise ValueError("No candidate has an eligible training score.")
    frame["selection_score"] = frame["sharpe_difference"].round(12)
    return frame.sort_values(["selection_score", "annualised_one_way_turnover", "lookback_days", "top_fraction"], ascending=[False, True, True, True], kind="mergesort").set_index(["lookback_days", "top_fraction"])


def run_training_sweep(policy_panel, gbp_prices, execution_schedule, execution_references, protocol, training_end="2024-01-01", progress=True):
    #run frozen 18 configs using data before training end

    _check_protocol(protocol)
    if training_end not in [fold["training_end"] for fold in protocol["folds"]]:
        raise ValueError("training_end must be one of the frozen fold boundaries.")
    start = pd.Timestamp(protocol["simulation_start"], tz="UTC")
    end = pd.Timestamp(training_end, tz="UTC")

    #cut off later data before feature building or portfolio simulation
    history = policy_panel.loc[policy_panel["timestamp"].lt(end)].copy()
    prices = gbp_prices.loc[gbp_prices["timestamp"].ge(start) & gbp_prices["timestamp"].lt(end)].copy()
    schedule = execution_schedule.loc[execution_schedule["day"].ge(start) & execution_schedule["day"].lt(end)].copy().sort_values("day")
    refs = execution_references.loc[execution_references["day"].isin(schedule["day"])].copy()

    if prices.empty or prices["timestamp"].max() != end - pd.Timedelta(days=1):
        raise ValueError("Price history does not reach the complete training cutoff.")
    decisions = pd.DatetimeIndex(schedule["day"])
    expected_decisions = pd.date_range(start, end - pd.Timedelta(days=1), freq="W-MON")

    if not decisions.equals(expected_decisions):
        raise ValueError("Training schedule omits a weekly decision or starts too late.")

    scores, runs = [], {}
    benchmark_run = benchmark_targets = None
    for lookback in protocol["lookback_days"]:
        featured = add_momentum_features(history, lookback_days=lookback)
        members, summary = build_weekly_universe(featured, **protocol["universe"])
    
        if not members["momentum_ready"].all():
            raise ValueError("Universe member lacks momentum history; do not drop it.")
        
        for fraction in protocol["top_fractions"]:
            _, targets, benchmark = build_momentum_targets(members, summary, top_fraction=fraction)
            targets, benchmark = targets.loc[decisions], benchmark.loc[decisions]
            
            if benchmark_targets is None:
                benchmark_targets = benchmark
                required = benchmark.drop(columns="CASH").gt(0)
                required = required | required.shift(1, fill_value=False)

                pairs = required.rename_axis(index="day", columns="product_id").stack()
                expected_pairs = pairs.loc[pairs].index
                actual_pairs = pd.MultiIndex.from_frame(refs[["day", "product_id"]])

                if (actual_pairs.has_duplicates or len(expected_pairs.difference(actual_pairs)) or len(actual_pairs.difference(expected_pairs))):
                    raise ValueError("Execution references do not match the training universe and exits.")
                benchmark_run = run_intraday_ledger(prices, benchmark, schedule, refs, initial_cash=protocol["initial_cash_gbp"], **protocol["costs"])
                benchmark_metrics = summarise_training_ledger(benchmark_run[0], protocol["training_score_start"], training_end)
            elif not benchmark.equals(benchmark_targets):
                raise ValueError("The matched benchmark changed across configurations.")
            
            run = run_intraday_ledger(prices, targets, schedule, refs, initial_cash=protocol["initial_cash_gbp"], **protocol["costs"])
            metrics = summarise_training_ledger(run[0], protocol["training_score_start"], training_end)
            metrics.update({
                "lookback_days": lookback, "top_fraction": fraction,
                "benchmark_sharpe": benchmark_metrics["sharpe_zero_cash_rate"],
                "sharpe_difference": metrics["sharpe_zero_cash_rate"] - benchmark_metrics["sharpe_zero_cash_rate"]
            })
            scores.append(metrics)
            runs[(lookback, fraction)] = run
            if progress:
                print(f"Completed {len(scores)}/18 training configurations", flush=True)
    table = pd.DataFrame(scores).set_index(["lookback_days", "top_fraction"])

    return table, runs, benchmark_run