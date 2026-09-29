"""Offline preparation and reporting for the completed ETF momentum study.

The original signal, execution, selection and bootstrap functions remain in
their existing modules. This file consolidates the research notebook.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import etf_data
import etf_research
import etf_momentum_backtest as backtest
import etf_momentum_features as features
import etf_momentum_validation as validation
import momentum_validation
import momentum_walk_forward


LABELS = {
    "walk_forward": "Walk-forward",
    "equal_weight": "Monthly equal weight",
    "buy_and_hold": "Buy and hold",
    "momentum": "Original baseline",
}


def study_protocol():
    """Return the settings used in the archived research notebook."""
    return {
        "data_start": "2000-01-01",
        "data_end": "2026-09-01",
        "analysis_start": "2015-01-01",
        "baseline_lookback_months": 12,
        "baseline_top_n": 3,
        "lookbacks": [3, 6, 9, 12],
        "top_ns": [1, 3, 5, 8],
        "backtest": {
            "initial_capital_gbp": 10000.0,
            "cost_per_side_bps": 10.0,
            "cash_rate_annual": 0.0,
        },
        "periods": {
            "development": ["2016-02-01", "2022-01-01"],
            "validation": ["2022-01-01", "2024-01-01"],
            "final_evaluation": ["2024-01-01", "2026-09-01"],
        },
        "walk_forward": {
            "training_years": 3,
            "selection_metric": "sharpe_zero_rf",
            "refit_frequency": "annual",
            "tie_break": "lowest_configuration_id",
        },
        "first_complete_training_year": "2017-01-01",
        "inference": {
            "block_length": 63, "n_bootstrap": 10000, "alpha": 0.05, "seed": 42,
        },
        "quote_exclusions": {
            "IJPN": ["2025-10-24"],
            "WLDS": ["2025-10-24"],
            "CMOP": ["2025-10-24"],
        },
    }


def _read_cache(project_dir, symbol, version, protocol):
    stem = "{}_{}_{}_{}".format(
        symbol, protocol["data_start"], protocol["data_end"], version
    )
    folder = Path(project_dir) / "data" / version
    price_path = folder / (stem + ".csv")
    metadata_path = folder / (stem + ".json")
    for path in (price_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError("Required saved study file is missing: {}".format(path))
    prices = pd.read_csv(price_path, index_col="Date", parse_dates=True)
    metadata = json.loads(metadata_path.read_text())
    if metadata["symbol"] != symbol:
        raise ValueError("Cached symbol does not match {}".format(symbol))
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Cached dates must be ordered and unique.")
    return prices, metadata


def prepare_cached_data(project_dir, protocol):
    """Load saved raw/repaired pairs; retain pre-listing gaps and reviewed exclusions."""
    universe = pd.read_csv(
        Path(project_dir) / "data/candidate_universe_v2.csv",
        dtype=str, keep_default_na=False,
    ).set_index("ticker", verify_integrity=True)
    expected = (
        "IUSA", "ISF", "IEUX", "IJPN", "IEEM", "CPJ1", "MIDD", "WLDS",
        "IGLS", "IGLT", "GLTL", "INXG", "IGTM", "XGSG", "SLXX", "CRHG",
        "GHYS", "IWDP", "SGLN", "CMOP",
    )
    if tuple(universe.index) != expected or not universe["isin"].is_unique:
        raise ValueError("The frozen 20-instrument universe has changed.")
    if (not universe["live_eligible"].eq("true").all()
            or not universe["api_currency"].isin(["GBP", "GBX"]).all()):
        raise ValueError("Unexpected catalogue eligibility or quote currency.")
    divisors = pd.to_numeric(universe["unit_divisor_to_gbp"], errors="raise")
    if not divisors.eq(universe["api_currency"].map({"GBP": 1, "GBX": 100})).all():
        raise ValueError("Catalogue quote units are inconsistent.")

    schedule = etf_data.build_lse_schedule(
        protocol["analysis_start"], protocol["data_end"]
    )
    sessions = schedule.index
    aligned, audit_rows = {}, []
    price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for ticker, symbol in universe["yf_symbol"].items():
        raw, raw_meta = _read_cache(project_dir, symbol, "raw", protocol)
        prices, metadata = _read_cache(project_dir, symbol, "repaired", protocol)
        if metadata["currency"] != "GBP":
            raise ValueError("{}: repaired quotes must already be in GBP.".format(ticker))
        frame = prices.loc[protocol["analysis_start"]:].copy()
        traded = frame.index[frame["Volume"].gt(0)]
        if traded.empty:
            raise ValueError("{} has no positive-volume observations.".format(ticker))
        first_date = traded[0]
        frame = frame.loc[first_date:].copy()
        if len(frame.index.difference(sessions)):
            raise ValueError("{} has off-calendar observations.".format(ticker))
        exclusions = pd.to_datetime(protocol["quote_exclusions"].get(ticker, []))
        if not exclusions.isin(frame.index).all():
            raise ValueError("{}: reviewed exclusion is missing.".format(ticker))
        frame["quote_excluded"] = frame.index.isin(exclusions)
        frame.loc[frame["quote_excluded"], price_columns + ["Volume"]] = np.nan
        panel = frame.reindex(sessions)
        panel["observed_row"] = sessions.isin(frame.index)
        panel["quote_excluded"] = panel["quote_excluded"].eq(True)
        aligned[ticker] = panel
        active = panel.loc[first_date:]
        invalid = (~np.isfinite(active[price_columns]) | active[price_columns].le(0)).any(axis=1)
        audit_rows.append({
            "ticker": ticker, "price_start": first_date,
            "raw_currency": raw_meta.get("currency"),
            "repaired_currency": metadata["currency"],
            "raw_rows": len(raw), "repaired_rows": len(prices),
            "missing_sessions": int((~active["observed_row"]).sum()),
            "excluded_quotes": int(active["quote_excluded"].sum()),
            "other_invalid_rows": int((invalid & ~active["quote_excluded"]).sum()),
        })

    monthly = etf_data.build_monthly_prices(aligned, schedule)
    opens, closes = etf_data.build_daily_prices(aligned, schedule)
    baseline_scores = features.build_momentum_scores(
        monthly, protocol["baseline_lookback_months"]
    )
    eligible = baseline_scores.notna()
    baseline = backtest.build_target_weights(baseline_scores, protocol["baseline_top_n"])
    first_decision = baseline.index[baseline["CASH"].lt(1)][0]
    calendar = backtest.build_execution_calendar(
        baseline.loc[first_decision:].index, sessions
    )
    calendar["period"] = pd.Series(index=calendar.index, dtype=object)
    for period, (start, stop) in protocol["periods"].items():
        mask = ((calendar["execution_date"] >= pd.Timestamp(start))
                & (calendar["execution_date"] < pd.Timestamp(stop)))
        if calendar.loc[mask, "period"].notna().any():
            raise ValueError("Evaluation periods overlap.")
        calendar.loc[mask, "period"] = period
    if calendar["period"].isna().any():
        raise ValueError("A scheduled trade falls outside the study periods.")

    executions = pd.DatetimeIndex(calendar["execution_date"], name="Date")
    baseline_targets = baseline.loc[calendar.index].copy()
    baseline_targets.index = executions
    benchmark = backtest.build_benchmark_weights(eligible).loc[calendar.index].copy()
    benchmark.index = executions
    excluded = pd.DataFrame({
        ticker: frame["quote_excluded"] for ticker, frame in aligned.items()
    }).reindex(index=opens.index, columns=opens.columns)
    final_start, final_stop = map(pd.Timestamp, protocol["periods"]["final_evaluation"])
    final_mask = (opens.index >= final_start) & (opens.index < final_stop)
    final_calendar = calendar.loc[calendar["period"].eq("final_evaluation")]
    marked_open, marked_close, stale = etf_data.prepare_valuation_marks(
        opens.loc[final_mask], closes.loc[final_mask], excluded.loc[final_mask],
        execution_dates=final_calendar["execution_date"],
    )
    valuation_open, valuation_close = opens.copy(), closes.copy()
    valuation_open.loc[final_mask] = marked_open
    valuation_close.loc[final_mask] = marked_close
    return {
        "universe": universe, "schedule": schedule,
        "audit": pd.DataFrame(audit_rows).set_index("ticker"),
        "monthly_prices": monthly, "adjusted_open": opens, "adjusted_close": closes,
        "valuation_open": valuation_open, "valuation_close": valuation_close,
        "common_eligibility": eligible, "calendar": calendar,
        "baseline_targets": baseline_targets, "benchmark_targets": benchmark,
        "stale_marks": stale,
    }


def summarise_results(results, initial_capital_gbp):
    metrics = etf_research.summarise_portfolios(results, initial_capital_gbp)
    returns = pd.DataFrame({
        name: result["daily"]["net_return"] for name, result in results.items()
    })
    metrics["sharpe_zero_rf"] = (
        np.sqrt(252) * returns.mean() / returns.std(ddof=1).replace(0, np.nan)
    )
    for result in results.values():
        daily = result["daily"]
        np.testing.assert_allclose(
            initial_capital_gbp * (1 + daily["net_return"]).cumprod(),
            daily["nav_close"], rtol=1e-10, atol=1e-8,
        )
        np.testing.assert_allclose(
            result["holdings"].sum(axis=1), daily["nav_close"],
            rtol=1e-10, atol=1e-8,
        )
    return metrics


def run_candidate_history(data, protocol):
    """Compute fixed candidates once; selectors still slice past-only training dates."""
    calendar = data["calendar"]
    first_trade = calendar["execution_date"].min()
    last_bound = pd.Timestamp(protocol["data_end"])
    mask = ((data["valuation_open"].index >= first_trade)
            & (data["valuation_open"].index < last_bound))
    with contextlib.redirect_stdout(io.StringIO()):
        summary, runs = validation.run_parameter_sweep(
            data["monthly_prices"].loc[lambda frame: frame.index < last_bound],
            data["valuation_open"].loc[mask], data["valuation_close"].loc[mask],
            calendar, data["common_eligibility"],
            lookbacks=tuple(protocol["lookbacks"]), top_ns=tuple(protocol["top_ns"]),
            **protocol["backtest"],
        )
    returns = pd.DataFrame({
        key: result["daily"]["net_return"] for key, result in runs.items()
    }).sort_index(axis=1)
    return {
        "parameter_grid": summary[["lookback_months", "top_n"]].sort_index(),
        "backtests": runs, "returns": returns,
    }


def _run_portfolios(data, target_sets, start, end, settings):
    return {
        name: backtest.run_backtest(
            data["valuation_open"].loc[start:end],
            data["valuation_close"].loc[start:end],
            weights, **settings,
        ) for name, weights in target_sets.items()
    }


def development_analysis(data, candidates, protocol):
    start, stop = map(pd.Timestamp, protocol["periods"]["development"])
    dates = candidates["returns"].index
    dates = dates[(dates >= start) & (dates < stop)]
    # These paths start from the same initial allocation as the full history.
    runs = {
        key: {name: frame.loc[dates].copy() for name, frame in result.items()}
        for key, result in candidates["backtests"].items()
    }
    grid = candidates["parameter_grid"]
    metrics = grid.join(summarise_results(runs, protocol["backtest"]["initial_capital_gbp"]))
    executions = pd.DatetimeIndex(
        data["calendar"].loc[data["calendar"]["period"].eq("development"), "execution_date"]
    )
    comparison = _run_portfolios(data, {
        "momentum": data["baseline_targets"].loc[executions],
        "equal_weight": data["benchmark_targets"].loc[executions],
        "buy_and_hold": data["benchmark_targets"].loc[[executions[0]]],
    }, dates[0], dates[-1], protocol["backtest"])
    returns = candidates["returns"].loc[dates]
    benchmark = comparison["equal_weight"]["daily"]["net_return"]
    excess = returns.sub(benchmark, axis=0)
    tests = momentum_validation.bootstrap_sweep_excess(excess, **protocol["inference"])
    metrics["mean_net_excess_bps"] = excess.mean() * 10000
    metrics["correlation_to_benchmark"] = returns.corrwith(benchmark)
    return {
        "sweep_metrics": metrics, "sweep_tests": tests, "returns": returns,
        "backtests": comparison,
        "metrics": summarise_results(comparison, protocol["backtest"]["initial_capital_gbp"]),
    }


def select_annual_choices(data, candidates, protocol):
    folds = momentum_walk_forward.build_annual_walk_forward_folds(
        candidates["returns"].loc[protocol["first_complete_training_year"]:],
        data["schedule"], training_years=protocol["walk_forward"]["training_years"],
    )
    return momentum_walk_forward.select_walk_forward_configurations(
        candidates["returns"], folds, candidates["parameter_grid"],
        protocol["walk_forward"],
    )


def build_walk_forward_targets(data, choices):
    """Retain month-end signals and trade at the following session's open."""
    if choices.empty or not choices.index.is_unique:
        raise ValueError("Provide choices with unique fold identifiers.")
    ordered = choices.sort_values("test_start")
    if (ordered["test_start"].iloc[1:].to_numpy()
            <= ordered["test_end"].iloc[:-1].to_numpy()).any():
        raise ValueError("Test periods overlap.")
    calendar = data["calendar"]
    parts = []
    for choice in ordered.itertuples():
        if not choice.selection_date < choice.test_start <= choice.test_end:
            raise ValueError("Configuration selection must precede its test period.")
        fold_calendar = calendar.loc[
            calendar["execution_date"].between(choice.test_start, choice.test_end)
        ]
        if (fold_calendar.empty or fold_calendar.index[0] != choice.selection_date
                or fold_calendar["execution_date"].iloc[0] != choice.test_start):
            raise ValueError("Selection and first monthly execution do not align.")
        scores = features.build_momentum_scores(
            data["monthly_prices"].loc[:choice.test_end],
            lookback_months=int(choice.lookback_months),
        ).loc[fold_calendar.index]
        eligible = data["common_eligibility"].reindex_like(scores)
        if (eligible.isna().any().any()
                or not eligible.isin([True, False]).all().all()
                or (eligible.astype(bool) & scores.isna()).any().any()):
            raise ValueError("Missing common eligibility or eligible momentum score.")
        weights = backtest.build_target_weights(
            scores.where(eligible.astype(bool)), top_n=int(choice.top_n)
        )
        weights.index = pd.DatetimeIndex(fold_calendar["execution_date"], name="Date")
        parts.append(weights)
    targets = pd.concat(parts).sort_index()
    expected = pd.DatetimeIndex(calendar.loc[
        calendar["execution_date"].between(
            ordered["test_start"].min(), ordered["test_end"].max()
        ), "execution_date"
    ], name="Date")
    if not targets.index.is_unique or not targets.index.equals(expected):
        raise ValueError("Walk-forward targets contain missing or duplicate rebalances.")
    return targets


def evaluate_walk_forward_period(data, choices, protocol):
    """Start a new period from cash, then carry holdings continuously across its years."""
    targets = build_walk_forward_targets(data, choices)
    start, end = targets.index[0], choices["test_end"].max()
    target_sets = {
        "walk_forward": targets,
        "equal_weight": data["benchmark_targets"].loc[targets.index],
        "buy_and_hold": data["benchmark_targets"].loc[[start]],
    }
    runs = _run_portfolios(data, target_sets, start, end, protocol["backtest"])
    returns = pd.DataFrame({
        name: result["daily"]["net_return"] for name, result in runs.items()
    })
    annual = 100 * ((1 + returns).groupby(returns.index.year).prod() - 1)
    return {
        "choices": choices.copy(), "targets": targets, "backtests": runs,
        "returns": returns, "annual_returns": annual.rename_axis("year"),
        "equity": pd.DataFrame({
            name: result["daily"]["nav_close"] for name, result in runs.items()
        }),
        "metrics": summarise_results(runs, protocol["backtest"]["initial_capital_gbp"]),
    }


def evaluate_all_periods(data, candidates, protocol):
    choices = select_annual_choices(data, candidates, protocol)
    evaluations = {}
    for period, (start, stop) in protocol["periods"].items():
        selected = choices.loc[
            (choices["test_start"] >= pd.Timestamp(start))
            & (choices["test_start"] < pd.Timestamp(stop))
        ]
        evaluations[period] = evaluate_walk_forward_period(data, selected, protocol)
    return evaluations


def attribute_profit(result, settings):
    """Allocate actual ledger P&L and proportional trading fees to each instrument."""
    if settings["cash_rate_annual"] != 0:
        raise ValueError("This attribution requires zero cash interest.")
    values, trades = result["holdings"].drop(columns="CASH"), result["trades"]
    if not values.index.equals(trades.index) or not values.columns.equals(trades.columns):
        raise ValueError("Holdings and signed trades must have matching labels.")
    gross = values - values.shift(1, fill_value=0.0) - trades
    costs = trades.abs() * settings["cost_per_side_bps"] / 10000
    net = gross - costs
    daily = result["daily"]
    previous_nav = daily["nav_close"].shift(
        1, fill_value=settings["initial_capital_gbp"]
    )
    np.testing.assert_allclose(
        costs.sum(axis=1), daily["cost"], rtol=1e-10, atol=1e-8,
    )
    np.testing.assert_allclose(
        net.sum(axis=1), daily["nav_close"] - previous_nav, rtol=1e-10, atol=1e-8,
    )
    attribution = pd.DataFrame({
        "gross_pnl_gbp": gross.sum(), "cost_gbp": costs.sum(),
        "net_pnl_gbp": net.sum(),
    })
    return attribution, net


def test_final_excess(evaluation, protocol):
    excess = (
        evaluation["returns"]["walk_forward"]
        - evaluation["returns"]["equal_weight"]
    ).to_frame("walk_forward")
    return momentum_validation.bootstrap_sweep_excess(
        excess, **protocol["inference"]
    ).rename_axis("strategy")


def export_results(destination, evaluations, final_test, attribution, protocol):
    """Save separate net-return streams; independent period restarts must stay explicit."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for period, result in evaluations.items():
        result["returns"].to_csv(destination / (period + "_net_returns.csv"), index_label="Date")
    metrics = pd.concat({
        period: result["metrics"] for period, result in evaluations.items()
    }, names=["period", "portfolio"])
    metrics.to_csv(destination / "period_metrics.csv")
    choices = pd.concat({
        period: result["choices"] for period, result in evaluations.items()
    }, names=["period", "fold"])
    choices.to_csv(destination / "annual_choices.csv")
    attribution.to_csv(destination / "final_attribution.csv")
    final_test.to_csv(destination / "final_excess_test.csv")
    metadata = {
        "protocol": protocol,
        "separate_initial_capital_per_period": True,
        "initial_entry_cost_included": True,
        "forced_final_liquidation": False,
        "daily_return_definition": "Closing NAV / previous closing NAV - 1; first denominator is initial capital.",
        "periods_are_not_one_continuous_investment": True,
        "final_2026_period_ends": "2026-08-28",
        "past_work_overlap": "Related ETF research had already examined later market history.",
        "status": "Research complete; standalone edge and combined-portfolio benefit not established.",
    }
    (destination / "study_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
