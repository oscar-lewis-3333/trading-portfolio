"""Cached-data preparation and reporting for the completed ETF trend study.

Strategy, portfolio accounting and inference remain in the original modules.
This module never downloads or replaces price history.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import etf_backtest
import etf_data
import etf_features
import etf_validation


def study_universe():
    """Return the ten reviewed listings, independent of today's catalogue."""
    rows = [
        ("IUSA", "US equities", "IE0031442068", "equity", "ETF", "physical", "Dist"),
        ("ISF", "UK equities", "IE0005042456", "equity", "ETF", "physical", "Dist"),
        ("IEUX", "Europe excluding UK equities", "IE00B14X4N27", "equity", "ETF", "physical", "Dist"),
        ("IJPN", "Japan equities", "IE00B02KXH56", "equity", "ETF", "physical", "Dist"),
        ("IEEM", "Emerging-market equities", "IE00B0M63177", "equity", "ETF", "physical", "Dist"),
        ("IGLT", "UK conventional gilts", "IE00B1FZSB30", "bond", "ETF", "physical", "Dist"),
        ("INXG", "UK inflation-linked gilts", "IE00B1FZSD53", "bond", "ETF", "physical", "Dist"),
        ("SLXX", "GBP corporate bonds", "IE00B00FV011", "bond", "ETF", "physical", "Dist"),
        ("SGLN", "Physical gold", "IE00B4ND3602", "commodity", "ETC", "physical_metal", "No income"),
        ("CMOP", "Broad commodities", "IE00BD6FTQ80", "commodity", "ETF", "synthetic_swap", "Acc"),
    ]
    universe = pd.DataFrame(rows, columns=[
        "ticker", "exposure", "isin", "asset_group", "instrument_type",
        "replication", "share_class",
    ]).set_index("ticker")
    universe["yf_symbol"] = universe.index + ".L"
    return universe.astype({"isin": "string", "yf_symbol": "string"})


def _read_cached_history(project_root, symbol, start, end, version):
    stem = f"{symbol}_{start}_{end}_{version}"
    directory = Path(project_root) / "data" / version
    price_path = directory / f"{stem}.csv"
    metadata_path = directory / f"{stem}.json"
    for path in (price_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required study cache is missing: {path}")
    prices = pd.read_csv(price_path, index_col="Date", parse_dates=True)
    metadata = json.loads(metadata_path.read_text())
    if metadata["symbol"] != symbol:
        raise ValueError(f"{symbol}: cached metadata names another instrument.")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: cache dates must be ordered and unique.")
    return prices, metadata


def prepare_cached_data(
    project_root, universe, *, data_start, data_end, analysis_start,
    holdout_start, quote_exclusions, baseline_lookback=12,
):
    """Replay the original repairs, exclusions, calendar and common dates."""
    schedule = etf_data.build_lse_schedule(analysis_start, data_end)
    sessions = schedule.index
    aligned_histories = {}
    audit_records = []
    quote_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for ticker, symbol in universe["yf_symbol"].items():
        raw, raw_metadata = _read_cached_history(
            project_root, symbol, data_start, data_end, "raw"
        )
        repaired, repaired_metadata = _read_cached_history(
            project_root, symbol, data_start, data_end, "repaired"
        )
        if repaired_metadata["currency"] != "GBP":
            raise ValueError(f"{ticker}: repaired prices must be in GBP.")

        frame = repaired.loc[analysis_start:].copy()
        frame[quote_columns] = frame[quote_columns].astype(float)
        excluded_dates = pd.to_datetime(quote_exclusions.get(ticker, []))
        if not excluded_dates.isin(frame.index).all():
            raise ValueError(f"{ticker}: a reviewed exclusion is absent from the cache.")
        frame["quote_excluded"] = frame.index.isin(excluded_dates)
        frame.loc[frame["quote_excluded"], quote_columns] = np.nan
        if len(frame.index.difference(sessions)):
            raise ValueError(f"{ticker}: cached dates fall outside the study calendar.")

        aligned = frame.reindex(sessions)
        aligned["observed_row"] = sessions.isin(frame.index)
        aligned["quote_excluded"] = aligned["quote_excluded"].eq(True)
        if not aligned["observed_row"].all():
            raise ValueError(f"{ticker}: missing calendar rows require a new review.")
        aligned_histories[ticker] = aligned

        raw_audit = etf_data.audit_etf_history(raw)
        repaired_audit = etf_data.audit_etf_history(repaired)
        audit_records.append({
            "ticker": ticker,
            "raw_currency": raw_metadata["currency"],
            "repaired_currency": repaired_metadata["currency"],
            "raw_adjustment_ratio": raw_audit["median_adjustment_ratio"],
            "repaired_adjustment_ratio": repaired_audit["median_adjustment_ratio"],
            "sessions": len(aligned),
            "excluded_quotes": int(aligned["quote_excluded"].sum()),
        })

    monthly_prices = etf_data.build_monthly_prices(aligned_histories, schedule)
    adjusted_open, adjusted_close = etf_data.build_daily_prices(
        aligned_histories, schedule
    )
    excluded_quotes = pd.DataFrame({
        ticker: frame["quote_excluded"]
        for ticker, frame in aligned_histories.items()
    })
    for quotes in (adjusted_open, adjusted_close):
        if not quotes.isna().equals(excluded_quotes):
            raise ValueError("Missing prices must match the reviewed exclusions.")
    if monthly_prices.isna().any().any():
        raise ValueError("Month-end signal prices must be observed.")

    # Preserve the original 12-month warm-up for every development candidate.
    _, baseline_signal = etf_features.build_trend_features(
        monthly_prices, lookback_months=baseline_lookback
    )
    baseline_weights = etf_backtest.build_target_weights(baseline_signal)
    next_session = pd.Series(sessions, index=sessions).shift(-1)
    decisions = pd.DataFrame({
        "execution_date": next_session.reindex(baseline_weights.index),
        "signal_ready": baseline_weights.notna().all(axis=1),
    })
    executable_schedule = decisions.loc[
        decisions["signal_ready"] & decisions["execution_date"].notna(),
        ["execution_date"],
    ].copy()
    executable_schedule["period"] = np.where(
        executable_schedule["execution_date"] < pd.Timestamp(holdout_start),
        "development", "holdout",
    )
    period_summary = executable_schedule.groupby("period", sort=False)[
        "execution_date"
    ].agg(first_execution="min", last_execution="max", rebalances="size")

    return {
        "monthly_prices": monthly_prices,
        "adjusted_open": adjusted_open,
        "adjusted_close": adjusted_close,
        "excluded_quotes": excluded_quotes,
        "executable_schedule": executable_schedule,
        "audit": pd.DataFrame(audit_records).set_index("ticker"),
        "period_summary": period_summary,
        "holdout_start": pd.Timestamp(holdout_start),
        "data_end": pd.Timestamp(data_end),
    }


def period_inputs(data, period):
    """Return common prices and decisions for one original evaluation period."""
    if period not in ("development", "holdout"):
        raise ValueError("Choose development or holdout.")
    schedule = data["executable_schedule"].loc[
        data["executable_schedule"]["period"].eq(period)
    ].copy()
    if schedule.empty:
        raise ValueError(f"No execution dates for {period}.")
    end = data["holdout_start"] if period == "development" else data["data_end"]
    dates = data["adjusted_open"].index
    dates = dates[(dates >= schedule["execution_date"].iloc[0]) & (dates < end)]
    return {
        "monthly_prices": data["monthly_prices"].loc[
            data["monthly_prices"].index < end
        ],
        "open_prices": data["adjusted_open"].loc[dates],
        "close_prices": data["adjusted_close"].loc[dates],
        "decision_schedule": schedule,
        "excluded_quotes": data["excluded_quotes"].loc[dates],
    }


def run_development(data, lookbacks, settings):
    """Repeat the four-lookback sweep and the two original passive baselines."""
    inputs = period_inputs(data, "development")
    summary, runs = etf_validation.run_lookback_sweep(
        inputs["monthly_prices"], inputs["open_prices"], inputs["close_prices"],
        inputs["decision_schedule"], lookbacks=lookbacks, **settings,
    )
    dates = pd.DatetimeIndex(inputs["decision_schedule"]["execution_date"], name="Date")
    weights = pd.DataFrame(
        1 / len(inputs["open_prices"].columns), index=dates,
        columns=inputs["open_prices"].columns,
    )
    weights["CASH"] = 0.0
    benchmarks = {}
    for name, targets in {
        "monthly_equal_weight": weights,
        "buy_and_hold": weights.iloc[:1],
    }.items():
        benchmarks[name] = etf_backtest.run_backtest(
            inputs["open_prices"], inputs["close_prices"], targets, **settings
        )
    return summary, runs, benchmarks


def run_buy_and_hold(inputs, settings):
    """Run the final contextual comparison: equal initial allocations, no rebalance."""
    execution_dates = inputs["open_prices"].index[:1]
    marked_open, marked_close, stale_quotes = etf_data.prepare_valuation_marks(
        inputs["open_prices"], inputs["close_prices"], inputs["excluded_quotes"],
        execution_dates=execution_dates,
    )
    targets = pd.DataFrame(
        1 / len(marked_open.columns), index=execution_dates,
        columns=marked_open.columns,
    )
    targets["CASH"] = 0.0
    result = etf_backtest.run_backtest(marked_open, marked_close, targets, **settings)
    result["stale_quotes"] = stale_quotes
    result["daily"]["stale_valuation"] = (
        stale_quotes & result["holdings"].drop(columns="CASH").gt(0)
    ).any(axis=1)
    return result


def summarise_portfolios(results, initial_capital_gbp):
    """Report comparable wealth, daily risk and costs, including initial capital."""
    records = {}
    for name, result in results.items():
        daily = result["daily"]
        nav = daily["nav_close"]
        years = ((nav.index[-1] - nav.index[0]).days + 1) / 365.25
        drawdown = nav / nav.cummax().clip(lower=initial_capital_gbp) - 1
        records[name] = {
            "final_value_gbp": nav.iloc[-1],
            "total_return_pct": (nav.iloc[-1] / initial_capital_gbp - 1) * 100,
            "cagr_pct": ((nav.iloc[-1] / initial_capital_gbp) ** (1 / years) - 1) * 100,
            "annualised_vol_pct": daily["net_return"].std(ddof=1) * np.sqrt(252) * 100,
            "max_drawdown_pct": drawdown.min() * 100,
            "mean_cash_pct": (daily["cash"] / nav).mean() * 100,
            "total_cost_gbp": daily["cost"].sum(),
        }
    return pd.DataFrame.from_dict(records, orient="index").rename_axis("portfolio")
