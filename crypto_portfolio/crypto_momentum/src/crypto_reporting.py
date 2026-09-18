import numpy as np
import pandas as pd


def summarise_backtest(ledger, initial_cash=1_000.0):
    #summarary function, assuming zero cash rate
    if len(ledger) < 2:
        raise ValueError("At least two daily observations are required.")
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite.")

    expected_first_close = initial_cash * (1.0 + float(ledger["net_return"].iloc[0]))

    if not np.isclose(ledger["equity_close"].iloc[0], expected_first_close, rtol=1e-10, atol=1e-10):
        raise ValueError("Starting equity does not match the first return.")
    
    equity = ledger["equity_close"].astype(float)
    returns = ledger["net_return"].astype(float)
    if (not np.isfinite(equity.to_numpy()).all() or not np.isfinite(returns.to_numpy()).all() or (equity <= 0).any()):
        raise ValueError("Invalid equity or return observations.")

    elapsed_days = (ledger["valuation_at"].iloc[-1] - ledger.index[0]).total_seconds() / 86_400
    if elapsed_days <= 0:
        raise ValueError("The evaluation period must have positive length.")

    years = elapsed_days / 365.25
    growth_multiple = equity.iloc[-1] / initial_cash

    daily_volatility = returns.std(ddof=1)
    annualised_volatility = daily_volatility * np.sqrt(365.25)
    sharpe = returns.mean() / daily_volatility * np.sqrt(365.25) if daily_volatility > 0 else np.nan

    #include inital capital in running peak
    running_peak = equity.cummax().clip(lower=initial_cash)
    drawdown = equity / running_peak - 1

    crypto_weight = 1 - ledger["cash_gbp"] / equity
    one_way_turnover = ledger["traded_notional"] / ledger["equity_open"]
    total_cost = ledger["fee"].sum() + ledger["other_cost"].sum()

    return pd.Series({
        "final_equity_gbp": equity.iloc[-1],
        "total_return": growth_multiple - 1,
        "cagr": growth_multiple ** (1 / years) - 1,
        "annualised_volatility": annualised_volatility,
        "sharpe_zero_cash_rate": sharpe,
        "max_drawdown": drawdown.min(),
        "mean_crypto_weight": crypto_weight.mean(),
        "annualised_one_way_turnover": one_way_turnover.sum() / years,
        "total_cost_gbp": total_cost
    })

def summarise_backtest_period(ledger, start, end):
    #summarise [start, end) backtest.
    boundaries = []
    for value in (start, end):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError("Period boundaries must be timezone-aware.")
        timestamp = timestamp.tz_convert("UTC")
        if timestamp != timestamp.normalize():
            raise ValueError("Period boundaries must be midnight UTC.")

        boundaries.append(timestamp)

    start, end = boundaries

    if start >= end:
        raise ValueError("Require start < end.")
    if ledger.index.has_duplicates:
        raise ValueError("Ledger dates must be unique.")

    ordered = ledger.sort_index()
    period = ordered.loc[(ordered.index >= start) & (ordered.index < end)].copy()
    expected_dates = pd.date_range(start=start,end=end, freq="D", inclusive="left")

    if not period.index.equals(expected_dates):
        raise ValueError("The requested period has missing daily rows.")
    previous_day = start - pd.Timedelta(days=1)
    if previous_day not in ordered.index:
        raise ValueError("The preceding day's closing equity is required.")
    previous_row = ordered.loc[previous_day]
    if previous_row["valuation_at"] != start:
        raise ValueError("The starting valuation has the wrong timestamp.")

    starting_equity = float(previous_row["equity_close"])

    #check daily return against continuous equity history
    previous_equities = np.concatenate([[starting_equity], period["equity_close"].to_numpy(dtype=float)[:-1]])
    reconstructed_equity = previous_equities * (1.0 + period["net_return"].to_numpy(dtype=float))
    if not np.allclose(reconstructed_equity, period["equity_close"].to_numpy(dtype=float), rtol=1e-10, atol=1e-10):
        raise ValueError("Daily returns do not reconcile with equity.")

    return summarise_backtest(ledger=period, initial_cash=starting_equity)