import numpy as np
import pandas as pd

from etf_backtest import rebalance_portfolio


def build_target_weights(momentum_scores, top_n=3):
    if momentum_scores.empty or not momentum_scores.columns.is_unique:
        raise ValueError("Provide scores for unique instruments.")
    if "CASH" in momentum_scores.columns:
        raise ValueError("CASH is reserved for the cash allocation.")
    if (isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or not 1 <= top_n <= len(momentum_scores.columns)):
        raise ValueError("top_n must be between 1 and the universe size.")

    scores = momentum_scores.astype(float)
    if (scores.notna() & ~np.isfinite(scores)).any().any():
        raise ValueError("Scores must be finite or missing.")

    #alphabetical order for tickers gives a tie breaker
    ordered_scores = scores.sort_index(axis=1)
    ranks = ordered_scores.rank(axis=1, ascending=False, method="first")

    ready = scores.notna().sum(axis=1).ge(top_n)
    weights = ranks.le(top_n).astype(float) / top_n

    #remain in cash until enough equities have valid scores
    weights.loc[~ready, :] = 0.0
    weights = weights.reindex(columns=scores.columns)
    weights["CASH"] = (1 - weights.sum(axis=1)).clip(0, 1)

    return weights

def build_execution_calendar(decision_dates, trading_sessions):
    decision_dates = pd.DatetimeIndex(decision_dates)
    sessions = pd.DatetimeIndex(trading_sessions)

    for label, dates in [("Decision dates", decision_dates), ("Trading sessions", sessions)]:
        if (dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing):
            raise ValueError(f"{label} must be nonempty, ordered and unique.")
    if not decision_dates.isin(sessions).all():
        raise ValueError("Every decision date must be a trading session.")

    next_session = pd.Series(sessions, index=sessions).shift(-1)
    calendar = pd.DataFrame({"execution_date": next_session.reindex(decision_dates)}).rename_axis("decision_date")

    #final decision may have no following session inside our data
    return calendar.dropna(subset=["execution_date"])

def build_benchmark_weights(eligible):
    if eligible.empty or not eligible.columns.is_unique:
        raise ValueError("Provide eligibility for unique instruments.")
    if "CASH" in eligible.columns:
        raise ValueError("CASH is reserved for the cash allocation.")
    if (eligible.isna().any().any() or not eligible.isin([True, False]).all().all()):
        raise ValueError("Eligibility must contain only True or False.")

    counts = eligible.astype(bool).sum(axis=1)
    weights = eligible.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0.0)
    weights["CASH"] = (1 - weights.sum(axis=1)).clip(0, 1)
    return weights


def simulate_portfolio_day(units, cash, open_prices, close_prices, *, cost_per_side_bps, target_weights=None):
    assets = units.index
    if (units.empty or not assets.is_unique or "CASH" in assets or not open_prices.index.equals(assets) or not close_prices.index.equals(assets)):
        raise ValueError("Holdings and prices must have matching unique assets.")

    units = units.astype(float).copy()
    cash = float(cash)
    if (not np.isfinite(units).all() or units.lt(0).any() or not np.isfinite(cash) or cash < 0):
        raise ValueError("Holdings and cash must be finite and nonnegative.")

    open_prices = open_prices.astype(float)
    close_prices = close_prices.astype(float)

    def mark_positions(quantities, prices, label):
        held = quantities.gt(0)
        bad = held & (~np.isfinite(prices) | prices.le(0))
        if bad.any():
            raise ValueError(f"Unusable {label} for holdings: {assets[bad].tolist()}")

        values = pd.Series(0.0, index=assets)
        values.loc[held] = quantities.loc[held] * prices.loc[held]
        return values

    #existing holdings experience overnight moves before trading
    opening_values = mark_positions(units, open_prices, "open")
    opening_values["CASH"] = cash
    nav_open = float(opening_values.sum())

    cost = 0.0
    traded_notional = 0.0
    trades = pd.Series(0.0, index=assets)

    if target_weights is not None:
        rebalance = rebalance_portfolio(opening_values, target_weights, cost_per_side_bps)
        new_values = rebalance["values"].drop("CASH")
        active = new_values.gt(0)
        bad = active & (~np.isfinite(open_prices) | open_prices.le(0))
        if bad.any():
            raise ValueError(f"Unusable open for targets: {assets[bad].tolist()}")

        units = pd.Series(0.0, index=assets)
        units.loc[active] = new_values.loc[active] / open_prices.loc[active]
        cash = float(rebalance["values"]["CASH"])
        cost = rebalance["cost"]
        traded_notional = rebalance["traded_notional"]
        trades = rebalance["trades"]

    closing_values = mark_positions(units, close_prices, "close")
    closing_values["CASH"] = cash

    return {
        "units": units,
        "cash": cash,
        "closing_values": closing_values,
        "nav_open_before_cost": nav_open,
        "nav_open_after_cost": nav_open - cost,
        "nav_close": float(closing_values.sum()),
        "cost": cost,
        "traded_notional": traded_notional,
        "trades": trades
    }

def run_backtest(open_prices, close_prices, execution_weights, *, initial_capital_gbp, cost_per_side_bps, cash_rate_annual=0.0):
    dates = open_prices.index
    assets = open_prices.columns

    if (open_prices.empty or not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
        or not assets.is_unique or "CASH" in assets or not close_prices.index.equals(dates) or not close_prices.columns.equals(assets)):
        raise ValueError("Provide matching price panels with ordered, unique dates.")

    trade_dates = execution_weights.index
    if (execution_weights.empty or not isinstance(trade_dates, pd.DatetimeIndex) or not trade_dates.is_unique or not trade_dates.is_monotonic_increasing
        or not trade_dates.isin(dates).all() or trade_dates[0] != dates[0] or not execution_weights.columns.is_unique or set(execution_weights.columns) != (set(assets) | {"CASH"})):
        raise ValueError("Targets must match the assets and start on the first price date.")

    capital = float(initial_capital_gbp)
    cost_bps = float(cost_per_side_bps)

    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Initial capital must be positive and finite.")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("Cost must be between 0 and 10,000 bp, excluding 10,000.")
    if cash_rate_annual != 0:
        raise NotImplementedError("This version models zero cash interest.")

    targets = execution_weights.reindex(columns=[*assets, "CASH"])
    units = pd.Series(0.0, index=assets)
    cash = capital
    previous_nav = capital
    records, holding_rows, trade_rows = [], [], []

    for date in dates:
        target = targets.loc[date] if date in trade_dates else None
        try:
            day = simulate_portfolio_day(
                units,
                cash,
                open_prices.loc[date],
                close_prices.loc[date],
                cost_per_side_bps=cost_bps,
                target_weights=target
            )
        except ValueError as e:
            raise ValueError(f"{date.date()}: {e}") from e

        units, cash = day["units"], day["cash"]
        records.append({
            "Date": date,
            "nav_open_before_cost": day["nav_open_before_cost"],
            "nav_open_after_cost": day["nav_open_after_cost"],
            "nav_close": day["nav_close"],
            "net_return": day["nav_close"] / previous_nav - 1,
            "cost": day["cost"],
            "traded_notional": day["traded_notional"],
            "cash": cash,
            "rebalanced": target is not None
        })

        holding_rows.append(day["closing_values"])
        trade_rows.append(day["trades"])
        previous_nav = day["nav_close"]

    return {
        "daily": pd.DataFrame(records).set_index("Date"),
        "holdings": pd.DataFrame(holding_rows, index=dates),
        "trades": pd.DataFrame(trade_rows, index=dates)
    }