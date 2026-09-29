import numpy as np
import pandas as pd
from scipy.optimize import brentq


def build_target_weights(signal):

    #give each active asset its share of portfolio capital
    if signal.empty or not signal.columns.is_unique:
        raise ValueError("Provide signals for unique instruments.")
    if "CASH" in signal.columns:
        raise ValueError("CASH is reserved for the cash allocation.")
    invalid = signal.notna() & ~signal.isin([0, 1])
    if invalid.any().any():
        raise ValueError("Signals must contain only 0, 1 or missing values.")

    n_assets = len(signal.columns)
    weights = signal.astype(float) / n_assets

    #incomplete signal row does not define a portfolio allocation
    ready = signal.notna().all(axis=1)
    weights.loc[~ready, :] = np.nan
    weights["CASH"] = (1 - weights.sum(axis=1, min_count=n_assets)).clip(lower=0, upper=1)

    return weights

def rebalance_portfolio(current_values, target_weights, cost_per_side_bps):
    #rebalance holdings (cash included) after transaction costs
    if (not current_values.index.is_unique or not target_weights.index.is_unique or "CASH" not in current_values.index or set(current_values.index) != set(target_weights.index)):
        raise ValueError("Values and weights must have matching unique labels, including cash.")

    values = current_values.astype(float)
    weights = target_weights.reindex(values.index).astype(float)

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Current values must be finite and nonnegative.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Target weights must be finite and nonnegative.")
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Target weights must sum to one.")

    #neglect insignicant differences from a total of one
    weights = weights / weights.sum()
    nav_before = float(values.sum())
    rate = float(cost_per_side_bps) / 10_000

    if nav_before <= 0 or not np.isfinite(nav_before):
        raise ValueError("Portfolio value must be positive and finite.")
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("Cost must be between 0 and 10,000 bp, excluding 10,000.")

    asset_values = values.drop("CASH")
    asset_weights = weights.drop("CASH")

    def funding_error(nav_after): #(needed as a function for brentq to work)
        traded = (asset_weights * nav_after - asset_values).abs().sum()
        return nav_after + rate * traded - nav_before

    #remaining capital plus transaction costs must be the starting capital
    nav_after = brentq(funding_error, 0.0, nav_before)

    new_values = weights * nav_after
    trades = new_values.drop("CASH") - asset_values
    traded_notional = float(trades.abs().sum())

    return {
        "values": new_values,
        "trades": trades,
        "traded_notional": traded_notional,
        "cost": rate * traded_notional,
        "nav_before": nav_before,
        "nav_after": nav_after
    }

def simulate_portfolio_day(units, cash, open_prices, close_prices, *, cost_per_side_bps, target_weights=None, cash_growth=1.0):
    #advance adjusted-price holdings through a single trading session. cash_growth gives interest rates
    if (units.empty or not units.index.is_unique or "CASH" in units.index or not units.index.equals(open_prices.index) or not units.index.equals(close_prices.index)):
        raise ValueError("Holdings and prices must have matching unique asset labels.")

    units = units.astype(float)
    cash = float(cash)

    if not np.isfinite(units).all() or (units < 0).any():
        raise ValueError("Units must be finite and nonnegative.")
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("Cash must be finite and nonnegative.")
    cash_growth = float(cash_growth)
    if not np.isfinite(cash_growth) or cash_growth <= 0:
        raise ValueError("Cash growth factor must be positive and finite.")

    #interest owed overnight is credited before open, so any rebalance can trade the gained cash
    interest = cash * (cash_growth - 1.0)
    cash = cash + interest

    prices = pd.DataFrame({"open": open_prices, "close": close_prices}).astype(float)
    if not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("This calculation requires complete, positive daily prices.")

    #existing holdings experience overnight price changes before trading commences
    opening_values = units * prices["open"]
    opening_values["CASH"] = cash
    nav_open_before_cost = float(opening_values.sum())

    cost = 0.0
    traded_notional = 0.0
    trades = pd.Series(0.0, index=units.index)

    if target_weights is not None:
        rebalance = rebalance_portfolio(opening_values, target_weights, cost_per_side_bps)
        opening_values = rebalance["values"]
        units = opening_values.drop("CASH") / prices["open"]
        cash = float(opening_values["CASH"])
        cost = rebalance["cost"]
        traded_notional = rebalance["traded_notional"]
        trades = rebalance["trades"]

    #during day, cash unchanged. interest credited before the following day open
    closing_values = units * prices["close"]
    closing_values["CASH"] = cash

    return {
        "units": units,
        "cash": cash,
        "closing_values": closing_values,
        "nav_open_before_cost": nav_open_before_cost,
        "nav_open_after_cost": float(opening_values.sum()),
        "nav_close": float(closing_values.sum()),
        "cost": cost,
        "interest": interest,
        "traded_notional": traded_notional,
        "trades": trades
    }

def cash_accrual_factors(cash_rate_annual, sessions, *, day=365, max_stale_days=7):
    #interest for each trading day  (1 + sum of (annual rate / day))
    sessions = pd.DatetimeIndex(sessions)
    if (sessions.empty or sessions.hasnans or not sessions.is_unique or not sessions.is_monotonic_increasing):
        raise ValueError("Sessions must be nonempty, ordered and unique.")

    if np.isscalar(cash_rate_annual):
        rate = float(cash_rate_annual)
        if not np.isfinite(rate) or rate <= -1:
            raise ValueError("A scalar cash rate must be finite and above -100%.")
        if rate == 0:
            return pd.Series(1.0, index=sessions)
        gaps = sessions.to_series().diff().dt.days.fillna(0).to_numpy()
        return pd.Series(1.0 + rate * gaps / day, index=sessions)

    rates = cash_rate_annual
    if (not isinstance(rates, pd.Series) or not isinstance(rates.index, pd.DatetimeIndex) or rates.index.hasnans or not rates.index.is_unique or not rates.index.is_monotonic_increasing):
        raise ValueError("Dated cash rates must be a Series with an ordered, unique DatetimeIndex.")
    values = rates.astype(float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Cash rates must be finite and above -100%.")
    if values.index[-1] < sessions[-1] - pd.Timedelta(days=max_stale_days):
        raise ValueError("Cash rates end too long before the final session.")

    calendar = pd.date_range(sessions[0], sessions[-1] - pd.Timedelta(days=1), freq="D")
    factors = pd.Series(1.0, index=sessions)
    if calendar.empty:
        return factors

    daily = values.reindex(values.index.union(calendar)).ffill().reindex(calendar)
    if daily.isna().any():
        raise ValueError("Cash rates must start on or before the first session.")

    #each calendar day is credited at the next session's open
    credited_at = sessions[np.searchsorted(sessions.to_numpy(), calendar.to_numpy(), side="right")]
    accrued = daily.groupby(credited_at).sum() / day
    return factors + accrued.reindex(sessions, fill_value=0.0)

def run_backtest(open_prices, close_prices, execution_weights, *, initial_capital_gbp, cost_per_side_bps, cash_rate_annual=0.0):

    #simulate holdings using targets indexed by trading date
    dates = open_prices.index

    if (open_prices.empty or not isinstance(dates, pd.DatetimeIndex) or not dates.is_unique or not dates.is_monotonic_increasing or not open_prices.columns.is_unique or not close_prices.index.equals(dates) or not close_prices.columns.equals(open_prices.columns)):
        raise ValueError("Provide matching price panels with ordered, unique dates.")

    if (execution_weights.empty or not execution_weights.index.is_unique or not execution_weights.index.isin(dates).all()):
        raise ValueError("Execution dates must be unique and present in the prices.")
    initial_capital_gbp = float(initial_capital_gbp)
    if not np.isfinite(initial_capital_gbp) or initial_capital_gbp <= 0:
        raise ValueError("Initial capital must be positive and finite.")
    #scalar or dated annual rates -> one interest factor per session (1.0 on the first session)
    cash_growth = cash_accrual_factors(cash_rate_annual, dates)

    units = pd.Series(0.0, index=open_prices.columns)
    cash = initial_capital_gbp
    previous_nav = initial_capital_gbp

    records = []
    holding_rows = []
    trade_rows = []

    for date in dates:
        target = execution_weights.loc[date] if date in execution_weights.index else None

        day = simulate_portfolio_day(units=units, cash=cash, open_prices=open_prices.loc[date], close_prices=close_prices.loc[date], cost_per_side_bps=cost_per_side_bps, target_weights=target, cash_growth=cash_growth.loc[date])
        units = day["units"]
        cash = day["cash"]

        records.append({
            "Date": date,
            "nav_open_before_cost": day["nav_open_before_cost"],
            "nav_open_after_cost": day["nav_open_after_cost"],
            "nav_close": day["nav_close"],
            "net_return": day["nav_close"] / previous_nav - 1,
            "cost": day["cost"],
            "interest": day["interest"],
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