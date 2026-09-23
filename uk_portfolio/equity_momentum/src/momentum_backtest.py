import pandas as pd
import numpy as np


def build_monthly_rebalance_calendar(schedule):
    #pair each month-end signal with the following trading session
    sessions = pd.DatetimeIndex(schedule.index)

    if (not sessions.is_unique or not sessions.is_monotonic_increasing or sessions.tz is not None):
        raise ValueError("Expected unique, sorted, timezone-naive trading sessions.")
    
    months = sessions.to_period("M")

    #month-end is confirmed when the next session is in another month
    month_changes = months[:-1] != months[1:]

    signal_dates = sessions[:-1][month_changes]
    execution_dates = sessions[1:][month_changes]

    return pd.DataFrame({"execution_date": execution_dates}, index=pd.DatetimeIndex(signal_dates, name="signal_date"))

def build_momentum_targets(features, rebalance_calendar, top_frac=0.20, score_column="raw_momentum_score"):
    #select stocks using signal-date information and assign target weights
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac must be in (0, 1].")

    if (features.index.names != ["Date", "ticker"] or not features.index.is_unique):
        raise ValueError("Expected unique (Date, ticker) feature rows.")

    execution_dates = pd.DatetimeIndex(rebalance_calendar["execution_date"], name="Date")
    signal_dates = pd.DatetimeIndex(rebalance_calendar.index)

    if (rebalance_calendar.empty or not signal_dates.is_unique or not execution_dates.is_unique
        or not execution_dates.is_monotonic_increasing or not (execution_dates > signal_dates).all()):
        raise ValueError("Invalid signal/execution calendar.")

    tickers = sorted(features.index.get_level_values("ticker").unique())
    targets = pd.DataFrame(0.0, index=execution_dates, columns=pd.Index(tickers, name="ticker"))
    records = []

    for signal_date, execution_date in zip(signal_dates, execution_dates):
        snapshot = features.xs(signal_date, level="Date")
        candidates = snapshot.loc[snapshot["eligible"].eq(True)].copy()
        if not np.isfinite(candidates[score_column]).all():
            raise ValueError(f"Eligible stocks have invalid scores on {signal_date}.")

        ranked = candidates.reset_index().sort_values([score_column, "ticker"], ascending=[False, True])
        n_eligible = len(ranked)
        n_selected = int(np.ceil(top_frac * n_eligible))
        selected = ranked.head(n_selected)["ticker"].tolist()
        if n_selected:
            targets.loc[execution_date, selected] = 1.0 / n_selected

        records.append({"signal_date": signal_date,
                "execution_date": execution_date,
                "eligible_stocks": n_eligible,
                "selected_stocks": n_selected,
                "target_cash_weight": 0.0 if n_selected else 1.0})

    summary = pd.DataFrame(records).set_index("execution_date").rename_axis("Date")

    return targets, summary

def rebalance_with_costs(
    holdings_gbp, cash_gbp, target_weights, cost_per_side_bps, frozen=None
):
    """Fund trades after costs, preserving positions that cannot be traded.

    Blocked target purchases leave cash. If trapped holdings prevent the other
    targets being fully funded, scale those tradable targets proportionally.
    """
    if (
        not holdings_gbp.index.is_unique
        or not holdings_gbp.index.equals(target_weights.index)
    ):
        raise ValueError("Holdings and weights must have matching unique tickers.")

    holdings = holdings_gbp.astype(float)
    weights = target_weights.astype(float)
    if (
        not np.isfinite(holdings).all()
        or not np.isfinite(weights).all()
        or holdings.lt(0).any()
        or weights.lt(0).any()
        or not np.isfinite(cash_gbp)
        or cash_gbp < 0
    ):
        raise ValueError("Expected finite, non-negative holdings, cash and weights.")

    total_weight = weights.sum()
    if total_weight > 1 + 1e-12:
        raise ValueError("Target weights cannot exceed 100%.")
    if total_weight > 1:
        weights = weights / total_weight

    if frozen is None:
        frozen = pd.Series(False, index=holdings.index)
    if (
        not frozen.index.equals(holdings.index)
        or frozen.isna().any()
        or not frozen.isin([True, False]).all()
    ):
        raise ValueError("Frozen-position flags must match the holdings index.")
    frozen = frozen.astype(bool)

    cost_rate = float(cost_per_side_bps) / 10_000
    if not np.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("Invalid transaction cost.")

    equity_before = float(holdings.sum() + cash_gbp)
    fixed_values = holdings.where(frozen, 0.)
    fixed_total = float(fixed_values.sum())
    tradable_weights = weights.where(~frozen, 0.)
    tradable_weight_sum = float(tradable_weights.sum())

    def allocation(equity):
        available = max(0., equity - fixed_total)
        desired = equity * tradable_weight_sum
        scale = min(1., available / desired) if desired > 0 else 1.
        return fixed_values + tradable_weights * equity * scale

    lower, upper = fixed_total, equity_before
    for _ in range(64):
        candidate_equity = (lower + upper) / 2
        candidate_trades = allocation(candidate_equity) - holdings
        candidate_cost = candidate_trades.abs().sum() * cost_rate
        if candidate_equity + candidate_cost > equity_before:
            upper = candidate_equity
        else:
            lower = candidate_equity

    holdings_after = allocation((lower + upper) / 2)
    trades = holdings_after - holdings
    costs = trades.abs() * cost_rate
    total_cost = float(costs.sum())
    cash_after = float(cash_gbp - trades.sum() - total_cost)
    if cash_after < -1e-10 * max(1., equity_before):
        raise ValueError("Rebalance spends more cash than available.")

    return {
        "holdings_gbp": holdings_after,
        "trades_gbp": trades,
        "costs_gbp": costs,
        "cash_gbp": max(0., cash_after),
        "total_cost_gbp": total_cost,
        "equity_before_gbp": equity_before,
        "equity_after_gbp": equity_before - total_cost,
    }


def step_portfolio_day(
    market_day, units, cash_gbp, target_weights=None,
    cost_per_side_bps=10., last_valuation_gbp=None,
    missing_valuation_counts=None, max_missing_valuation_sessions=0,
):
    """Advance one session with explicit stale-valuation rules.

    Stale marks estimate portfolio value and never supply an execution price.
    Failed suspension orders expire; the next trade attempt is the next
    scheduled rebalance. Existing units remain held when trading resumes.
    An optional bounded allowance covers missing closing marks, never fills.
    """
    if not units.index.is_unique or not market_day.index.is_unique:
        raise ValueError("Expected unique tickers.")
    units = units.astype(float).copy()
    if (
        not np.isfinite(units).all() or units.lt(0).any()
        or not np.isfinite(cash_gbp) or cash_gbp < 0
    ):
        raise ValueError("Invalid starting units or cash.")

    if (isinstance(max_missing_valuation_sessions, bool)
        or not isinstance(max_missing_valuation_sessions, int)
        or max_missing_valuation_sessions < 0):
        raise ValueError("Missing-valuation allowance must be a non-negative integer.")
    if missing_valuation_counts is None:
        missing_valuation_counts = pd.Series(0, index=units.index, dtype=int)
    if (not missing_valuation_counts.index.equals(units.index)
        or not np.isfinite(missing_valuation_counts).all()
        or missing_valuation_counts.lt(0).any()
        or missing_valuation_counts.mod(1).ne(0).any()):
        raise ValueError("Invalid missing-valuation counters.")

    day = market_day.reindex(units.index)
    cash = float(cash_gbp)
    held_at_start = units.gt(0)
    false_flags = pd.Series(False, index=units.index)
    suspended_open = day.get("known_suspended", false_flags).eq(True)
    suspended_close = day.get("close_suspended", false_flags).eq(True)
    unverified = day.get("unverified_quote", false_flags).eq(True)
    valid_open = (
        day["open_reference_usable"].eq(True)
        & np.isfinite(day["open_gbp"]) & day["open_gbp"].gt(0)
        & ~suspended_open & ~unverified
    )
    valid_close = (
        day["close_reference_usable"].eq(True)
        & np.isfinite(day["close_gbp"]) & day["close_gbp"].gt(0)
        & ~suspended_close & ~unverified
    )

    if last_valuation_gbp is None:
        valuation = pd.Series(np.nan, index=units.index)
    else:
        if not last_valuation_gbp.index.equals(units.index):
            raise ValueError("Previous valuation prices must match the holdings index.")
        valuation = last_valuation_gbp.astype(float).copy()
    valuation.loc[valid_open] = day.loc[valid_open, "open_gbp"]

    def reject(mask, message):
        if mask.any():
            raise ValueError(f"{message}: {mask.index[mask].tolist()}")

    dividends = day.loc[held_at_start, "dividend_gbp"]
    if not np.isfinite(dividends).all() or dividends.lt(0).any():
        raise ValueError("Invalid dividend data for existing holdings.")
    dividend_income = float((units.loc[held_at_start] * dividends).sum())
    trades = pd.Series(0., index=units.index)
    total_cost = 0.
    blocked_target_count = 0

    if target_weights is not None:
        if not target_weights.index.equals(units.index):
            raise ValueError("Target weights must match the holdings index.")
        needed = held_at_start | target_weights.gt(0)
        reject(
            needed & ~valid_open & ~(suspended_open & ~unverified),
            "Unusable open_gbp for required holdings",
        )
        reject(held_at_start & (~np.isfinite(valuation) | valuation.le(0)), "No prior valuation for suspended holdings")
        opening_values = pd.Series(0., index=units.index)
        opening_values.loc[held_at_start] = units.loc[held_at_start] * valuation.loc[held_at_start]
        rebalance = rebalance_with_costs(opening_values, cash, target_weights, cost_per_side_bps, frozen=suspended_open)
        new_values = rebalance["holdings_gbp"]
        invested = new_values.gt(0) & ~suspended_open
        #preserve frozen units exactly
        units = units.where(suspended_open, 0.)
        units.loc[invested] = new_values.loc[invested] / day.loc[invested, "open_gbp"]
        cash = rebalance["cash_gbp"]
        trades = rebalance["trades_gbp"]

        total_cost = rebalance["total_cost_gbp"]
        blocked_target_count = int((suspended_open & target_weights.gt(0)).sum())

    held_at_close = units.gt(0)
    missing_valuation = held_at_close & ~valid_close & ~suspended_close & ~unverified
    missing_counts = (missing_valuation_counts + 1).where(missing_valuation, 0).astype(int)
    reject(held_at_close & unverified, "Unusable close_gbp for required holdings")
    reject(missing_valuation & missing_counts.gt(max_missing_valuation_sessions), "Unusable close_gbp for required holdings (missing-valuation allowance exceeded)")

    valuation.loc[valid_close] = day.loc[valid_close, "close_gbp"]
    reject(held_at_close & (~np.isfinite(valuation) | valuation.le(0)), "No prior valuation for suspended holdings")

    closing_values = pd.Series(0., index=units.index)
    closing_values.loc[held_at_close] = units.loc[held_at_close] * valuation.loc[held_at_close]
    stale = held_at_close & ~valid_close
    cash += dividend_income

    return {
        "units": units,
        "holdings_gbp": closing_values,
        "cash_gbp": cash,
        "equity_gbp": float(closing_values.sum() + cash),
        "total_cost_gbp": total_cost,
        "dividends_gbp": dividend_income,
        "trades_gbp": trades,
        "valuation_gbp": valuation,
        "stale_value_gbp": float(closing_values.loc[stale].sum()),
        "stale_holdings_count": int(stale.sum()),
        "blocked_target_count": blocked_target_count,
        "missing_valuation_counts": missing_counts,
        "missing_valuation_holdings_count": int(missing_valuation.sum()),
    }


def run_momentum_backtest(market, targets, sessions, initial_capital_gbp=10_000.0, cost_per_side_bps=10.0, max_missing_valuation_sessions=0):
    #run scheduled rebalances and retain daily values, weights and trades
    sessions = pd.DatetimeIndex(sessions, name="Date")

    if (sessions.empty or not sessions.is_unique or not sessions.is_monotonic_increasing or sessions.tz is not None):
        raise ValueError("Expected sorted, unique trading sessions.")
    if (market.index.names != ["Date", "ticker"] or not market.index.is_unique):
        raise ValueError("Expected unique (Date, ticker) market rows.")

    available_dates = market.index.get_level_values("Date").unique().sort_values()
    expected_dates = available_dates[(available_dates >= sessions[0]) & (available_dates <= sessions[-1])]
    if not sessions.equals(expected_dates):
        raise ValueError("The backtest must include every market session.")

    if (not isinstance(targets.index, pd.DatetimeIndex) or not targets.index.is_unique or not targets.index.is_monotonic_increasing 
        or not targets.columns.is_unique or not targets.index.isin(sessions).all()):
        raise ValueError("Invalid target dates or ticker columns.")

    if not targets.columns.isin(market.index.get_level_values("ticker")).all():
        raise ValueError("Target tickers are missing from market data.")
    if not np.isfinite(initial_capital_gbp) or initial_capital_gbp <= 0:
        raise ValueError("Starting capital must be positive and finite.")

    cost_rate = float(cost_per_side_bps) / 10_000
    if not np.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("Invalid transaction cost.")

    units = pd.Series(0.0, index=targets.columns)
    cash = float(initial_capital_gbp)
    previous_equity = float(initial_capital_gbp)
    last_valuation = None
    missing_valuation_counts = None

    daily_records = []
    weight_records = []
    trade_records = []

    for date in sessions:
        is_rebalance = date in targets.index
        target = targets.loc[date] if is_rebalance else None

        try:
            result = step_portfolio_day(market_day=market.xs(date, level="Date"),
                units=units,
                cash_gbp=cash,
                target_weights=target,
                cost_per_side_bps=cost_per_side_bps,
                last_valuation_gbp=last_valuation,
                missing_valuation_counts=missing_valuation_counts,
                max_missing_valuation_sessions=max_missing_valuation_sessions,
            )
        except ValueError as e:
            raise ValueError(f"{date:%Y-%m-%d}: {e}") from e

        equity = result["equity_gbp"]
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError(f"Invalid portfolio equity on {date:%Y-%m-%d}.")

        daily_records.append({
            "Date": date,
            "equity_gbp": equity,
            "net_return": equity / previous_equity - 1,
            "cash_gbp": result["cash_gbp"],
            "cash_weight": result["cash_gbp"] / equity,
            "total_cost_gbp": result["total_cost_gbp"],
            "dividends_gbp": result["dividends_gbp"],
            "traded_value_gbp": result["trades_gbp"].abs().sum(),
            "holdings_count": int(result["units"].gt(0).sum()),
            "is_rebalance": is_rebalance,
            "stale_value_gbp": result["stale_value_gbp"],
            "stale_weight": result["stale_value_gbp"] / equity,
            "stale_holdings_count": result["stale_holdings_count"],
            "blocked_target_count": result["blocked_target_count"],
            "missing_valuation_holdings_count": result["missing_valuation_holdings_count"],
        })

        weight_records.append((result["holdings_gbp"] / equity).rename(date))
        trades = result["trades_gbp"]
        for ticker, value in trades.loc[trades.ne(0)].items():
            trade_records.append({
                "Date": date,
                "ticker": ticker,
                "trade_gbp": float(value),
                "cost_gbp": float(abs(value) * cost_rate)
            })

        units = result["units"]
        cash = result["cash_gbp"]
        previous_equity = equity
        last_valuation = result["valuation_gbp"]
        missing_valuation_counts = result["missing_valuation_counts"]

    daily = pd.DataFrame(daily_records).set_index("Date")
    weights = pd.DataFrame(weight_records).rename_axis("Date")
    trades = pd.DataFrame(trade_records, columns=["Date", "ticker", "trade_gbp", "cost_gbp"]).set_index(["Date", "ticker"])

    return {
        "daily": daily,
        "weights": weights,
        "trades": trades,
        "final_units": units,
        "final_cash_gbp": cash,
        "final_valuation_gbp": last_valuation,
        "final_missing_valuation_counts": missing_valuation_counts,
    }