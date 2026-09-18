import numpy as np
import pandas as pd

from reversal_backtest import execute_open_orders

def run_cash_interest_backtest(decisions, market, schedule, initial_capital_gbp=10_000, holding_sessions=5, cost_per_side_bps=0.0, net_orders=False, *, holding_sessions_by_date=None, cash_aer=0.0):
    #simulate opening trades, closing values and dividends reserved
    initial_capital_gbp = float(initial_capital_gbp)

    cash_aer = float(cash_aer)
    if not np.isfinite(cash_aer) or cash_aer < 0:
        raise ValueError("Cash AER must be finite and nonnegative.")

    previous_session = None

    if not np.isfinite(initial_capital_gbp) or initial_capital_gbp <= 0:
        raise ValueError("Initial capital must be finite and positive.")
    if (isinstance(holding_sessions, bool) or not isinstance(holding_sessions, int) or holding_sessions < 1):
        raise ValueError("Holding sessions must be a positive integer.")

    sessions = schedule.index
    if (sessions.empty or not sessions.is_unique or not sessions.is_monotonic_increasing or sessions.hasnans):
        raise ValueError("Provide a complete, sorted session calendar.")
    for name, frame in [("decisions", decisions), ("market", market)]:
        if (frame.empty or list(frame.index.names) != ["Date", "ticker"] or not frame.index.is_unique):
            raise ValueError(f"{name} needs unique Date-ticker rows.")

    market = market.sort_index()
    tickers = market.index.get_level_values("ticker").unique().sort_values()
    expected_index = pd.MultiIndex.from_product([sessions, tickers], names=["Date", "ticker"])
    if not market.index.equals(expected_index):
        raise ValueError("The market must contain every ticker-session row.")

    weights = decisions["target_weight"].astype(float).unstack("ticker").sort_index()
    if (not np.isfinite(decisions["target_weight"]).all() or decisions["target_weight"].lt(0).any() or not weights.columns.isin(tickers).all()):
        raise ValueError("Invalid weights or unsupported tickers.")
    weights = weights.fillna(0.0)
    if weights.sum(axis=1).gt(1 + 1e-12).any():
        raise ValueError("Target weights must total at most one.")

    if holding_sessions_by_date is None:
        holding_lengths = pd.Series(holding_sessions, index=weights.index, dtype="int64")
    else:
        if (not isinstance(holding_sessions_by_date, pd.Series) or not holding_sessions_by_date.index.is_unique or not holding_sessions_by_date.index.equals(weights.index)):
            raise ValueError("Holding periods must be a Series matching the decision dates.")

        holding_lengths = holding_sessions_by_date.copy()

        if (not pd.api.types.is_integer_dtype(holding_lengths.dtype) or holding_lengths.isna().any() or holding_lengths.lt(1).any()):
            raise ValueError("Holding periods must contain positive integers.")

    decision_positions = sessions.get_indexer(weights.index)
    if ((decision_positions < 0).any() or holding_lengths.ge(len(sessions)).any()):
        raise ValueError("A decision or planned exit falls outside the calendar.")

    #signal at today's close, entry next open, exit holding_length sessions later
    exit_positions = (decision_positions + 1 + holding_lengths.to_numpy(dtype=np.int64))

    if (exit_positions >= len(sessions)).any():
        raise ValueError("A planned exit falls outside the calendar.")
    
    planned_exits = pd.Series(sessions[exit_positions], index=weights.index)
    expected_signal_times = decisions.index.get_level_values("Date").map(schedule["market_close"])
    if not decisions["signal_time"].eq(expected_signal_times).all():
        raise ValueError("Decisions must be timestamped at their session close.")

    cash = initial_capital_gbp
    dividend_reserve = 0.0
    positions = {}
    marks = {}
    pending_budgets = pd.Series(dtype=float)
    pending_exit = None
    daily_rows = []
    holding_rows = []
    order_frames = []

    for calendar_position in range(decision_positions.min(), len(sessions)):
        session = sessions[calendar_position]
        interest_today = 0.0
        if previous_session is not None:
            calendar_days = (session - previous_session).days
            interest_today = cash * np.expm1(np.log1p(cash_aer) * calendar_days / 365.25)
            cash += interest_today

        previous_session = session
        market_day = market.xs(session, level="Date")
        dividends_today = 0.0

        #overnight owners recieve todays dividends before opening sales or purchases
        for ticker, position in positions.items():
            dividend = market_day.at[ticker, "dividend_gbp"]
            if not np.isfinite(dividend) or dividend < 0:
                raise ValueError(f"Unknown or invalid dividend for {ticker} on {session.date()}.")

            dividends_today += position["units"] * dividend

            #avoid counting a cumalitive dividend and the dividend itself. valid closing price will replace adjusted mark below
            marks[ticker] -= dividend

        dividend_reserve += dividends_today

        execution = execute_open_orders(cash_gbp=cash, positions=positions, market_day=market_day, buy_budgets=pending_budgets, session=session, exit_session=pending_exit, cost_per_side_bps=cost_per_side_bps, net_orders=net_orders)

        cash = execution["cash_gbp"]
        positions = execution["positions"]
        orders_today = execution["orders"]
        costs_today = 0.0

        if not orders_today.empty:
            order_frames.append(orders_today)
            costs_today = orders_today["cost_gbp"].sum()

        #refresh opening marks for additions, reduction and retained lots
            marked_positions = orders_today.loc[orders_today["status"].isin(["assumed_fill", "retained"]) & orders_today["ticker"].isin(positions)]
            for order in marked_positions.itertuples(index=False):
                marks[order.ticker] = order.price_gbp

        pending_budgets = pd.Series(dtype=float)
        pending_exit = None
        stock_value = 0.0
        stale_value = 0.0
        stale_positions = 0
        pending_exits = 0

        for ticker, position in positions.items():
            row = market_day.loc[ticker]
            close = row["close_gbp"]

            fresh_close = (not row["known_suspended"] and np.isfinite(close) and close > 0)
            if fresh_close:
                marks[ticker] = float(close)

            mark = marks[ticker]
            if not np.isfinite(mark) or mark <= 0:
                raise ValueError(f"An unresolved valuation is required for {ticker} on {session.date()}.")

            value = position["units"] * mark
            stale = not fresh_close
            stock_value += value
            stale_value += value if stale else 0.0
            stale_positions += int(stale)
            pending_exits += int(position["exit_session"] <= session)

            holding_rows.append({
                "Date": session,
                "ticker": ticker,
                "units": position["units"],
                "mark_gbp": mark,
                "value_gbp": value,
                "stale_mark": stale,
                "scheduled_exit": position["exit_session"]
            })

        #dividends contribute to wealth but not trading budget
        trading_equity = cash + stock_value
        total_equity = trading_equity + dividend_reserve

        daily_rows.append({
            "Date": session,
            "cash_gbp": cash,
            "stock_value_gbp": stock_value,
            "dividend_reserve_gbp": dividend_reserve,
            "equity_gbp": total_equity,
            "dividends_accrued_gbp": dividends_today,
            "cost_gbp": costs_today,
            "positions": len(positions),
            "pending_exits": pending_exits,
            "stale_positions": stale_positions,
            "stale_value_gbp": stale_value,
            "cash_interest_gbp": interest_today
        })

        #form tomorrows budget after on today's closing valuation
        if session in weights.index:
            pending_budgets = weights.loc[session] * trading_equity
            pending_exit = planned_exits.at[session]
            
    daily = pd.DataFrame(daily_rows).set_index("Date")
    daily["equity_return"] = daily["equity_gbp"].pct_change(fill_method=None)

    order_columns = ["Date", "ticker", "side", "status", "reason", "units","price_gbp", "notional_gbp", "cost_gbp", "cash_change_gbp", "volume_warning"]
    orders = (pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame(columns=order_columns))
    holdings = pd.DataFrame(holding_rows,
        columns=["Date", "ticker", "units", "mark_gbp", "value_gbp", "stale_mark", "scheduled_exit"]).set_index(["Date", "ticker"]).sort_index()

    return {
        "daily": daily,
        "orders": orders,
        "holdings": holdings,
        "final_positions": positions
    }