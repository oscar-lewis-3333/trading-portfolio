import numpy as np
import pandas as pd

def execute_open_orders(cash_gbp, positions, market_day, buy_budgets, session, exit_session=None, cost_per_side_bps=0.0, net_orders=False):
    #process one session using opening prices as assumed fills. buy_budgets: series (ticker index) including entry costs, positions: {ticker: "units": ..., "exit_session": ...}
    cash_gbp = float(cash_gbp)
    rate = float(cost_per_side_bps) / 10_000
    session = pd.Timestamp(session)
    budgets = pd.Series(buy_budgets, dtype=float)

    if not np.isfinite(cash_gbp) or cash_gbp < 0:
        raise ValueError("Cash must be finite and nonnegative.")
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("Cost per side must be between 0 and 10,000 bps.")
    if pd.isna(session) or session.tzinfo is not None:
        raise ValueError("Provide a valid, timezone-naive session date.")
    if not budgets.index.is_unique or budgets.index.hasnans:
        raise ValueError("Buy budgets need unique, valid ticker labels.")
    if not np.isfinite(budgets).all() or budgets.lt(0).any():
        raise ValueError("Buy budgets must be finite and nonnegative.")

    required = {
        "open_gbp",
        "known_suspended",
        "open_reference_usable",
        "positive_reported_volume"
    }
    if not required.issubset(market_day.columns):
        raise ValueError("The execution market is missing required columns.")
    if not market_day.index.is_unique:
        raise ValueError("Provide one market row per ticker.")
    for column in required - {"open_gbp"}:
        if (not pd.api.types.is_bool_dtype(market_day[column]) or market_day[column].isna().any()):
            raise ValueError(f"{column} must contain known booleans.")
        
    budgets = budgets.loc[budgets.gt(0)].sort_index()
    if not budgets.empty:
        exit_session = pd.Timestamp(exit_session)
        if (pd.isna(exit_session) or exit_session.tzinfo is not None or exit_session <= session):
            raise ValueError("New positions need a later exit session.")

    #copy state so caller's previous state is unchanged
    positions = {ticker: dict(position) for ticker, position in positions.items()}

    for position in positions.values():
        position["units"] = float(position["units"])
        position["exit_session"] = pd.Timestamp(position["exit_session"])
        if (not np.isfinite(position["units"]) or position["units"] <= 0 or pd.isna(position["exit_session"]) or position["exit_session"].tzinfo is not None):
            raise ValueError("A position has invalid units or exit timing.")

    events = []
    def record(ticker, side, status, **details):
        events.append({
            "Date": session,
            "ticker": ticker,
            "side": side,
            "status": status,
            "reason": "",
            "units": 0.0,
            "price_gbp": np.nan,
            "notional_gbp": 0.0,
            "cost_gbp": 0.0,
            "cash_change_gbp": 0.0,
            "volume_warning": False,
            **details
        })

    def opening_quote(ticker):
        if ticker not in market_day.index:
            return None, "missing_market_row", False

        row = market_day.loc[ticker]
        warning = not row["positive_reported_volume"]
        price = row["open_gbp"]

        if row["known_suspended"]:
            return None, "suspended", warning
        if (not row["open_reference_usable"] or not np.isfinite(price) or price <= 0):
            return None, "unusable_open", warning
        return float(price), "", warning

    if not isinstance(net_orders, bool):
        raise ValueError("net_orders must be a boolean.")

    #now avoid selling and buying the same stock in the same rebalance (if we are selling 20 of something, and buying 10 of it again, then just sell 10)
    if net_orders:
        quotes = {}
        due_units = {}
        target_budgets = {}


        #only positions with a due exit (that's allowed) can be released
        for ticker, position in sorted(positions.items()):
            if position["exit_session"] > session:
                continue

            price, reason, warning = opening_quote(ticker)
            if price is None:
                record(ticker, "sell", "blocked", reason=reason, volume_warning=warning)
                continue

            quotes[ticker] = (price, warning)
            due_units[ticker] = position["units"]

        for ticker, budget in budgets.items():
            if ticker in positions and ticker not in due_units:
                record(ticker, "buy", "cancelled", reason="already_held")
                continue

            price, reason, warning = opening_quote(ticker)
            if price is None:
                record(ticker, "buy", "cancelled", reason=reason, volume_warning=warning)
                continue

            quotes[ticker] = (price, warning)
            target_budgets[ticker] = float(budget)

        tickers = sorted(set(due_units) | set(target_budgets))

        current_values = np.array([due_units.get(ticker, 0.0) * quotes[ticker][0] for ticker in tickers], dtype=float)
        requested_budgets = np.array([target_budgets.get(ticker, 0.0) for ticker in tickers], dtype=float)

        #cancelled allocations remain included (slots stay in cash)
        total_budget = float(budgets.sum())
        available_capital = cash_gbp + current_values.sum()

        if (not np.isfinite(total_budget) or not np.isfinite(available_capital)):
            raise ValueError("Nonfinite funding requirement.")

        def funding_required(scale):
            reductions = np.maximum(current_values - scale * requested_budgets, 0.0)
            return scale * total_budget + rate * reductions.sum()

        #buy costs included in allocation budget. sell costs require additional funding
        if total_budget == 0:
            scale = 0.0
        elif funding_required(1.0) <= available_capital:
            scale = 1.0
        else:
            lower, upper = 0.0, 1.0
            for _ in range(60):
                midpoint = (lower + upper) / 2
                if funding_required(midpoint) <= available_capital:
                    lower = midpoint
                else:
                    upper = midpoint
            scale = lower

        scaled_budgets = scale * requested_budgets

        #target + fee on addition = scaled allocation budget
        target_values = np.where(scaled_budgets <= current_values, scaled_budgets, (scaled_budgets + rate * current_values) / (1.0 + rate))

        next_units = {}
        for ticker, value in zip(tickers, target_values):
            units = float(value / quotes[ticker][0])
            old_units = due_units.get(ticker, 0.0)

            #avoid negligible trades caused solely by rounding floats
            if (old_units > 0 and units > 0 and np.isclose(units, old_units, rtol=1e-12, atol=0.0)):
                units = old_units

            next_units[ticker] = units


        #execute reductions, then afterwards fund additions from cash
        for side in ["sell", "buy"]:
            for ticker in tickers:
                change = next_units[ticker] - due_units.get(ticker, 0.0)

                if ((side == "sell" and change >= 0) or (side == "buy" and change <= 0)):
                    continue

                price, warning = quotes[ticker]
                traded_units = abs(change)
                notional = traded_units * price
                cost = notional * rate

                cash_change = notional - cost if side == "sell" else -(notional + cost)
                cash_gbp += cash_change

                record(ticker, side, "assumed_fill", units=traded_units, price_gbp=price, notional_gbp=notional, cost_gbp=cost, cash_change_gbp=cash_change, volume_warning=warning)

        for ticker in tickers:
            units = next_units[ticker]
            old_units = due_units.get(ticker, 0.0)

            if units > 0:
                positions[ticker] = {
                    "units": units,
                    "exit_session": exit_session
                }

                if units == old_units:
                    price, warning = quotes[ticker]
                    record(ticker, "hold", "retained", units=units, price_gbp=price, volume_warning=warning)
            else:
                positions.pop(ticker, None)
                if ticker in target_budgets:
                    record(ticker, "buy", "cancelled", reason="no_cash")

        tolerance = 1e-10 * max(1.0, available_capital)
        if cash_gbp < -tolerance:
            raise ArithmeticError("Netted orders exceeded available cash.")

        return {
            "cash_gbp": max(0.0, cash_gbp),
            "positions": positions,
            "orders": pd.DataFrame(events),
            "budget_scale": scale
        }

    #failed exit stays due. can be attempted next session
    for ticker in sorted(list(positions)):
        position = positions[ticker]
        if position["exit_session"] > session:
            continue

        price, reason, warning = opening_quote(ticker)
        if price is None:
            record(ticker, "sell", "blocked", reason=reason, volume_warning=warning)
            continue
        notional = position["units"] * price
        cost = notional * rate
        proceeds = notional - cost
        cash_gbp += proceeds

        record(ticker, "sell", "assumed_fill", units=position["units"], price_gbp=price, notional_gbp=notional, cost_gbp=cost, cash_change_gbp=proceeds, volume_warning=warning)
        del positions[ticker]

    #scale planned budgets together if sale proceeds insufficient. cancelled buys leave allocation in cash
    total_budget = float(budgets.sum())
    scale = min(1.0, cash_gbp / total_budget) if total_budget else 0.0
    cash_before_buys = cash_gbp
    for ticker, budget in budgets.items():
        if ticker in positions:
            record(ticker, "buy", "cancelled", reason="already_held")
            continue

        price, reason, warning = opening_quote(ticker)
        if price is None:
            record(ticker, "buy", "cancelled", reason=reason, volume_warning=warning)
            continue

        spend = float(budget) * scale
        if spend <= 0:
            record(ticker, "buy", "cancelled", reason="no_cash")
            continue

        notional = spend / (1.0 + rate)
        cost = spend - notional
        units = notional / price
        cash_gbp -= spend

        positions[ticker] = {
            "units": units,
            "exit_session": exit_session
        }

        record(ticker, "buy", "assumed_fill", units=units, price_gbp=price, notional_gbp=notional, cost_gbp=cost, cash_change_gbp=-spend, volume_warning=warning)

    tolerance = 1e-10 * max(1.0, cash_before_buys)
    if cash_gbp < -tolerance:
        raise ArithmeticError("Purchases exceeded available cash.")

    return {
        "cash_gbp": max(0.0, cash_gbp),
        "positions": positions,
        "orders": pd.DataFrame(events),
        "budget_scale": scale
    }

def run_execution_backtest(decisions, market, schedule, initial_capital_gbp=10_000, holding_sessions=5, cost_per_side_bps=0.0, net_orders=False, *, holding_sessions_by_date=None):
    #simulate opening trades, closing values and dividends reserved
    initial_capital_gbp = float(initial_capital_gbp)

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
            "stale_value_gbp": stale_value
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