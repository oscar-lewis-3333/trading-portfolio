
import numpy as np
import pandas as pd
import json

from broker import decisions_to_orders
from reporting import latest_rebalance_summary, load_rebalance_history

from contextlib import contextmanager

@contextmanager
def bot_run_lock():
    #bot is only run when this is called.
    from pathlib import Path
    import fcntl

    lock_path = (Path(__file__).resolve().parents[1] / 'data' / 'bot_run.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another bot run is already active; this run cannot proceed") from error
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def select_momentum_basket(close_prices, tickers, signal_date, lookback_days=63, top_frac=0.20): #this is the collection of tickers we select every 21 days based on 63d momentum.
    #close_prices: dates as index, tickers as columns, closing prices as values
    #signal_date must be a completed trading session, checked by the caller

    if isinstance(tickers, str):
        raise TypeError("Tickers must be a collection, not one string")
    tickers = list(tickers)
    if any(not isinstance(ticker, str) or not ticker for ticker in tickers):
        raise ValueError("Tickers must be non-empty strings")
    if len(tickers) < 5 or len(set(tickers)) != len(tickers):
        raise ValueError("Provide at least five distinct tickers")
    if (not isinstance(lookback_days, (int, np.integer)) or isinstance(lookback_days, bool) or lookback_days < 1):
        raise ValueError("Lookback days must be a positive integer")
    if not 0 < top_frac <= 1:
        raise ValueError("Top fraction must be greater than 0 and at most 1")
    if not isinstance(close_prices.index, pd.DatetimeIndex):
        raise TypeError("Closing prices must have a DatetimeIndex")
    if close_prices.index.tz is not None:
        raise ValueError("Use timezone-naive trading-session dates")
    if (close_prices.index.hasnans or close_prices.index.has_duplicates or not close_prices.index.is_monotonic_increasing or close_prices.columns.has_duplicates):
        raise ValueError("Price dates must be valid, unique and chronological")

    signal_date = pd.Timestamp(signal_date)

    if (pd.isna(signal_date) or signal_date.tzinfo is not None or signal_date != signal_date.normalize()):
        raise ValueError("Signal date must be a timezone-naive session date")
    missing = set(tickers).difference(close_prices.columns)
    if missing:
        raise ValueError(f"Missing universe tickers: {sorted(missing)}")

    #exclude observations after the signal date; never shrink the universe
    history = close_prices.loc[close_prices.index <= signal_date, tickers].tail(lookback_days + 1)

    if len(history) < lookback_days + 1:
        raise ValueError("Insufficient price history for the momentum lookback")

    if history.index[-1] != signal_date:
        raise ValueError("Prices do not reach the required signal date")

    history = history.astype(float)

    if (not np.isfinite(history.to_numpy()).all() or (history <= 0).any().any()):
        raise ValueError("Every ticker needs complete, positive, finite prices")

    momentum = history.iloc[-1] / history.iloc[0] - 1

    ranked = pd.DataFrame({
        'ticker': momentum.index,
        'momentum': momentum.to_numpy(),
        'signal_close': history.iloc[-1].to_numpy(),
    }).sort_values(['momentum', 'ticker'], ascending=[False, True])

    n_selected = max(1, int(len(tickers) * top_frac))
    selected = ranked.head(n_selected).copy().reset_index(drop=True)

    selected['target_weight'] = 1.0 / n_selected
    selected['signal_date'] = signal_date

    return selected

def generate_trading_decisions(tickers, signal_date, trading_dates, top_frac=0.20, rebalance_days=21, lookback_days=63, state_path=None, period="1y"):

    #generate trading choices for each rebalance (if one due). rank by 63d momentum every 21d, selecting top_frac. does not submit orders yet.
    #trading_dates comes from the exchange calendar
    #signal_date must be the latest completed trading session

    from pathlib import Path
    from data_loader import fetch_price_data

    #start by checking errors and ensuring all data is inputted correctly

    if isinstance(tickers, str):
        raise TypeError("Tickers must be a collection, not one string")

    tickers = list(tickers)
    if any(not isinstance(ticker, str) or not ticker for ticker in tickers):
        raise ValueError("Tickers must be non-empty strings")

    if len(tickers) < 5 or len(set(tickers)) != len(tickers):
        raise ValueError("Provide at least five distinct tickers")

    for name, value in (('lookback_days', lookback_days), ('rebalance_days', rebalance_days)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1):
            raise ValueError(f"{name} must be a positive integer")

    if not 0 < top_frac <= 1:
        raise ValueError("Top fraction must be greater than 0 and at most 1")


    signal_date = pd.Timestamp(signal_date)
    sessions = pd.DatetimeIndex(trading_dates)

    if (pd.isna(signal_date) or signal_date.tzinfo is not None or signal_date != signal_date.normalize()):
        raise ValueError("Signal date must be a timezone-naive session date")

    if (sessions.tz is not None or sessions.hasnans or sessions.has_duplicates or not sessions.is_monotonic_increasing or not sessions.equals(sessions.normalize())):
        raise ValueError("Trading dates must be unique, chronological session dates")

    sessions = sessions[sessions <= signal_date]

    if len(sessions) < lookback_days + 1:
        raise ValueError("Calendar does not cover the momentum lookback")
    if sessions[-1] != signal_date:
        raise ValueError("Signal date is not in the supplied trading calendar")

    #loading state

    if state_path is None:
        state_file = (Path(__file__).resolve().parents[1] / 'data' / 'basket_state.json')
    else:
        state_file = Path(state_path).expanduser().resolve()

    state = None
    if state_file.exists():
        with state_file.open() as file:
            state = json.load(file)
        if not isinstance(state, dict):
            raise ValueError("Saved basket state must be a dictionary")

    #record data so old state cannot be reused 
    strategy = {'lookback_days': int(lookback_days),
        'top_frac': float(top_frac),
        'rebalance_days': int(rebalance_days),
        'universe': sorted(tickers),
        'weighting': 'equal_at_rebalance'}

    state_matches = (state is not None and state.get('schema_version') == 2 and state.get('strategy') == strategy)

    #now decide whether rebalance is due
    trading_days_since = None
    current_basket = []
    if state_matches:
        last_selection = pd.Timestamp(state['selection_date'])

        if last_selection not in sessions:
            raise ValueError("Calendar must include the saved selection date")

        current_basket = state.get('tickers', [])
        expected_count = max(1, int(len(tickers) * top_frac))

        if (not isinstance(current_basket, list) or any(not isinstance(t, str) for t in current_basket) or len(current_basket) != expected_count or len(set(current_basket)) != expected_count or not set(current_basket).issubset(tickers)):
            raise ValueError("Saved basket is inconsistent with the strategy")

        trading_days_since = int((sessions > last_selection).sum())

        if trading_days_since > rebalance_days:
            scheduled_signal_date = sessions[sessions > last_selection][rebalance_days - 1]

            raise RuntimeError(f"The rebalance based on the {scheduled_signal_date.date()} close was missed. Refusing to silently re-anchor the strategy.")

        rebalance_needed = trading_days_since == rebalance_days
        reason = 'scheduled_rebalance' if rebalance_needed else 'basket_unchanged'
    else:
        rebalance_needed = True
        reason = 'initial_selection' if state is None else 'strategy_changed'

    plan = {
        'signal_date': signal_date.strftime('%Y-%m-%d'),
        'rebalance_needed': rebalance_needed,
        'reason': reason,
        'trading_days_since': trading_days_since,
        'current_basket': current_basket,
        'state_path': str(state_file),
        'proposed_state': None,
    }

    #holding the basket means holding share quantities, not restoring weights
    if not rebalance_needed:
        return None, plan
    
    #align tickers to exchange sessions
    required_dates = sessions[-(lookback_days + 1):]
    price_data = {}

    for ticker in tickers:
        df = fetch_price_data(ticker, period=period)
        if df.empty or 'Close' not in df.columns:
            raise ValueError(f"No closing prices returned for {ticker}")

        close = df['Close'].copy()
        if not isinstance(close, pd.Series):
            raise ValueError(f"Expected one closing-price series for {ticker}")

        #daily bars are labelled by their exchange-session date
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        if close.index.hasnans or close.index.has_duplicates:
            raise ValueError(f"Invalid or duplicate price dates for {ticker}")

        close = pd.to_numeric(close, errors='raise')
        price_data[ticker] = close.reindex(required_dates)

    close_prices = pd.DataFrame(price_data, index=required_dates)

    #missing sessions remain NaN and are rejected by selector
    decisions = select_momentum_basket(close_prices=close_prices, tickers=tickers, signal_date=signal_date, lookback_days=lookback_days, top_frac=top_frac)

    #propose next state

    plan['proposed_state'] = {'schema_version': 2,
        'strategy': strategy,
        'selection_date': signal_date.strftime('%Y-%m-%d'),
        'tickers': decisions['ticker'].tolist()}
    return decisions, plan
def rebalance_summary(history_path=None):
    rebalances, orders = load_rebalance_history(history_path=history_path)
    return latest_rebalance_summary(rebalances, orders)

def build_rebalance_preview(client, tickers, env_path, state_path, lookback_days=63, top_frac=0.20, rebalance_days=21, execution_window_minutes=15, price_max_age_seconds=60, cash_reserve_fraction=0.01):
    #calculate the next plan without saving or submitting

    import math
    from pathlib import Path
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.requests import GetCalendarRequest
    from broker import get_data_client

    if (not isinstance(execution_window_minutes, int) or isinstance(execution_window_minutes, bool) or execution_window_minutes < 1):
        raise ValueError("Execution window must be a positive integer")

    if isinstance(price_max_age_seconds, bool):
        raise ValueError("Maximum price age must be numeric")

    price_max_age_seconds = float(price_max_age_seconds)
    if not math.isfinite(price_max_age_seconds) or price_max_age_seconds <= 0:
        raise ValueError("Maximum price age must be positive")

    if isinstance(cash_reserve_fraction, bool):
        raise ValueError("Cash reserve fraction must be numeric")

    cash_reserve_fraction = float(cash_reserve_fraction)
    if (not math.isfinite(cash_reserve_fraction) or not 0 <= cash_reserve_fraction < 1):
        raise ValueError("Cash reserve fraction must be between 0 and 1")

    state_file = Path(state_path).expanduser().resolve()
    clock = client.get_clock()
    broker_time = pd.Timestamp(clock.timestamp)

    if broker_time.tzinfo is None:
        raise ValueError("Broker timestamp must be timezone-aware")

    broker_time = broker_time.tz_convert('America/New_York')

    #request sufficient history for the momentum calculation
    calendar_start = (broker_time - pd.Timedelta(days=400)).date() #at least 63 days

    if state_file.exists():
        with state_file.open(encoding='utf-8') as file:
            existing_state = json.load(file)

        if not isinstance(existing_state, dict):
            raise ValueError("Existing basket state must be a dictionary")

        stored_date = existing_state.get('selection_date')

        if stored_date is not None:
            stored_date = pd.Timestamp(stored_date)

            if (pd.isna(stored_date) or stored_date.tzinfo is not None or stored_date != stored_date.normalize()):
                raise ValueError("Invalid stored selection date")

            calendar_start = min(calendar_start, (stored_date - pd.Timedelta(days=7)).date())

    calendar_end = (broker_time + pd.Timedelta(days=14)).date()

    calendar = client.get_calendar(filters=GetCalendarRequest(start=calendar_start, end=calendar_end))
    if not calendar:
        raise RuntimeError("No exchange calendar was returned")

    schedule_rows = []

    for session in calendar:
        session_open = pd.Timestamp(session.open)
        session_close = pd.Timestamp(session.close)

        if session_open.tzinfo is None:
            session_open = session_open.tz_localize('America/New_York')
        else:
            session_open = session_open.tz_convert('America/New_York')
        if session_close.tzinfo is None:
            session_close = session_close.tz_localize('America/New_York')
        else:
            session_close = session_close.tz_convert('America/New_York')

        schedule_rows.append({
            'date': pd.Timestamp(session.date).normalize(),
            'open': session_open,
            'close': session_close
        })

    schedule = pd.DataFrame(schedule_rows).sort_values('date').reset_index(drop=True)
    if schedule['date'].duplicated().any():
        raise ValueError("Exchange calendar contains duplicate dates")

    completed = schedule.loc[schedule['close'] <= broker_time]
    if completed.empty:
        raise RuntimeError("No completed signal session was found")

    signal_date = completed.iloc[-1]['date']
    future_sessions = schedule.loc[schedule['date'] > signal_date]
    if future_sessions.empty:
        raise RuntimeError("Next execution session was not found")

    execution_session = future_sessions.iloc[0]
    execution_date = execution_session['date']
    execution_open = execution_session['open']
    execution_close = execution_session['close']

    execution_end = min(execution_open + pd.Timedelta(minutes=execution_window_minutes), execution_close)

    calendar_gate = bool(clock.is_open and execution_open <= broker_time < execution_end)

    trading_dates = pd.DatetimeIndex(schedule['date'])

    decisions, plan = generate_trading_decisions(tickers=tickers, signal_date=signal_date, trading_dates=trading_dates, lookback_days=lookback_days, top_frac=top_frac, rebalance_days=rebalance_days, state_path=state_file)

    plan['execution_policy'] = {
        'cash_reserve_fraction': cash_reserve_fraction,
        'execution_window_minutes': int(execution_window_minutes),
        'price_max_age_seconds': float(price_max_age_seconds),
        'order_type': 'market',
        'time_in_force': 'day',
        'extended_hours': False,
    }

    result = {
        'status': 'holding_existing_basket',
        'signal_date': signal_date,
        'execution_date': execution_date,
        'execution_open': execution_open,
        'execution_window_end': execution_end,
        'calendar_gate': calendar_gate,
        'execution_allowed': False,
        'decisions': decisions,
        'plan': plan,
        'orders': [],
        'current_positions': {},
        'account_id': None,
        'account_snapshot': None,
        'bid_prices': {},
        'ask_prices': {},
        'price_ages_seconds': {},
        'blockers': [],
    }
    #no rebalance, then no data or work left to be done
    if not plan['rebalance_needed']:
        return result

    account = client.get_account()
    account_status = getattr(account.status, 'value', account.status)

    blockers = []
    if account_status != 'ACTIVE':
        blockers.append(f"account status is {account_status}")

    for attribute in ('account_blocked', 'trading_blocked', 'trade_suspended_by_user'):
        if bool(getattr(account, attribute, False)):
            blockers.append(attribute)

    equity_value = account.equity if account.equity is not None else account.portfolio_value
    equity_value = float(equity_value)
    cash = float(account.cash)

    if (not math.isfinite(equity_value) or equity_value <= 0 or not math.isfinite(cash)):
        raise ValueError("Invalid account values")

    current_positions = {}
    available_positions = {}

    for position in client.get_all_positions():
        ticker = position.symbol
        side = getattr(position.side, 'value', position.side)
        quantity = float(position.qty)
        available = float(position.qty_available) if position.qty_available is not None else quantity

        if (ticker in current_positions or side != 'long' or not math.isfinite(quantity) or quantity <= 0 or not math.isfinite(available) or available < 0 or available > quantity + 1e-9):
            raise RuntimeError(f"Unexpected broker position: {ticker}")

        current_positions[ticker] = quantity
        available_positions[ticker] = available

    required_symbols = sorted(set(current_positions).union(decisions['ticker']))

    data_client = get_data_client(env_path)
    quotes = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=required_symbols, feed=DataFeed.IEX))

    missing_quotes = set(required_symbols).difference(quotes)
    if missing_quotes:
        raise RuntimeError(f"Missing quotes: {sorted(missing_quotes)}")

    bid_prices = {}
    ask_prices = {}
    price_ages = {}
    fresh_prices = True

    for ticker in required_symbols:
        quote = quotes[ticker]
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        quote_time = pd.Timestamp(quote.timestamp)

        if (not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0 or ask < bid):
            raise ValueError(f"Invalid quote for {ticker}")

        if quote_time.tzinfo is None:
            raise ValueError(f"Quote timestamp for {ticker} lacks a timezone")

        quote_time = quote_time.tz_convert('America/New_York')
        age = (broker_time - quote_time).total_seconds()
        if age < -5:
            raise ValueError(f"Quote timestamp for {ticker} is in the future")

        bid_prices[ticker] = bid
        ask_prices[ticker] = ask
        price_ages[ticker] = age

        if not (execution_open <= quote_time <= broker_time and 0 <= age <= price_max_age_seconds):
            fresh_prices = False

    orders = decisions_to_orders(decisions_df=decisions, current_positions=current_positions, available_cash=cash, bid_prices=bid_prices, ask_prices=ask_prices, managed_tickers=tickers, cash_reserve_fraction=cash_reserve_fraction)

    planned_positions = {ticker: float(quantity) for ticker, quantity in current_positions.items() if float(quantity) > 0}

    for order in orders:
        ticker = order['ticker']
        change = float(order['qty']) if order['side'] == 'buy' else -float(order['qty'])
        final_quantity = round(planned_positions.get(ticker, 0.0) + change, 9)

        if final_quantity < 0:
            raise RuntimeError(f"The plan would create a negative holding for {ticker}")
        if final_quantity == 0:
            planned_positions.pop(ticker, None)
        else:
            planned_positions[ticker] = final_quantity

    missing_selected = set(decisions['ticker']).difference(planned_positions)
    if missing_selected:
        blockers.append(f"insufficient capital to hold: {sorted(missing_selected)}")

    for order in orders:
        asset = client.get_asset(order['ticker'])
        quantity = float(order['qty'])

        if not bool(asset.tradable):
            blockers.append(f"{order['ticker']} is not tradable")

        fractional = not math.isclose(quantity, round(quantity), rel_tol=0.0, abs_tol=1e-9)
        if fractional and not bool(asset.fractionable):
            blockers.append(f"{order['ticker']} does not permit fractional orders")

        if order['side'] == 'sell' and available_positions.get(order['ticker'], 0.0) + 1e-9 < quantity:
            blockers.append(f"insufficient available shares of {order['ticker']}")

    execution_allowed = bool(calendar_gate and fresh_prices and not blockers)

    if blockers:
        status = 'execution_blocked'
    elif not calendar_gate:
        status = 'outside_execution_window'
    elif not fresh_prices:
        status = 'waiting_for_fresh_prices'
    else:
        status = 'ready_to_prepare'

    result.update({
        'status': status,
        'execution_allowed': execution_allowed,
        'orders': orders,
        'current_positions': current_positions,
        'account_id': str(account.id),
        'account_snapshot': {
            'cash': cash,
            'equity': equity_value,
        },
        'bid_prices': bid_prices,
        'ask_prices': ask_prices,
        'price_ages_seconds': price_ages,
        'blockers': blockers,
    })

    return result

def advance_pending_rebalance(client, pending_path, env_path, max_wait_seconds=45, poll_interval_seconds=5):
    #strictly serial: reconcile, validate, submit one, repeat
    import math
    import time
    from pathlib import Path
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.requests import GetCalendarRequest
    from broker import get_data_client, reconcile_pending_orders, submit_pending_order, complete_rebalance, check_pending_orders, mark_pending_attention

    if isinstance(max_wait_seconds, bool) or not isinstance(max_wait_seconds, (int, float)) or not math.isfinite(max_wait_seconds) or max_wait_seconds < 0:
        raise ValueError("Maximum wait cannot be negative")
    if isinstance(poll_interval_seconds, bool) or not isinstance(poll_interval_seconds, (int, float)) or not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
        raise ValueError("Polling interval must be positive")

    pending_file = Path(pending_path).expanduser().resolve()
    data_client = get_data_client(env_path)
    deadline = time.monotonic() + max_wait_seconds

    with pending_file.open(encoding='utf-8') as file:
        initial_pending = json.load(file)

    if initial_pending.get('schema_version') != 2:
        raise ValueError("Unsupported pending rebalance format")

    execution_policy = initial_pending.get('execution_policy')
    if not isinstance(execution_policy, dict):
        raise ValueError("Pending rebalance lacks an execution policy")

    execution_window_minutes = execution_policy.get('execution_window_minutes')
    raw_price_max_age_seconds = execution_policy.get('price_max_age_seconds')

    if (not isinstance(execution_window_minutes, int) or isinstance(execution_window_minutes, bool) or execution_window_minutes < 1):
        raise ValueError("Invalid pending execution window")
    if isinstance(raw_price_max_age_seconds, bool) or not isinstance(raw_price_max_age_seconds, (int, float)):
        raise ValueError("Invalid pending maximum price age")
    price_max_age_seconds = float(raw_price_max_age_seconds)
    if not math.isfinite(price_max_age_seconds) or price_max_age_seconds <= 0:
        raise ValueError("Invalid pending maximum price age")

    execution_date = pd.Timestamp(initial_pending['execution_date'])

    if (pd.isna(execution_date) or execution_date.tzinfo is not None or execution_date != execution_date.normalize()):
        raise ValueError("Invalid execution date")

    calendar = client.get_calendar(filters=GetCalendarRequest(start=execution_date.date(), end=execution_date.date()))

    if len(calendar) != 1:
        raise RuntimeError("Planned execution session was not found")

    execution_open = pd.Timestamp(calendar[0].open)
    execution_close = pd.Timestamp(calendar[0].close)

    if execution_open.tzinfo is None:
        execution_open = execution_open.tz_localize('America/New_York')
    else:
        execution_open = execution_open.tz_convert('America/New_York')

    if execution_close.tzinfo is None:
        execution_close = execution_close.tz_localize('America/New_York')
    else:
        execution_close = execution_close.tz_convert('America/New_York')

    execution_end = min(execution_open + pd.Timedelta(minutes=execution_window_minutes), execution_close)

    saved_execution_open = pd.Timestamp(initial_pending['execution_open'])
    saved_execution_end = pd.Timestamp(initial_pending['execution_window_end'])

    if saved_execution_open.tzinfo is None or saved_execution_end.tzinfo is None:
        raise ValueError("Saved execution times must be timezone-aware")

    saved_execution_open = saved_execution_open.tz_convert('America/New_York')
    saved_execution_end = saved_execution_end.tz_convert('America/New_York')

    if saved_execution_open != execution_open or saved_execution_end != execution_end:
        raise RuntimeError("Saved execution window differs from the exchange calendar")

    while True:
        pending = reconcile_pending_orders(client, pending_file)

        if pending['status'] == 'needs_attention':
            raise RuntimeError("The pending rebalance requires manual attention")

        #filled orders may be completed even after the opening window
        if pending['status'] == 'orders_filled':
            completed_record, completed = complete_rebalance(client, pending_file)

            if completed:
                return {
                    'mode': 'paper_execution',
                    'status': 'completed',
                    'pending': completed_record
                }

            if time.monotonic() >= deadline:
                return {
                    'mode': 'paper_execution',
                    'status': 'waiting_for_completion',
                    'pending': completed_record
                }

            time.sleep(poll_interval_seconds)
            continue

        clock = client.get_clock()
        broker_time = pd.Timestamp(clock.timestamp)

        if broker_time.tzinfo is None:
            raise ValueError("Broker timestamp must be timezone-aware")

        broker_time = broker_time.tz_convert('America/New_York')

        #an empty plan still needs confirmation during its planned window
        if pending['status'] == 'no_orders':
            already_confirmed = pending.get('no_order_confirmed_at') is not None

            if (not already_confirmed and
                    not (clock.is_open and execution_open <= broker_time < execution_end)):
                if broker_time >= execution_end:
                    reason = "The no-order rebalance missed its execution window"
                    mark_pending_attention(pending_file, reason, code='missed_execution_window')
                    raise RuntimeError(reason)

                return {
                    'mode': 'paper_execution',
                    'status': 'waiting_for_execution_window',
                    'pending': pending
                }

            completed_record, completed = complete_rebalance(client, pending_file)
            return {
                'mode': 'paper_execution',
                'status': ('completed' if completed else 'waiting_for_completion'),
                'pending': completed_record
            }

        active_orders = [order for order in pending['orders'] if order['status'] not in {'filled', 'not_submitted'}]

        #never submit another order while an earlier one is active
        if active_orders:
            if time.monotonic() >= deadline:
                return {
                    'mode': 'paper_execution',
                    'status': 'orders_in_progress',
                    'pending': pending
                }

            time.sleep(poll_interval_seconds)
            continue

        unsent_orders = [order for order in pending['orders'] if order['status'] == 'not_submitted']

        if not unsent_orders:
            if time.monotonic() >= deadline:
                return {
                    'mode': 'paper_execution',
                    'status': 'waiting_for_reconciliation',
                    'pending': pending
                }

            time.sleep(poll_interval_seconds)
            continue

        if not (clock.is_open and execution_open <= broker_time < execution_end):
            if broker_time >= execution_end:
                reason = "Unsubmitted orders remain after the execution window"
                mark_pending_attention(pending_file, reason, code='missed_execution_window')
                raise RuntimeError(reason)

            return {
                'mode': 'paper_execution',
                'status': 'waiting_for_execution_window',
                'pending': pending
            }

        has_open, open_orders = check_pending_orders(client)

        if has_open:
            saved_ids = {order['client_order_id'] for order in pending['orders'] }
            unexpected = [str(order.id) for order in open_orders if order.client_order_id not in saved_ids]

            if unexpected:
                reason = f"Unrelated open orders found: {unexpected}"
                mark_pending_attention(pending_file, reason, code='unexpected_open_orders')
                raise RuntimeError(reason)

            if time.monotonic() >= deadline:
                return {
                    'mode': 'paper_execution',
                    'status': 'orders_in_progress',
                    'pending': pending
                }

            time.sleep(poll_interval_seconds)
            continue

        #derive the holdings that should exist after recorded fills
        expected_now = {ticker: float(quantity) for ticker, quantity in pending['starting_positions'].items()}

        for order in pending['orders']:
            if order['status'] != 'filled':
                continue

            change = float(order['filled_qty']) if order['side'] == 'buy' else -float(order['filled_qty'])

            final_quantity = (expected_now.get(order['ticker'], 0.0) + change)
            final_quantity = round(final_quantity, 9)

            if final_quantity < 0:
                reason = f"Filled orders produced a negative holding for {order['ticker']}"
                mark_pending_attention(pending_file, reason, code='negative_expected_holding')
                raise RuntimeError(reason)

            if final_quantity == 0:
                expected_now.pop(order['ticker'], None)
            else:
                expected_now[order['ticker']] = final_quantity

        actual_now = {}
        available_now = {}

        for position in client.get_all_positions():
            ticker = position.symbol
            side = getattr(position.side, 'value', position.side)
            quantity = float(position.qty)

            if (ticker in actual_now or side != 'long' or not math.isfinite(quantity) or quantity <= 0):
                reason = f"Unexpected broker position: {ticker}"
                mark_pending_attention(pending_file, reason, code='invalid_broker_position')
                raise RuntimeError(reason)

            available = float(position.qty_available) if position.qty_available is not None else quantity

            if not math.isfinite(available) or available < 0 or available > quantity + 1e-9:
                reason = f"Unexpected available quantity for {ticker}"
                mark_pending_attention(pending_file, reason, code='invalid_broker_position')
                raise RuntimeError(reason)

            actual_now[ticker] = quantity
            available_now[ticker] = available

        mismatches = [ticker for ticker in sorted(set(actual_now).union(expected_now))
            if (ticker not in actual_now or ticker not in expected_now or not math.isclose(actual_now[ticker], expected_now[ticker], rel_tol=0.0,abs_tol=1e-9,))]

        if mismatches:
            if time.monotonic() < deadline:
                time.sleep(poll_interval_seconds)
                continue

            return {
                'mode': 'paper_execution',
                'status': 'holdings_not_yet_reconciled',
                'mismatches': mismatches,
                'pending': pending
            }

        next_order = unsent_orders[0]
        ticker = next_order['ticker']
        quantity = float(next_order['qty'])
        asset = client.get_asset(ticker)
        if not asset.tradable:
            reason = f"{ticker} is not tradable"
            mark_pending_attention(pending_file, reason, code='asset_not_tradable')
            raise RuntimeError(reason)

        fractional = not math.isclose(quantity, round(quantity), rel_tol=0.0, abs_tol=1e-9)

        if fractional and not asset.fractionable:
            reason = f"{ticker} does not permit fractional orders"
            mark_pending_attention(pending_file, reason, code='fractional_order_not_supported')
            raise RuntimeError(reason)

        account = client.get_account()
        account_status = getattr(account.status, 'value', account.status)

        if (account_status != 'ACTIVE' or bool(account.account_blocked) or bool(account.trading_blocked) or bool(account.trade_suspended_by_user)):
            reason = "The account cannot currently trade"
            mark_pending_attention(pending_file, reason, code='account_cannot_trade')
            raise RuntimeError(reason)

        if next_order['side'] == 'sell':
            if available_now.get(ticker, 0.0) + 1e-9 < quantity:
                reason = f"Insufficient available shares of {ticker}"
                mark_pending_attention(pending_file, reason, code='insufficient_available_shares')
                raise RuntimeError(reason)

        else:
            #validate all remaining buys against current cash
            remaining_buys = [order for order in unsent_orders if order['side'] == 'buy']
            buy_tickers = sorted({order['ticker'] for order in remaining_buys})

            quotes = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=buy_tickers, feed=DataFeed.IEX))

            if set(buy_tickers).difference(quotes):
                raise RuntimeError("Quotes are missing for remaining buys")

            ask_prices = {}
            fresh_prices = True
            for symbol in buy_tickers:
                quote = quotes[symbol]
                ask = float(quote.ask_price)
                quote_time = pd.Timestamp(quote.timestamp)

                if (not math.isfinite(ask) or ask <= 0 or quote_time.tzinfo is None):
                    raise ValueError(f"Invalid quote for {symbol}")

                quote_time = quote_time.tz_convert('America/New_York')
                age = (broker_time - quote_time).total_seconds()

                if not (execution_open <= quote_time <= broker_time and 0 <= age <= price_max_age_seconds):
                    fresh_prices = False
                    break
                ask_prices[symbol] = ask

            if not fresh_prices:
                if time.monotonic() < deadline:
                    time.sleep(poll_interval_seconds)
                    continue

                return {
                    'mode': 'paper_execution',
                    'status': 'waiting_for_fresh_prices',
                    'pending': pending
                }

            estimated_cost = sum(float(order['qty']) * ask_prices[order['ticker']] for order in remaining_buys)
            available_cash = float(account.cash)

            if (not math.isfinite(available_cash) or available_cash < 0):
                reason = "Invalid available cash"
                mark_pending_attention(pending_file, reason, code='invalid_available_cash')
                raise RuntimeError(reason)

            if estimated_cost > available_cash:
                if time.monotonic() < deadline:
                    time.sleep(poll_interval_seconds)
                    continue

                return {
                    'mode': 'paper_execution',
                    'status': 'insufficient_cash',
                    'estimated_remaining_cost': estimated_cost,
                    'available_cash': available_cash,
                    'pending': pending
                }

        pending, submitted = submit_pending_order(client=client, pending_path=pending_file, client_order_id=next_order['client_order_id'], env_path=env_path)

        if not submitted:
            if time.monotonic() < deadline:
                time.sleep(poll_interval_seconds)
                continue

            return {
                'mode': 'paper_execution',
                'status': 'submission_deferred',
                'pending': pending
            }

        if time.monotonic() >= deadline:
            return {
                'mode': 'paper_execution',
                'status': 'order_submitted',
                'pending': pending
            }

        time.sleep(poll_interval_seconds)

#create one compact function which runs the entire process
def run_bot(tickers, submit_orders=False, env_path=None, state_path=None, lookback_days=63, top_frac=0.20, rebalance_days=21, execution_window_minutes=15, price_max_age_seconds=60, cash_reserve_fraction=0.01, max_wait_seconds=45, poll_interval_seconds=5):
    import math
    from pathlib import Path
    from broker import get_client, check_pending_orders, prepare_rebalance, get_positions

    if not isinstance(submit_orders, bool):
        raise TypeError("submit_orders must be True or False")

    if (not isinstance(execution_window_minutes, int) or isinstance(execution_window_minutes, bool) or execution_window_minutes < 1):
        raise ValueError("Execution window must be a positive integer")

    project_root = Path(__file__).resolve().parents[1]
    env_file = (Path(env_path).expanduser().resolve() if env_path is not None else project_root / '.env')
    state_file = (Path(state_path).expanduser().resolve() if state_path is not None else project_root / 'data' / 'basket_state.json')
    pending_file = state_file.parent / 'pending_rebalance.json'
    mode = 'paper_execution' if submit_orders else 'dry_run'

    with bot_run_lock():
        #get_client currently creates a paper-trading client
        client = get_client(str(env_file))

        #finish the outstanding plan before considering a new basket
        if pending_file.exists():
            if not submit_orders:
                with pending_file.open(encoding='utf-8') as file:
                    pending = json.load(file)

                if (pending.get('schema_version') != 2 or Path(pending['pending_path']).resolve() != pending_file):
                    raise ValueError("Invalid pending rebalance record")

                if str(client.get_account().id) != pending['account_id']:
                    raise RuntimeError("Pending rebalance belongs to another account")

                #read only: do not refresh statuses or change saved state
                return {
                    'mode': mode,
                    'status': 'pending_rebalance',
                    'pending': pending,
                    'message': (
                        "An unfinished rebalance exists. "
                        "Saved order statuses have not been refreshed."
                    )}

            return advance_pending_rebalance(client=client, pending_path=pending_file, env_path=env_file, max_wait_seconds=max_wait_seconds, poll_interval_seconds=poll_interval_seconds)

        #untracked open orders also prevent a new execution plan
        has_pending, open_orders = check_pending_orders(client)

        if has_pending:
            return {
                'mode': mode,
                'status': 'blocked_open_orders',
                'open_order_ids': [str(order.id) for order in open_orders],
            }

        preview = build_rebalance_preview(client=client, tickers=tickers, env_path=env_file, state_path=state_file, lookback_days=lookback_days, top_frac=top_frac, rebalance_days=rebalance_days, execution_window_minutes=execution_window_minutes, price_max_age_seconds=price_max_age_seconds, cash_reserve_fraction=cash_reserve_fraction)

        preview['mode'] = mode
        #between rebalances, leave quantities unchanged (shares originally)
        if not preview['plan']['rebalance_needed']:
            with state_file.open(encoding='utf-8') as file:
                completed_state = json.load(file)

            expected_positions = completed_state.get('positions_at_completion')
            if not isinstance(expected_positions, dict):
                raise ValueError("Completed basket state lacks its position snapshot")

            expected_positions = {ticker: float(quantity) for ticker, quantity in expected_positions.items()}
            if any(not math.isfinite(quantity) or quantity <= 0 for quantity in expected_positions.values()):
                raise ValueError("Completed basket state has invalid position quantities")

            account = client.get_account()
            if str(account.id) != completed_state.get('account_id'):
                raise RuntimeError("Completed basket state belongs to another account")

            actual_positions = get_positions(client)
            mismatches = [ticker for ticker in sorted(set(actual_positions).union(expected_positions)) if (ticker not in actual_positions or ticker not in expected_positions or not math.isclose(float(actual_positions[ticker]), expected_positions[ticker], rel_tol=0.0, abs_tol=1e-9))]

            preview['current_positions'] = actual_positions
            preview['account_id'] = str(account.id)
            preview['account_snapshot'] = {
                'cash': float(account.cash),
                'equity': float(account.equity if account.equity is not None else account.portfolio_value),
            }

            if mismatches:
                preview['status'] = 'holdings_mismatch'
                preview['blockers'] = [f"holdings differ for: {mismatches}"]

            return preview

        #a preview must never create a pending execution record
        if not submit_orders:
            return preview

        if not preview['execution_allowed']:
            return preview

        prepare_rebalance(plan=preview['plan'], orders=preview['orders'], account_id=preview['account_id'], execution_date=preview['execution_date'], execution_open=preview['execution_open'], execution_window_end=preview['execution_window_end'], current_positions=preview['current_positions'], pending_path=pending_file)

        return advance_pending_rebalance(client=client, pending_path=pending_file, env_path=env_file, max_wait_seconds=max_wait_seconds, poll_interval_seconds=poll_interval_seconds)
