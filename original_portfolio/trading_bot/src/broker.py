import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
import json
import pandas as pd

#we connect our decision maker directly to alpaca, a site for paper trading first to test quality

def get_client(env_path=None):
    #load alpaca and return our trading client
    from pathlib import Path
    from dotenv import dotenv_values

    env_file = (Path(env_path).expanduser().resolve() if env_path is not None else Path(__file__).resolve().parents[1] / '.env')
    file_values = dotenv_values(env_file) if env_file.exists() else {}

    api_key = os.getenv('ALPACA_API_KEY') or file_values.get('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or file_values.get('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        raise RuntimeError("Alpaca credentials were not found")

    return TradingClient(api_key, secret_key, paper=True)

def get_data_client(env_path=None):
    from pathlib import Path
    from dotenv import dotenv_values
    from alpaca.data.historical import StockHistoricalDataClient

    env_file = (Path(env_path).expanduser().resolve() if env_path is not None else Path(__file__).resolve().parents[1] / '.env')
    file_values = dotenv_values(env_file) if env_file.exists() else {}

    api_key = os.getenv('ALPACA_API_KEY') or file_values.get('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY') or file_values.get('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        raise RuntimeError("Alpaca credentials were not found")

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

def get_positions(client):
    #get all current positions in a dictionary {ticker: quantity}
    import math

    positions = {}

    for position in client.get_all_positions():
        ticker = position.symbol
        side = getattr(position.side, 'value', position.side)
        quantity = float(position.qty)

        if (not isinstance(ticker, str) or not ticker or ticker in positions or side != 'long' or not math.isfinite(quantity) or quantity <= 0):
            raise RuntimeError(f"Unexpected broker position: {ticker}")

        positions[ticker] = quantity

    return positions

#make a function to quickly tell us our order status.

def check_order_status(client):
    print("--- Open/pending orders ---")
    order_list = client.get_orders()
    if not order_list:
        print(" None")
    for o in order_list:
        print(f" {o.symbol}: {o.side} {o.qty} - {o.status}") #(in laymens terms, ticker: buy/sell qty - status)

    print("--- Current positions")
    positions = client.get_all_positions()
    if not positions:
        print(" None")
    for p in positions: 
        print(f" {p.symbol}: {p.qty} shares @ avg ${p.avg_entry_price}," f"current_value ${p.market_value}")

    account = client.get_account()
    print(f"\nCash: {account.cash}  Portfolio value: {account.portfolio_value}")

def check_pending_orders(client):
    #check for currently outstanding orders, (used to prevent duplicates). any outstanding order blocks a rebalance
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
    pending = client.get_orders(filter=request)

    return bool(pending), pending

#save 2 files, one pending_rebalance, one for fixed basket of last rebalance
def save_state_atomic(state, state_path):
    from pathlib import Path
    import tempfile

    if not isinstance(state, dict):
        raise TypeError("State must be a dictionary")

    path = Path(state_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as file:
            temporary_path = Path(file.name)

            json.dump(state, file, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

def _pending_execution_window(pending):
    import math

    execution_date = pd.Timestamp(pending['execution_date'])
    execution_open = pd.Timestamp(pending['execution_open'])
    execution_window_end = pd.Timestamp(pending['execution_window_end'])

    if (pd.isna(execution_date) or execution_date.tzinfo is not None or execution_date != execution_date.normalize()):
        raise ValueError("Invalid pending execution date")

    if execution_open.tzinfo is None or execution_window_end.tzinfo is None:
        raise ValueError("Pending execution times must be timezone-aware")

    execution_open = execution_open.tz_convert('America/New_York')
    execution_window_end = execution_window_end.tz_convert('America/New_York')

    if (execution_open.date() != execution_date.date() or execution_window_end.date() != execution_date.date() or execution_window_end <= execution_open):
        raise ValueError("Invalid pending execution window")

    execution_policy = pending.get('execution_policy')
    if not isinstance(execution_policy, dict):
        raise ValueError("Invalid pending execution policy")

    required_policy = {'cash_reserve_fraction', 'execution_window_minutes', 'price_max_age_seconds', 'order_type', 'time_in_force', 'extended_hours'}
    if required_policy.difference(execution_policy):
        raise ValueError("Pending execution policy is incomplete")

    cash_reserve_fraction = execution_policy['cash_reserve_fraction']
    execution_window_minutes = execution_policy['execution_window_minutes']
    price_max_age_seconds = execution_policy['price_max_age_seconds']

    if isinstance(cash_reserve_fraction, bool):
        raise ValueError("Invalid pending cash reserve fraction")
    cash_reserve_fraction = float(cash_reserve_fraction)
    if not math.isfinite(cash_reserve_fraction) or not 0 <= cash_reserve_fraction < 1:
        raise ValueError("Invalid pending cash reserve fraction")

    if (not isinstance(execution_window_minutes, int) or isinstance(execution_window_minutes, bool) or execution_window_minutes < 1):
        raise ValueError("Invalid pending execution window length")

    if isinstance(price_max_age_seconds, bool):
        raise ValueError("Invalid pending maximum price age")
    price_max_age_seconds = float(price_max_age_seconds)
    if not math.isfinite(price_max_age_seconds) or price_max_age_seconds <= 0:
        raise ValueError("Invalid pending maximum price age")

    if (execution_policy['order_type'] != 'market' or execution_policy['time_in_force'] != 'day' or execution_policy['extended_hours'] is not False):
        raise ValueError("Unsupported pending execution policy")

    if execution_window_end > execution_open + pd.Timedelta(minutes=execution_window_minutes):
        raise ValueError("Pending execution window conflicts with its policy")

    return execution_open, execution_window_end

def _validate_pending_record(pending, pending_file):
    from pathlib import Path
    import math

    if not isinstance(pending, dict) or pending.get('schema_version') != 2:
        raise ValueError("Invalid pending rebalance record")

    pending_file = Path(pending_file).expanduser().resolve()
    required = {'account_id', 'pending_path', 'state_path', 'proposed_state', 'orders', 'starting_positions', 'expected_positions', 'previous_state'}
    if required.difference(pending):
        raise ValueError("Pending rebalance record is incomplete")

    if Path(pending['pending_path']).expanduser().resolve() != pending_file:
        raise ValueError("Pending state path does not match its saved location")
    if not isinstance(pending['account_id'], str) or not pending['account_id'].strip():
        raise ValueError("Pending account identifier is invalid")
    if not isinstance(pending['orders'], list):
        raise ValueError("Pending orders must be a list")
    if not isinstance(pending['starting_positions'], dict) or not isinstance(pending['expected_positions'], dict):
        raise ValueError("Pending positions must be dictionaries")
    if pending['previous_state'] is not None and not isinstance(pending['previous_state'], dict):
        raise ValueError("Previous basket state is invalid")

    proposed_state = pending['proposed_state']
    if not isinstance(proposed_state, dict) or not isinstance(proposed_state.get('strategy'), dict):
        raise ValueError("Proposed basket state is invalid")

    selected = proposed_state.get('tickers')
    universe = proposed_state['strategy'].get('universe')
    if (not isinstance(selected, list) or not selected or any(not isinstance(ticker, str) or not ticker for ticker in selected) or len(selected) != len(set(selected))):
        raise ValueError("Proposed basket tickers are invalid")
    if (not isinstance(universe, list) or any(not isinstance(ticker, str) or not ticker for ticker in universe) or len(universe) != len(set(universe)) or not set(selected).issubset(universe)):
        raise ValueError("Proposed strategy universe is invalid")

    for positions in (pending['starting_positions'], pending['expected_positions']):
        for ticker, quantity in positions.items():
            if (not isinstance(ticker, str) or not ticker or isinstance(quantity, bool)):
                raise ValueError("Pending position is invalid")
            quantity = float(quantity)
            if not math.isfinite(quantity) or quantity <= 0:
                raise ValueError("Pending position is invalid")

    if set(pending['starting_positions']).difference(universe):
        raise ValueError("Starting positions lie outside the strategy universe")
    if set(pending['expected_positions']) != set(selected):
        raise ValueError("Expected positions do not match the proposed basket")

    execution_open, execution_window_end = _pending_execution_window(pending)

    no_order_confirmed_at = pending.get('no_order_confirmed_at')
    if no_order_confirmed_at is not None:
        if pending['orders']:
            raise ValueError("Only an empty rebalance can have a no-order confirmation")

        confirmed_at = pd.Timestamp(no_order_confirmed_at)
        if confirmed_at.tzinfo is None:
            raise ValueError("No-order confirmation must be timezone-aware")

        confirmed_at = confirmed_at.tz_convert('America/New_York')
        if not execution_open <= confirmed_at < execution_window_end:
            raise ValueError("No-order confirmation lies outside its execution window")

    return pending

def _persist_pending_attention(pending, pending_file, reason, code):
    from datetime import datetime, timezone

    pending['status'] = 'needs_attention'
    pending['halt_code'] = code
    pending['halt_reason'] = reason
    pending['attention_reason'] = reason
    pending['updated_at'] = datetime.now(timezone.utc).isoformat()
    save_state_atomic(pending, pending_file)
    return pending

def mark_pending_attention(pending_path, reason, code='manual_attention'):
    from datetime import datetime, timezone
    from pathlib import Path

    pending_file = Path(pending_path).expanduser().resolve()

    with pending_file.open(encoding='utf-8') as file:
        pending = json.load(file)

    _validate_pending_record(pending, pending_file)

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Attention reason must be a non-empty string")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Attention code must be a non-empty string")

    return _persist_pending_attention(pending, pending_file, reason, code)
#record intended orders before we submit them
def prepare_rebalance(plan, orders, account_id, execution_date, execution_open, execution_window_end, current_positions, pending_path=None):
    from copy import deepcopy
    from pathlib import Path
    from datetime import datetime, timezone
    import math
    import uuid

    if not plan.get('rebalance_needed'):
        raise ValueError("No rebalance is due")
    proposed_state = plan.get('proposed_state')
    if not isinstance(proposed_state, dict):
        raise ValueError("The plan has no proposed basket state")
    if account_id is None or not str(account_id).strip():
        raise ValueError("An account identifier is required")
    if not isinstance(orders, list):
        raise TypeError("Orders must be a list")

    signal_date = pd.Timestamp(plan['signal_date'])
    execution_date = pd.Timestamp(execution_date)
    execution_open = pd.Timestamp(execution_open)
    execution_window_end = pd.Timestamp(execution_window_end)

    if (pd.isna(execution_date) or execution_date.tzinfo is not None or execution_date != execution_date.normalize() or execution_date <= signal_date):
        raise ValueError("Execution must be on a session after the signal")

    if execution_open.tzinfo is None or execution_window_end.tzinfo is None:
        raise ValueError("Execution times must be timezone-aware")

    execution_open = execution_open.tz_convert('America/New_York')
    execution_window_end = execution_window_end.tz_convert('America/New_York')

    if (execution_open.date() != execution_date.date() or execution_window_end.date() != execution_date.date() or execution_window_end <= execution_open):
        raise ValueError("Invalid execution window")

    execution_policy = plan.get('execution_policy')
    if not isinstance(execution_policy, dict):
        raise ValueError("The plan has no execution policy")

    required_policy = {'cash_reserve_fraction', 'execution_window_minutes', 'price_max_age_seconds', 'order_type', 'time_in_force', 'extended_hours'}
    if required_policy.difference(execution_policy):
        raise ValueError("The execution policy is incomplete")

    if (execution_policy['order_type'] != 'market' or execution_policy['time_in_force'] != 'day' or execution_policy['extended_hours'] is not False):
        raise ValueError("Unsupported execution policy")

    cash_reserve_fraction = float(execution_policy['cash_reserve_fraction'])
    execution_window_minutes = execution_policy['execution_window_minutes']
    price_max_age_seconds = float(execution_policy['price_max_age_seconds'])

    if (isinstance(execution_policy['cash_reserve_fraction'], bool) or not math.isfinite(cash_reserve_fraction) or not 0 <= cash_reserve_fraction < 1):
        raise ValueError("Invalid cash reserve fraction")
    if (not isinstance(execution_window_minutes, int) or isinstance(execution_window_minutes, bool) or execution_window_minutes < 1):
        raise ValueError("Invalid execution window length")
    if (not math.isfinite(price_max_age_seconds) or price_max_age_seconds <= 0):
        raise ValueError("Invalid maximum price age")
    if execution_window_end > execution_open + pd.Timedelta(minutes=execution_window_minutes):
        raise ValueError("Execution window conflicts with its policy")

    execution_policy = {
        'cash_reserve_fraction': cash_reserve_fraction,
        'execution_window_minutes': execution_window_minutes,
        'price_max_age_seconds': price_max_age_seconds,
        'order_type': 'market',
        'time_in_force': 'day',
        'extended_hours': False
    }

    basket_file = Path(plan['state_path']).expanduser().resolve()
    if pending_path is None:
        pending_file = basket_file.with_name('pending_rebalance.json')
    else:
        pending_file = Path(pending_path).expanduser().resolve()

    if pending_file == basket_file:
        raise ValueError("Pending and completed state need separate files")
    if pending_file.exists():
        raise RuntimeError("A pending rebalance already exists; reconcile it before creating another")

    orders = sorted(orders, key=lambda order: (order.get('side') != 'sell', order.get('ticker', '')))

    #identifiers are generated once and then reused from the saved record
    rebalance_id = uuid.uuid4().hex
    allowed_tickers = set(proposed_state['strategy']['universe'])
    seen_tickers = set()
    recorded_orders = []

    for order_i, order in enumerate(orders):
        ticker = order['ticker']
        side = order['side']

        if ticker not in allowed_tickers:
            raise ValueError(f"Order outside the configured universe: {ticker}")
        if ticker in seen_tickers:
            raise ValueError(f"Multiple orders proposed for {ticker}")
        if side not in ('buy', 'sell'):
            raise ValueError(f"Invalid order side for {ticker}")

        quantity = float(order['qty'])
        if (isinstance(order['qty'], bool) or not math.isfinite(quantity) or quantity <= 0):
            raise ValueError(f"Invalid order quantity for {ticker}")

        reference_price = float(order['reference_price'])
        if (isinstance(order['reference_price'], bool) or not math.isfinite(reference_price) or reference_price <= 0):
            raise ValueError(f"Invalid reference price for {ticker}")

        seen_tickers.add(ticker)

        recorded_orders.append({
            'ticker': ticker,
            'side': side,
            'qty': quantity,
            'reference_price': reference_price,
            'reason': str(order.get('reason', 'rebalance')),
            'client_order_id': f"mom-{rebalance_id}-{order_i:02d}",
            'broker_order_id': None,
            'status': 'not_submitted',
            'filled_qty': 0.0,
            'filled_avg_price': None
        })

    if not isinstance(current_positions, dict):
        raise TypeError("Current positions must be a dictionary")

    managed_tickers = set(plan['proposed_state']['strategy']['universe'])

    starting_positions = {}

    for ticker, quantity in current_positions.items():
        if isinstance(quantity, bool):
            raise ValueError(f"Invalid quantity for {ticker}")

        quantity = float(quantity)
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError(f"Invalid position quantity for {ticker}")
        if quantity == 0:
            continue

        if ticker not in managed_tickers:
            raise ValueError(f"Unmanaged position found: {ticker}")
        starting_positions[ticker] = quantity

    expected_positions = starting_positions.copy()
    quantity_tolerance = 1e-9  #allow tiny floating-point differences

    for order in recorded_orders:
        ticker = order['ticker']
        change = order['qty'] if order['side'] == 'buy' else -order['qty']

        final_quantity = expected_positions.get(ticker, 0.0) + change

        if (not math.isfinite(final_quantity) or final_quantity < -quantity_tolerance):
            raise ValueError(f"Invalid planned final holding: {ticker}")

        final_quantity = round(max(final_quantity, 0.0), 9)
        if final_quantity == 0:
            expected_positions.pop(ticker, None)
        else:
            expected_positions[ticker] = final_quantity

    outside_basket = set(expected_positions).difference(plan['proposed_state']['tickers'])
    if outside_basket:
        raise ValueError(f"Orders would leave holdings outside the new basket: "
            f"{sorted(outside_basket)}")

    missing_basket = set(plan['proposed_state']['tickers']).difference(expected_positions)
    if missing_basket:
        raise ValueError(f"Orders would leave selected tickers without holdings: {sorted(missing_basket)}")

    previous_state = None
    if basket_file.exists():
        with basket_file.open(encoding='utf-8') as file:
            previous_state = json.load(file)

        if not isinstance(previous_state, dict):
            raise ValueError("Existing basket state must be a dictionary")

    pending = {
        'schema_version': 2,
        'rebalance_id': rebalance_id,
        'account_id': str(account_id),
        'status': 'prepared',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'signal_date': signal_date.strftime('%Y-%m-%d'),
        'execution_date': execution_date.strftime('%Y-%m-%d'),
        'execution_open': execution_open.isoformat(),
        'execution_window_end': execution_window_end.isoformat(),
        'execution_policy': deepcopy(execution_policy),
        'state_path': str(basket_file),
        'pending_path': str(pending_file),
        'proposed_state': deepcopy(proposed_state),
        'orders': recorded_orders,
        'starting_positions': starting_positions,
        'expected_positions': expected_positions,
        'previous_state': previous_state
    }

    _validate_pending_record(pending, pending_file)
    save_state_atomic(pending, pending_file)
    return pending

#construct orders, does not submit
def decisions_to_orders(decisions_df, current_positions, available_cash, bid_prices, ask_prices, managed_tickers, cash_reserve_fraction=0.01):
    import math
    if decisions_df is None:
        return []
    if not isinstance(decisions_df, pd.DataFrame) or decisions_df.empty:
        raise ValueError("Expected a non-empty decisions DataFrame")

    required = {'ticker', 'target_weight'}
    missing = required.difference(decisions_df.columns)
    if missing:
        raise ValueError(f"Missing decision columns: {sorted(missing)}")
    if isinstance(managed_tickers, str):
        raise TypeError("Managed tickers must be a collection")

    managed_tickers = list(managed_tickers)
    if (not managed_tickers or any(not isinstance(t, str) or not t for t in managed_tickers)):
        raise ValueError("Managed tickers must be non-empty strings")
    if not isinstance(current_positions, dict):
        raise TypeError("Current positions must be a dictionary")
    if not isinstance(bid_prices, dict) or not isinstance(ask_prices, dict):
        raise TypeError("Bid and ask prices must be dictionaries")
    if isinstance(available_cash, bool):
        raise ValueError("Available cash must be numeric")

    available_cash = float(available_cash)
    if not math.isfinite(available_cash) or available_cash < 0:
        raise ValueError("Available cash must be finite and non-negative")
    if isinstance(cash_reserve_fraction, bool):
        raise ValueError("Cash reserve fraction must be numeric")

    cash_reserve_fraction = float(cash_reserve_fraction)
    if (not math.isfinite(cash_reserve_fraction) or not 0 <= cash_reserve_fraction < 1):
        raise ValueError("Cash reserve fraction must be between 0 and 1")

    allowed = set(managed_tickers)
    decisions = decisions_df.copy()

    if (decisions['ticker'].isna().any() or decisions['ticker'].duplicated().any() or not decisions['ticker'].isin(allowed).all()):
        raise ValueError("Decision tickers must be unique and within the universe")

    decisions['target_weight'] = pd.to_numeric(decisions['target_weight'], errors='raise').astype(float)
    weights = decisions.set_index('ticker')['target_weight'].to_dict()

    if not all(math.isfinite(weight) and 0 <= weight <= 1 for weight in weights.values()):
        raise ValueError("Target weights must be finite and between 0 and 1")

    if sum(weights.values()) > 1.0 + 1e-10:
        raise ValueError("Total target weight cannot exceed 100%")

    positions = {}

    for ticker, quantity in current_positions.items():
        if isinstance(quantity, bool):
            raise ValueError(f"Invalid position for {ticker}")

        quantity = float(quantity)

        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError(f"Invalid or short position found: {ticker}")

        if quantity > 0:
            positions[ticker] = quantity

    unexpected = set(positions).difference(allowed)
    if unexpected:
        raise ValueError(f"Account contains unmanaged holdings: {sorted(unexpected)}")

    symbols = set(positions).union(ticker for ticker, weight in weights.items() if weight > 0)

    missing_bids = symbols.difference(bid_prices)
    missing_asks = symbols.difference(ask_prices)

    if missing_bids or missing_asks:
        raise ValueError(f"Missing bid prices: {sorted(missing_bids)}; " f"missing ask prices: {sorted(missing_asks)}")

    bids = {ticker: float(bid_prices[ticker]) for ticker in symbols}
    asks = {ticker: float(ask_prices[ticker]) for ticker in symbols}

    for ticker in symbols:
        if (not math.isfinite(bids[ticker]) or not math.isfinite(asks[ticker]) or bids[ticker] <= 0 or asks[ticker] <= 0 or asks[ticker] < bids[ticker]):
            raise ValueError(f"Invalid bid/ask prices for {ticker}")

    liquidation_value = available_cash + sum(positions[ticker] * bids[ticker] for ticker in positions)

    if not math.isfinite(liquidation_value) or liquidation_value <= 0:
        raise ValueError("Conservative portfolio value must be positive")

    investable_value = liquidation_value * (1.0 - cash_reserve_fraction)

    orders = []

    for ticker in sorted(symbols):
        target_value = investable_value * weights.get(ticker, 0.0)
        target_qty = int(target_value / asks[ticker])
        current_qty = positions.get(ticker, 0.0)
        quantity_difference = round(target_qty - current_qty, 9)

        if quantity_difference == 0:
            continue

        is_buy = quantity_difference > 0

        orders.append({
            'ticker': ticker,
            'side': 'buy' if is_buy else 'sell',
            'qty': abs(quantity_difference),
            'reference_price': (asks[ticker] if is_buy else bids[ticker]),
            'reason': (f"target {target_qty}, current {current_qty:g}"),
        })

    estimated_sale_proceeds = sum(order['qty'] * bids[order['ticker']] for order in orders if order['side'] == 'sell')

    estimated_buy_cost = sum(order['qty'] * asks[order['ticker']] for order in orders if order['side'] == 'buy')

    estimated_funds = available_cash + estimated_sale_proceeds
    if estimated_buy_cost > estimated_funds + 1e-9:
        raise RuntimeError("Proposed orders are not fundable at current bid/ask prices")

    orders.sort(key=lambda order: (order['side'] != 'sell', order['ticker']))

    return orders

def reconcile_pending_orders(client, pending_path):
    from pathlib import Path
    from datetime import datetime, timezone
    from alpaca.common.exceptions import APIError
    import math

    pending_file = Path(pending_path).expanduser().resolve()

    with pending_file.open() as file:
        pending = json.load(file)

    _validate_pending_record(pending, pending_file)

    account = client.get_account()
    if str(account.id) != pending['account_id']:
        raise ValueError("Pending rebalance belongs to a different account")

    active_statuses = {
        'accepted',
        'pending_new',
        'new',
        'partially_filled',
        'accepted_for_bidding'
    }

    for order in pending['orders']:
        try:
            broker_order = client.get_order_by_client_id(order['client_order_id'])
        except APIError as error:
            if error.status_code != 404:
                raise

            #only expect to not find an order if one has not been submitted
            never_submitted = order['status'] == 'not_submitted' and order['broker_order_id'] is None
            if not never_submitted:
                order['status'] = 'unknown'
            continue

        if order.get('status') == 'not_submitted':
            reason = "Broker order exists without a recorded submission attempt"
            _persist_pending_attention(
                pending, pending_file, reason, 'unrecorded_submission'
            )
            raise RuntimeError(reason)

        broker_side = getattr(broker_order.side, 'value', broker_order.side)
        broker_status = str(getattr(broker_order.status, 'value', broker_order.status))
        broker_order_type = getattr(broker_order, 'order_type', getattr(broker_order, 'type', None))
        broker_order_type = getattr(broker_order_type, 'value', broker_order_type)
        broker_time_in_force = getattr(broker_order.time_in_force, 'value', broker_order.time_in_force)
        broker_extended_hours = broker_order.extended_hours

        #ensure order we have matches the order that we intended to make
        if (broker_order.client_order_id != order['client_order_id'] or broker_order.symbol != order['ticker'] or broker_side != order['side'] or broker_order.qty is None or not math.isclose(float(broker_order.qty), order['qty'], rel_tol=0, abs_tol=1e-9) or broker_order_type != 'market' or broker_time_in_force != 'day' or broker_extended_hours is not False):
            reason = "Broker order differs from the saved plan"
            _persist_pending_attention(pending, pending_file, reason, 'broker_order_mismatch')
            raise RuntimeError(reason)

        broker_id = str(broker_order.id)
        if order['broker_order_id'] not in (None, broker_id):
            reason = "Broker order identifier unexpectedly changed"
            _persist_pending_attention(pending, pending_file, reason, 'broker_order_id_changed')
            raise RuntimeError(reason)

        filled_qty = float(broker_order.filled_qty or 0.0)
        average_price = (float(broker_order.filled_avg_price) if broker_order.filled_avg_price is not None else None)

        if (not math.isfinite(filled_qty) or filled_qty < 0 or filled_qty > order['qty'] + 1e-9):
            reason = "Broker returned an invalid filled quantity"
            _persist_pending_attention(pending, pending_file, reason, 'invalid_filled_quantity')
            raise RuntimeError(reason)
        if filled_qty > 0 and (average_price is None or not math.isfinite(average_price) or average_price <= 0):
            reason = "Filled order has no valid average fill price"
            _persist_pending_attention(pending, pending_file, reason, 'invalid_fill_price')
            raise RuntimeError(reason)

        if broker_status == 'partially_filled' and not (0 < filled_qty < order['qty']):
            reason = "Partially filled order has an inconsistent filled quantity"
            _persist_pending_attention(pending, pending_file, reason, 'inconsistent_partial_fill')
            raise RuntimeError(reason)

        if broker_status in active_statuses.difference({'partially_filled'}) and filled_qty != 0:
            reason = "Active order has an inconsistent filled quantity"
            _persist_pending_attention(pending, pending_file, reason, 'inconsistent_active_fill')
            raise RuntimeError(reason)

        order.update({
            'broker_order_id': broker_id,
            'status': broker_status,
            'filled_qty': filled_qty,
            'filled_avg_price': average_price,
        })

    #summarise progress without declaring the basket successfully rebalanced
    #only these broker statuses may be waited on automatically
    valid_statuses = {'not_submitted', 'filled'}.union(active_statuses)
    order_statuses = [order.get('status') for order in pending['orders']]
    invalid_statuses = sorted({repr(status) for status in order_statuses if (not isinstance(status, str) or status not in valid_statuses)})

    incorrect_fills = [order['ticker'] for order in pending['orders']
        if order.get('status') == 'filled' and not math.isclose(float(order['filled_qty']), float(order['qty']), rel_tol=0.0, abs_tol=1e-9)]

    if pending.get('halt_code') == 'submission_outcome_uncertain':
        unresolved = any(status in {'submitting', 'unknown'} for status in order_statuses)
        if not unresolved:
            pending.pop('halt_code', None)
            pending.pop('halt_reason', None)

    halt_reason = pending.get('halt_reason')
    pending.pop('attention_reason', None)

    if halt_reason:
        pending['status'] = 'needs_attention'
        pending['attention_reason'] = halt_reason

    elif invalid_statuses:
        pending['status'] = 'needs_attention'
        pending['attention_reason'] = f"Unexpected order statuses: {invalid_statuses}"

    elif incorrect_fills:
        pending['status'] = 'needs_attention'
        pending['attention_reason'] = (f"Orders marked filled with incorrect quantities: "
            f"{sorted(incorrect_fills)}")

    elif not pending['orders']:
        pending['status'] = 'no_orders'

    elif all(status == 'filled' for status in order_statuses):
        pending['status'] = 'orders_filled'

    elif all(status == 'not_submitted' for status in order_statuses):
        pending['status'] = 'prepared'

    else:
        pending['status'] = 'in_progress'

    pending['updated_at'] = datetime.now(timezone.utc).isoformat()

    save_state_atomic(pending, pending_file)
    return pending

def submit_pending_order(client, pending_path, client_order_id, env_path=None):
    #called by run_bot under its run lock, after price/funding checks
    #returns (updated_pending_state, submission_made)
    from datetime import datetime, timezone
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockLatestQuoteRequest
    import math

    pending = reconcile_pending_orders(client, pending_path)

    if pending['status'] == 'needs_attention':
        raise RuntimeError("The pending rebalance needs attention before further submission")

    matches = [order for order in pending['orders'] if order['client_order_id'] == client_order_id]
    if len(matches) != 1:
        raise ValueError("Order identifier must match exactly one saved order")

    order = matches[0]

    #if order already in process, then do not resubmit it.
    if order['status'] != 'not_submitted':
        return pending, False

    #only submit during the intended regular trading session
    clock = client.get_clock()
    broker_time = pd.Timestamp(clock.timestamp)
    if broker_time.tzinfo is None:
        raise RuntimeError("Broker clock must include its timezone")

    broker_time = broker_time.tz_convert('America/New_York')
    execution_open, execution_window_end = _pending_execution_window(pending)

    if broker_time >= execution_window_end:
        reason = "The intended execution window has passed; review the plan"
        mark_pending_attention(pending_path, reason, code='missed_execution_window')
        raise RuntimeError(reason)
    if broker_time < execution_open or not clock.is_open:
        return pending, False

    #purchases wait until every planned sale has actually filled
    if order['side'] == 'buy':
        unfinished_sales = any(other['side'] == 'sell' and other['status'] != 'filled' for other in pending['orders'])
        if unfinished_sales:
            return pending, False

    #also block if the account has any other outstanding orders
    has_pending, _ = check_pending_orders(client)
    if has_pending:
        return pending, False

    quantity = float(order['qty'])
    allowed_tickers = pending['proposed_state']['strategy']['universe']

    if (order['ticker'] not in allowed_tickers or order['side'] not in ('buy', 'sell') or not math.isfinite(quantity) or quantity <= 0):
        reason = "Saved order contains invalid trading instructions"
        _persist_pending_attention(pending, pending_path, reason, 'invalid_saved_order')
        raise ValueError(reason)

    account = client.get_account()
    account_status = getattr(account.status, 'value', account.status)

    if (account_status != 'ACTIVE' or bool(getattr(account, 'account_blocked', False)) or bool(getattr(account, 'trading_blocked', False)) or bool(getattr(account, 'trade_suspended_by_user', False))):
        reason = "The account cannot currently trade"
        _persist_pending_attention(pending, pending_path, reason, 'account_cannot_trade')
        raise RuntimeError(reason)

    asset = client.get_asset(order['ticker'])
    if not bool(asset.tradable):
        reason = f"{order['ticker']} is not tradable"
        _persist_pending_attention(pending, pending_path, reason, 'asset_not_tradable')
        raise RuntimeError(reason)

    fractional = not math.isclose(quantity, round(quantity), rel_tol=0.0, abs_tol=1e-9)
    if fractional and not bool(asset.fractionable):
        reason = f"{order['ticker']} does not permit fractional orders"
        _persist_pending_attention(pending, pending_path, reason, 'fractional_order_not_supported')
        raise RuntimeError(reason)

    positions = {}
    available_positions = {}

    for position in client.get_all_positions():
        ticker = position.symbol
        side = getattr(position.side, 'value', position.side)
        position_quantity = float(position.qty)
        available_quantity = float(position.qty_available) if position.qty_available is not None else position_quantity

        if (ticker in positions or side != 'long' or not math.isfinite(position_quantity) or position_quantity <= 0 or not math.isfinite(available_quantity) or available_quantity < 0 or available_quantity > position_quantity + 1e-9):
            reason = f"Unexpected broker position: {ticker}"
            _persist_pending_attention(pending, pending_path, reason, 'invalid_broker_position')
            raise RuntimeError(reason)

        positions[ticker] = position_quantity
        available_positions[ticker] = available_quantity

    expected_positions = {ticker: float(position_quantity) for ticker, position_quantity in pending['starting_positions'].items()}

    for saved_order in pending['orders']:
        if saved_order['status'] != 'filled':
            continue

        change = float(saved_order['filled_qty']) if saved_order['side'] == 'buy' else -float(saved_order['filled_qty'])
        final_quantity = round(expected_positions.get(saved_order['ticker'], 0.0) + change, 9)

        if final_quantity < 0:
            reason = f"Filled orders produced a negative holding for {saved_order['ticker']}"
            _persist_pending_attention(pending, pending_path, reason, 'negative_expected_holding')
            raise RuntimeError(reason)
        if final_quantity == 0:
            expected_positions.pop(saved_order['ticker'], None)
        else:
            expected_positions[saved_order['ticker']] = final_quantity

    position_mismatches = [ticker for ticker in sorted(set(positions).union(expected_positions)) if (ticker not in positions or ticker not in expected_positions or not math.isclose(positions[ticker], expected_positions[ticker], rel_tol=0.0, abs_tol=1e-9))]

    if position_mismatches:
        return pending, False

    if order['side'] == 'sell' and available_positions.get(order['ticker'], 0.0) + 1e-9 < quantity:
        reason = f"Insufficient available shares of {order['ticker']}"
        _persist_pending_attention(pending, pending_path, reason, 'insufficient_available_shares')
        raise RuntimeError(reason)

    remaining_buys = [saved_order for saved_order in pending['orders'] if saved_order['side'] == 'buy' and saved_order['status'] == 'not_submitted']
    quote_tickers = ([order['ticker']] if order['side'] == 'sell' else sorted({saved_order['ticker'] for saved_order in remaining_buys}))
    data_client = get_data_client(env_path)
    quotes = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=quote_tickers, feed=DataFeed.IEX))

    if set(quote_tickers).difference(quotes):
        raise RuntimeError("Quotes are missing for pending submission")

    clock = client.get_clock()
    broker_time = pd.Timestamp(clock.timestamp)
    if broker_time.tzinfo is None:
        raise RuntimeError("Broker clock must include its timezone")

    broker_time = broker_time.tz_convert('America/New_York')

    if broker_time >= execution_window_end:
        reason = "The intended execution window has passed; review the plan"
        mark_pending_attention(pending_path, reason, code='missed_execution_window')
        raise RuntimeError(reason)
    if broker_time < execution_open or not clock.is_open:
        return pending, False

    bid_prices = {}
    ask_prices = {}
    quote_times = {}
    maximum_age = float(pending['execution_policy']['price_max_age_seconds'])

    for ticker in quote_tickers:
        quote = quotes[ticker]
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        quote_time = pd.Timestamp(quote.timestamp)

        if (not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0 or ask < bid or quote_time.tzinfo is None):
            raise ValueError(f"Invalid quote for {ticker}")

        quote_time = quote_time.tz_convert('America/New_York')
        age = (broker_time - quote_time).total_seconds()

        if not (execution_open <= quote_time <= broker_time and 0 <= age <= maximum_age):
            return pending, False

        bid_prices[ticker] = bid
        ask_prices[ticker] = ask
        quote_times[ticker] = quote_time

    if order['side'] == 'buy':
        estimated_cost = sum(float(saved_order['qty']) * ask_prices[saved_order['ticker']] for saved_order in remaining_buys)
        available_cash = float(account.cash)

        if not math.isfinite(available_cash) or available_cash < 0:
            reason = "Invalid available cash"
            _persist_pending_attention(pending, pending_path, reason, 'invalid_available_cash')
            raise RuntimeError(reason)
        if estimated_cost > available_cash:
            return pending, False

    submission_clock = client.get_clock()
    submission_time = pd.Timestamp(submission_clock.timestamp)
    if submission_time.tzinfo is None:
        raise RuntimeError("Broker clock must include its timezone")

    submission_time = submission_time.tz_convert('America/New_York')
    if submission_time >= execution_window_end:
        reason = "The intended execution window passed before submission"
        mark_pending_attention(pending_path, reason, code='missed_execution_window')
        raise RuntimeError(reason)
    if submission_time < execution_open or not submission_clock.is_open:
        return pending, False
    if any(
        not (quote_time <= submission_time and
             (submission_time - quote_time).total_seconds() <= maximum_age)
        for quote_time in quote_times.values()
    ):
        return pending, False

    order['submission_reference_price'] = (
        ask_prices[order['ticker']]
        if order['side'] == 'buy'
        else bid_prices[order['ticker']]
    )
    order['submission_quote_at'] = quote_times[order['ticker']].isoformat()

    request = MarketOrderRequest(symbol=order['ticker'], qty=quantity, side=(OrderSide.BUY if order['side'] == 'buy' else OrderSide.SELL), time_in_force=TimeInForce.DAY, extended_hours=False, client_order_id=order['client_order_id'])

    #persist the attempt before making the network request
    attempted_at = submission_time.tz_convert('UTC').isoformat()
    order['status'] = 'submitting'
    order['submission_attempted_at'] = attempted_at
    pending['status'] = 'in_progress'
    pending['updated_at'] = attempted_at

    save_state_atomic(pending, pending_path)

    try:
        response = client.submit_order(order_data=request)
        if response.client_order_id != order['client_order_id']:
            raise RuntimeError("Submission response has an unexpected identifier")

    except Exception as error:
        #the request might have reached the broker despite the exception
        order['status'] = 'unknown'
        pending['status'] = 'needs_attention'
        pending['halt_code'] = 'submission_outcome_uncertain'
        pending['halt_reason'] = "Submission outcome is uncertain; reconcile the saved order identifier"
        pending['attention_reason'] = pending['halt_reason']
        pending['updated_at'] = datetime.now(timezone.utc).isoformat()

        save_state_atomic(pending, pending_path)
        raise RuntimeError("Submission outcome is uncertain. Reconcile the saved order identifier before taking further action.") from error

    order['broker_order_id'] = str(response.id)
    order['status'] = str(getattr(response.status, 'value', response.status))
    pending['updated_at'] = datetime.now(timezone.utc).isoformat()
    save_state_atomic(pending, pending_path)

    pending = reconcile_pending_orders(client, pending_path)

    return pending, True

def complete_rebalance(client, pending_path):
    #call under bot_run_lock, never during a dry run
    #returns (execution_record, completed)
    from copy import deepcopy
    from datetime import datetime, timezone
    from pathlib import Path
    from uuid import UUID
    import math

    pending_file = Path(pending_path).expanduser().resolve()

    with pending_file.open(encoding='utf-8') as file:
        pending = json.load(file)

    _validate_pending_record(pending, pending_file)

    account = client.get_account()
    if str(account.id) != pending['account_id']:
        raise RuntimeError("Pending rebalance belongs to another account")

    rebalance_id = UUID(pending['rebalance_id']).hex
    if rebalance_id != pending['rebalance_id']:
        raise ValueError("Invalid rebalance ID")

    state_file = Path(pending['state_path']).resolve()
    if state_file == pending_file:
        raise ValueError("Basket and pending paths must differ")
    def read_basket_state():
        if not state_file.exists():
            return None
        with state_file.open(encoding='utf-8') as file:
            return json.load(file)

    current_state = read_basket_state()
    completed_state = pending.get('completed_state')

    already_committed = (completed_state is not None and current_state == completed_state and completed_state.get('rebalance_id') == rebalance_id)

    if not already_committed:
        if current_state != pending['previous_state']:
            reason = "Basket state changed since preparation; review required"
            _persist_pending_attention(pending, pending_file, reason, 'basket_state_changed')
            raise RuntimeError(reason)

        pending = reconcile_pending_orders(client, pending_file)

        if pending['status'] == 'needs_attention':
            raise RuntimeError("Order reconciliation requires attention")

        if pending['status'] not in {'orders_filled', 'no_orders'}:
            return pending, False

        #no-order rebalance still needs its own execution session as planned
        if (pending['status'] == 'no_orders' and
                pending.get('no_order_confirmed_at') is None):
            clock = client.get_clock()
            timestamp = pd.Timestamp(clock.timestamp)

            if timestamp.tzinfo is None:
                raise ValueError("Broker timestamp must be timezone-aware")

            timestamp = timestamp.tz_convert('America/New_York')
            execution_open, execution_window_end = _pending_execution_window(pending)

            if timestamp >= execution_window_end:
                reason = "Missed the planned execution window"
                mark_pending_attention(pending_file, reason, code='missed_execution_window')
                raise RuntimeError(reason)

            if timestamp < execution_open or not clock.is_open:
                return pending, False

            pending['no_order_confirmed_at'] = timestamp.isoformat()
            pending['updated_at'] = datetime.now(timezone.utc).isoformat()
            save_state_atomic(pending, pending_file)

        has_pending, _ = check_pending_orders(client)
        if has_pending:
            return pending, False

        expected = {ticker: float(quantity) for ticker, quantity in pending['expected_positions'].items()}

        if any(not math.isfinite(quantity) or quantity <= 0 for quantity in expected.values()):
            raise ValueError("Invalid expected holdings")
        if not set(expected).issubset(pending['proposed_state']['tickers']):
            raise ValueError("Expected holdings lie outside the new basket")

        actual = {}

        for position in client.get_all_positions():
            ticker = position.symbol
            quantity = float(position.qty)
            side = getattr(position.side, 'value', position.side)

            if ticker in actual:
                reason = f"Duplicate broker position: {ticker}"
                _persist_pending_attention(pending, pending_file, reason, 'duplicate_broker_position')
                raise RuntimeError(reason)

            if (side != 'long' or not math.isfinite(quantity) or quantity <= 0):
                reason = f"Unexpected broker position: {ticker}"
                _persist_pending_attention(pending, pending_file, reason, 'invalid_broker_position')
                raise RuntimeError(reason)
            actual[ticker] = quantity

        mismatches = [ticker for ticker in sorted(set(actual).union(expected))
            if (ticker not in actual or ticker not in expected or not math.isclose(actual[ticker], expected[ticker], rel_tol=0.0, abs_tol=1e-9))]

        if mismatches:
            pending['position_mismatches'] = mismatches
            pending['updated_at'] = datetime.now(timezone.utc).isoformat()
            save_state_atomic(pending, pending_file)
            return pending, False

        pending.pop('position_mismatches', None)

        #save the exact intended state before committing it
        #on a retry, reuse it rather than changing its completion time
        if completed_state is None:
            account = client.get_account()
            if str(account.id) != pending['account_id']:
                reason = "Pending rebalance belongs to another account"
                _persist_pending_attention(pending, pending_file, reason, 'account_changed')
                raise RuntimeError(reason)

            def account_number(value, positive=False):
                if isinstance(value, bool):
                    raise ValueError
                number = float(value)
                if not math.isfinite(number) or (positive and number <= 0):
                    raise ValueError
                return number

            try:
                raw_equity = getattr(account, 'equity', None)
                raw_portfolio_value = getattr(account, 'portfolio_value', None)
                cash = account_number(account.cash)
                equity = account_number(
                    raw_equity if raw_equity is not None else raw_portfolio_value,
                    positive=True,
                )
                portfolio_value = account_number(
                    raw_portfolio_value if raw_portfolio_value is not None else equity,
                    positive=True,
                )
            except (AttributeError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "Invalid account snapshot at completion; retry completion"
                ) from error

            completed_state = deepcopy(pending['proposed_state'])
            completed_state.update({
                'rebalance_id': rebalance_id,
                'account_id': pending['account_id'],
                'planned_execution_date': pending['execution_date'],
                'execution_policy': deepcopy(pending['execution_policy']),
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'positions_at_completion': actual,
                'account_snapshot': {
                    'cash': cash,
                    'equity': equity,
                    'portfolio_value': portfolio_value,
                }
            })

        pending['completed_state'] = completed_state
        pending['status'] = 'ready_to_commit'
        save_state_atomic(pending, pending_file)

        if read_basket_state() != pending['previous_state']:
            reason = "Basket state changed before completion"
            _persist_pending_attention(pending, pending_file, reason, 'basket_state_changed')
            raise RuntimeError(reason)

        save_state_atomic(completed_state, state_file)

    #if interrupted after committing, a retry finishes 
    pending['status'] = 'completed'
    pending['completed_at'] = completed_state['completed_at']
    save_state_atomic(pending, pending_file)

    archive_dir = pending_file.parent / 'rebalance_history'
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_file = archive_dir / f'{rebalance_id}.json'

    if archive_file.exists():
        with archive_file.open(encoding='utf-8') as file:
            archived = json.load(file)

        archived_comparable = dict(archived)
        pending_comparable = dict(pending)
        archived_comparable.pop('updated_at', None)
        pending_comparable.pop('updated_at', None)

        if archived_comparable != pending_comparable:
            raise RuntimeError("A conflicting execution archive exists")
    else:
        #preserve the record without overwriting an existing archive
        os.link(pending_file, archive_file)

    pending_file.unlink()
    return pending, True
