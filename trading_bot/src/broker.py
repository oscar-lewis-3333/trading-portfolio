from dotenv import load_dotenv
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import json
from datetime import datetime
import pandas as pd

#we connect our decision maker directly to alpaca, a site for paper trading first to test quality

def get_client(env_path='../env'):
    #load alpaca and return our trading client
    load_dotenv(env_path)
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    return TradingClient(api_key, secret_key, paper=True)

def get_positions(client):
    #get all current positions in a dictionary {ticker: quantity}
    positions = client.get_all_positions()
    return {p.symbol: float(p.qty) for p in positions}

def decisions_to_orders(decisions_df, client, dry_run=True):
    #convert pipeline into actual paper orders

    #dry_run=True gives a print of what would be ordered, just a safety mechanism before setting False

    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    current_positions = get_positions(client)

    orders = []

    #close positions for tickers not in basket
    target_tickers = set(decisions_df['ticker'])
    for ticker, qty in current_positions.items():
        if ticker not in target_tickers and qty > 0:
            orders.append({
                'ticker': ticker,
                'side': 'sell',
                'qty': int(qty),
                'reason': 'no longer in target basket'
            })
    #buy/sell differences between current position, and target position recommended by system followed
    for _, row in decisions_df.iterrows():
        ticker = row['ticker']
        target_money = row['final_size'] * portfolio_value
        target_qty = int(target_money / row['current_price'])

        current_qty = int(current_positions.get(ticker, 0))
        qty_diff = target_qty - current_qty

        if qty_diff > 0:
            orders.append({
                'ticker': ticker,
                'side': 'buy',
                'qty': qty_diff,
                'reason': f'target {target_qty}, current {current_qty}'
            })
        elif qty_diff < 0:
            orders.append({
                'ticker': ticker,
                'side': 'sell',
                'qty': abs(qty_diff),
                'reason': f'target {target_qty}, current {current_qty}'
            })
    print(f"{'DRY RUN - ' if dry_run else ''}Orders set to be placed")
    for o in orders:
        print(f" {o['side'].upper():4s} {o['qty']:>5d} {o['ticker']:6s} ({o['reason']})")

    if not dry_run:
        for o in orders:
            request = MarketOrderRequest(
                symbol=o['ticker'],
                qty=o['qty'],
                side=OrderSide.BUY if o['side'] == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(request)
            print(f" SUBMITTED: {o['side']} {o['qty']} {o['ticker']}")

    return orders

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
        print(f" {p.symbol}: {p.qty} shares @ avg ${p.avg_entry_price}," f"current_value ${p.marekt_value}")

    account = client.get_account()
    print(f"\nCash: {account.cash}  Portfolio value: {account.portfolio_value}")

def check_pending_orders(client):
    #check for currently outstanding orders, (used to prevent duplicates)
    orders = client.get_orders()
    pending = [o for o in orders if o.status in ('new', 'accepted', 'pending_new')]
    return len(pending) > 0, pending

#get a function to report a 'run' of the bot to a jsonl

def log_run(decisions_df, orders, current_exposure, current_drawdown, account_snapshot, log_path='../data/bot_log.jsonl'):
    #jsonl chosen (one json object per line) over csv as nested structure -  a crash in one run cannot ruin a previous run
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    entry = {
        'timestamp': datetime.now().isoformat(),
        'circuit_breaker_exposure': float(current_exposure),
        'current_drawdown': float(current_drawdown),
        'account': account_snapshot,
        'decisions': decisions_df.to_dict('records'),
        'orders': orders,
    }

    with open(log_path, 'a') as f:
        f.write(json.dumps(entry, default=str) + '\n') #append file with entry on a new line

    print(f"Logged run to {log_path}")
    return entry

def load_run_history(log_path='../data/bot_log.jsonl'):
    #read the resulting file from above and turn into dataframe for analysis
    if not os.path.exists(log_path):
        print("No log file found")
        return pd.DataFrame()
    entries = []
    with open(log_path) as f:
        for line in f:
            entries.append(json.loads(line))

    return pd.DataFrame(entries)

