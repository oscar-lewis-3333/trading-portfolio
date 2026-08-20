
import sys
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

from broker import get_client, decisions_to_orders, log_run, check_pending_orders, load_run_history

def generate_trading_decisions(tickers, top_frac=0.25, rebalance_days=63, soft_limit=-0.10, hard_limit=-0.25, target_risk=0.02, max_position=0.25, state_path='../data/basket_state.json', period="10y"):

    #effectively combining the recommendations from ml_trading_signals and risk_management into one pipeline

    from data_loader import fetch_price_data
    from position_sizing import volatility_scaled_size
    from risk_limits import compute_drawdown, drawdown_trading_exposure

    #1: get price data, momentum rank.
    price_data = {}
    for t in tickers:
        df = fetch_price_data(t, period=period)
        price_data[t] = df['Close']
    prices = pd.DataFrame(price_data).dropna(how='all')
    returns = prices.pct_change()

    #2: load previous portfolio, check if we need to rebalance the momentum or just check circuit-breaker:

    state = None
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    
    trading_days_since = None
    if state is not None:
        last_date = pd.Timestamp(state['selection_date'])
        trading_days_since = (prices.index > last_date).sum()

    rebalance_needed = (state is None or trading_days_since >= rebalance_days)

    if rebalance_needed:

        momentum = prices.pct_change(21).iloc[-1].dropna()
        if len(momentum) < 5:
            raise ValueError("Not enough tickers with valid momentum data")

        ranked = momentum.sort_values(ascending=False)
        n = max(1, int(len(ranked) * top_frac))
        tickers_chose = list(ranked.head(n).index)
        selected = ranked.head(n) #momentum' of the tickers chosen
 
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, 'w') as f:
            json.dump({
                'selection_date': prices.index[-1].strftime('%Y-%m-%d'),
                'tickers': tickers_chose
            }, f)

        print(f"Rebalanced. New tickers: {tickers_chose}")
    else:
        tickers_chose = state['tickers']
        selected = prices.pct_change(21).iloc[-1][tickers_chose]
        print(f"No rebalance ({trading_days_since} / {rebalance_days} since last rebalance)")



    #3: position sizing - volatility scaling per asset. initialise final decisions dataframe with relative weights from the sizing
    vol_63d = returns.rolling(63).std().iloc[-1] * np.sqrt(252)
    sizes = {}
    for ticker in selected.index:
        asset_vol = vol_63d[ticker]
        sizes[ticker] = volatility_scaled_size(target_risk, asset_vol, max_size=max_position)

    decisions = []
    for ticker in selected.index:
        decisions.append({
            'ticker': ticker,
            'momentum_21': selected[ticker],
            'base_size': sizes[ticker],
            'current_price': prices[ticker].iloc[-1],
        })
    decisions_df = pd.DataFrame(decisions)

    raw_total = decisions_df['base_size'].sum() #choose to scale weights so that the sum of weight is 1. else have a situation where only 15% of capital invested, when this is the whole system
    if raw_total > 0:
        decisions_df['relative_weight'] = decisions_df['base_size'] / raw_total
    else:
        decisions_df['relative_weight'] = 0
    
    #4: applying circuit breaker. use equal weight universe as baseline (both are equity curves like risk_management)
    equal_weight_returns = returns[selected.index].mean(axis=1)
    equity = (1 + equal_weight_returns.dropna()).cumprod()
    drawdown = compute_drawdown(equity)
    exposure_series = drawdown_trading_exposure(drawdown, soft_limit, hard_limit)
    current_exposure = exposure_series[-1] if len(exposure_series) > 0 else 1.0

    #5: 'append' decisions with final sizes
    decisions_df['circuit_breaker_exposure'] = current_exposure
    decisions_df['final_size'] = decisions_df['relative_weight'] * current_exposure

    return decisions_df, current_exposure, drawdown.iloc[-1]

#create a weekly summary of the changes the bot made in the most recent week

def weekly_summary(log_path):

    history = load_run_history(log_path=log_path)

    if len(history) < 1:
        return "No runs logged"

    latest = history.iloc[-1]
    latest_value = latest['account']['portfolio_value']

    if len(history) >=2:
        prev_value = history.iloc[-2]['account']['portfolio_value']
        change = ((latest_value / prev_value) -1) * 100 #(% change)
        change_str = f"{change:.2f}% since last run"
    else:
        change_str = "First run"

    tickers = [d['ticker'] for d in latest['decisions']]

    summary = (f"Portfolio value: ${latest_value:,.2f} ({change_str})\n"
        f"Exposure: {latest['circuit_breaker_exposure']:.2f}\n"
        f"Drawdown: {latest['current_drawdown']*100:.1f}%\n"
        f"Current basket: {', '.join(tickers)}"
    )
    return summary

#create one compact function which runs the entire process, logs everything etc.

def run_bot(tickers, submit_orders=False, log_path='../data/bot_log.jsonl', env_path='../.env', **decision_kwargs):
    #have submit order set to false by default as to avoid any accidental orders being placed when just testing the project

    from monitoring import send_alert

    print(f"--- System run: {pd.Timestamp.now()} ---\n")

    decisions, current_exposure, current_drawdown = generate_trading_decisions(tickers, **decision_kwargs)

    print(f"Circuit breaker exposure: {current_exposure:.4f}")
    print(f"Current drawdown: {current_drawdown:.4f}")
    print(decisions.to_string())

    client = get_client(env_path)
    account = client.get_account()
    account_snapshot = {
        'cash': float(account.cash),
        'portfolio_value': float(account.portfolio_value),
        'buying_power': float(account.buying_power),
    }
    print(f"\nAccount value: ${account_snapshot['portfolio_value']:,.2f}")

    has_pending, pending = check_pending_orders(client)
    if has_pending and submit_orders:
        print(f"\n⚠ {len(pending)} orders still pending from a previous run — "
                f"skipping submission to avoid duplicates.")
        for o in pending:
            print(f"{o.symbol}: {o.side} {o.qty} ({o.status})")
        submit_orders=False

    #order as done in broker, log to log_path
    orders = decisions_to_orders(decisions, client, dry_run= not submit_orders)

    log_run(decisions, orders, current_exposure, current_drawdown, account_snapshot, log_path)
    #send alert to me that everything works correctly
    summary = weekly_summary(log_path)

    send_alert("Trading bot — weekly run complete", summary, env_path=env_path)

    return decisions, orders




