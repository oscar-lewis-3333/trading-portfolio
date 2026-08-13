import numpy as np
import pandas as pd

def simple_labels(df, horizon=5, threshold=0.0): 
    #threshold defines our minimum market move in order to count as a signal. higher thresholds mean less signals but less false positives, and zeros moves that wouldn't make as much as trading costs. horizon defines how long we look forwards
    close = df['Close']

    fwd_return = close.shift(-horizon) / close - 1
    labels = (fwd_return > threshold).astype(int) #1 if fwd_returns are above our threshold, 0 if not. 
    labels[fwd_return.isna()] = np.nan

    return labels, fwd_return

def triple_barrier_labels(df, horizon=10, profit_mult=2.0, stop_mult=1.0, vol_window=21):
    #for each day, set profit target and stop loss scaled to current volatility, plus a time limit. Label according to which barrier hit first
    #basically encoding an actual trade with risk management, rather than an abstract forward return
    #profit/stop _mult barrier distances in terms of volatility

    #chose profit and loss target at 2:1 - can be right less than half of the time and still make profit

    close = df['Close']
    returns = close.pct_change()
    daily_vol = returns.rolling(vol_window).std()

    labels = pd.Series(np.nan, index=df.index) #initialising series
    holding_days = pd.Series(np.nan, index=df.index)
    realised_return = pd.Series(np.nan, index=df.index)

    prices = close.values
    vol = daily_vol.values
    n = len(prices)

    for i in range(n - horizon): #each day
        if np.isnan(vol[i]):
            continue

        entry = prices[i] #start price of the trade 
        upper = entry * (1 + profit_mult * vol[i]) #barriers scale with the current volatility at a 2:1 rate
        lower = entry * (1 - stop_mult * vol[i]) 

        outcome, days, ret = 0, horizon, 0.0

        for j in range(1, horizon+1):
            price = prices[i+j]
            if price >= upper:
                outcome, days, ret = 1, j, price/entry - 1 #price >= upper then trade succesful
                break
            if price <= lower:
                outcome, days, ret=0, j, price/entry - 1 #price <= lower boundary then trade unsuccessful
                break
        else:
            final = prices[i + horizon]
            ret = final/entry - 1
            outcome = int(ret > 0)

        labels.iloc[i] = outcome
        holding_days.iloc[i] = days
        realised_return.iloc[i] = ret

    return labels, holding_days, realised_return

# part 2: use formulation from book referenced in notebook.

def excess_return_target(pooled_df, horizon=21, epsilon=0.0):
    #continous regression target, each stocks forward return minus its equal-weighted universe average over the same period
    #we cancel out market-wide drift, isolating stock performance specifically. used for ranking rather than a label
    #epsilon: optional deadband. excess returns within +- epsilon from 0 can be treated as noise in a classification system. Kept small as large epsilon turns this into an outlier detection problem.

    import sys
    sys.path.append('../../past_market_analysis/src')
    from data_loader import fetch_price_data

    tickers = pooled_df['ticker'].unique()
    fwd_returns = {}

    for ticker in tickers:
        df = fetch_price_data(ticker, period="10y")
        close = df['Close']
        fwd_returns[ticker] = close.shift(-horizon) / close - 1

    fwd_df = pd.DataFrame(fwd_returns)
    universe_avg = fwd_df.mean(axis=1)
    excess = fwd_df.sub(universe_avg, axis=0)

    excess_long = excess.stack().rename('excess_return')
    excess_long.index.names = [pooled_df.index.name, 'ticker']
    excess_long = excess_long.reset_index()

    merged = pooled_df.reset_index().merge(excess_long, on=[pooled_df.index.name, 'ticker'], how='left').set_index(pooled_df.index.name)

    return merged

