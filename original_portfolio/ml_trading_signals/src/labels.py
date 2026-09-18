import numpy as np
import pandas as pd

def simple_labels(df, horizon=5, threshold=0.0): 
    #threshold defines our minimum market move in order to count as a signal. higher thresholds mean less signals but less false positives, and zeros moves that wouldn't make as much as trading costs. horizon defines how long we look forwards
    columns = ['Open', 'Close']
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Required column is missing: {missing}")

    entry = df['Open'].shift(-1)
    exit = df['Close'].shift(-horizon)

    fwd_return = exit / entry - 1
    labels = (fwd_return > threshold).astype(float) #1 if fwd_returns are above our threshold, 0 if not. 
    labels[fwd_return.isna()] = np.nan

    return labels, fwd_return

def triple_barrier_labels(df, horizon=10, profit_mult=2.0, stop_mult=1.0, vol_window=21):
    #for each day, set profit target and stop loss scaled to current volatility, plus a time limit. Label according to which barrier hit first
    #basically encoding an actual trade with risk management, rather than an abstract forward return
    #profit/stop _mult barrier distances in terms of volatility

    #chose profit and loss target at 2:1 - can be right less than half of the time and still make profit

    columns = ['Open', 'Close', 'Low', 'High']
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Required column is missing: {missing}")

    close = df['Close'].astype(float)
    returns = close.pct_change()
    daily_vol = returns.rolling(vol_window).std()

    labels = pd.Series(np.nan, index=df.index) #initialising series
    holding_days = pd.Series(np.nan, index=df.index)
    realised_return = pd.Series(np.nan, index=df.index)

    open = df['Open'].to_numpy(dtype=float)
    high = df['High'].to_numpy(dtype=float)
    low = df['Low'].to_numpy(dtype=float)
    closes = df['Close'].to_numpy(dtype=float)
    daily_vols = daily_vol.to_numpy(dtype=float)

    n = len(df)

    for i in range(n - horizon): #each day
        entry_i = i + 1
        final_i = i + horizon

        entry_price = open[entry_i]
        entry_vol = daily_vols[i]

        if not np.isfinite(entry_price) or not np.isfinite(entry_vol):
            continue
        if entry_price <=0 or entry_vol <= 0:
            continue

 
        upper = entry_price * (1 + profit_mult * entry_vol) #barriers scale with the current volatility at a 2:1 rate
        lower = entry_price * (1 - stop_mult * entry_vol) 

        outcome = None
        exit_price = None
        days = None

        for day, j in enumerate(range(entry_i, final_i+1), start=1):
            day_open = open[j]
            day_high = high[j]
            day_low = low[j]

            if not all(np.isfinite([day_open, day_high, day_low])):
                continue
            if day_open <= lower:
                outcome = 0
                exit_price = day_open
                days = day
                break
            if day_open >= upper:
                outcome = 1
                exit_price = day_open
                days = day
                break

            upper_hit = day_high >= upper
            lower_hit = day_low <= lower

            if upper_hit and lower_hit: #if both boundaries hit, daily data cannot determine which one was reached first, so conservatively sell
                outcome = 0
                exit_price = lower
                days = day
                break

            if lower_hit:
                outcome = 0
                exit_price = lower
                days = day
                break
            if upper_hit:
                outcome = 1
                exit_price = upper
                days = day
                break

        if outcome is None:
            exit_price = closes[final_i]

            if not np.isfinite(exit_price):
                continue
            days = horizon
            outcome = int(exit_price > entry_price)

        labels.iloc[i] = outcome
        holding_days.iloc[i] = days
        realised_return.iloc[i] = exit_price/entry_price - 1

    return labels, holding_days, realised_return

# part 2: use formulation from book referenced in notebook.

def excess_return_target(pooled_df, epsilon=0.0, return_col='ranking_return'):
    #continous regression target, each stocks forward return minus its equal-weighted universe average over the same period
    #we cancel out market-wide drift, isolating stock performance specifically. used for ranking rather than a label
    #epsilon: optional deadband. excess returns within +- epsilon from 0 can be treated as noise in a classification system. Kept small as large epsilon turns this into an outlier detection problem.

    if epsilon < 0:
        raise ValueError("Epsilon cannot be negative")

    required_cols = {'ticker', return_col}
    missing = required_cols.difference(pooled_df.columns)

    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    result = pooled_df.copy()
    universe_return = result.groupby(level=0)[return_col].transform('mean')

    result['excess_return'] = result[return_col] - universe_return

    if epsilon > 0:
        result['excess_return'] = result['excess_return'].mask(np.abs(result['excess_return']) < epsilon, 0.0) #if excess less than epsilon, then treat as noise

    return result

