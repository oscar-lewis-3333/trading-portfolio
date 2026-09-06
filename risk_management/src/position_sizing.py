import numpy as np
import pandas as pd

def kelly_sizing(win_prob, win_loss_ratio, frac=1.0):
    #return kelly-optimal bet size as a proportion of capital. frac gives scale frac for kelly (half kelly, full kelly etc)

    p = win_prob
    q = 1 - p
    b = win_loss_ratio

    f_star = p - (q/b)
    f = max(0, f_star) #if negative return 0

    return f * frac

def kelly_from_returns(returns, frac=1.0):

    #somewhat naively estimate kelly sizing from historical data

    if not isinstance(returns, pd.Series):
        raise TypeError("Returns must be a pandas Series")
    if not np.isfinite(frac) or not 0 <= frac <= 1:
        raise ValueError("Kelly fraction must be between 0 and 1 (hence finite)")

    clean = pd.to_numeric(returns.copy(), errors='raise').dropna().astype(float)

    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Returns must be finite")

    non_zero = clean[clean != 0]
    wins = non_zero[non_zero > 0]
    losses = non_zero[non_zero < 0]

    if wins.empty or losses.empty:
        return 0.0

    win_prob = len(wins)/len(non_zero)
    win_loss_ratio = wins.mean() / np.abs(losses.mean())

    return kelly_sizing(win_prob, win_loss_ratio, frac=frac)

def volatility_scaled_size(target_risk, asset_vol, max_size=1.0):

    #target risk:desired risk as part of the portfolio. asset_vol: assets volatility, max_size: most volatility the asset can have in the portfolio

    if asset_vol <= 0 or np.isnan(asset_vol):
        return 0.0
    size = target_risk/asset_vol

    return min(size, max_size)

def volatility_scaled_portfolio(returns_df, target_risk=0.02, lookback=63, max_size=0.25):

    #another portfolio layout where each weight is due to its volatility, which a max size included. target risk directly scales sizes.
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("Returns must be a pandas DataFrame")
    if returns_df.empty:
        raise ValueError("Returns cannot be empty")
    if not isinstance(lookback, (int, np.integer)) or isinstance(lookback, bool) or lookback < 2:
        raise ValueError("Lookback must be an integer and at least 2")
    if not np.isfinite(target_risk) or target_risk < 0:
        raise ValueError("Target risk must be finite and non-negative")
    if not np.isfinite(max_size) or max_size < 0:
        raise ValueError("Maximum size must be finite and non-negative")

    returns = returns_df.apply(pd.to_numeric, errors='raise').astype(float)
    if np.isinf(returns.to_numpy()).any():
        raise ValueError("Returns cannot contain infinity")

    vol = returns.rolling(lookback).std() * np.sqrt(252)
    safe_vol = vol.where(np.isfinite(vol) & vol.gt(0))

    sizes = target_risk/safe_vol
    sizes = sizes.clip(lower=0, upper=max_size).fillna(0) #do not allow shorting
    return sizes

