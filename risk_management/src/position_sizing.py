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

    #somewhat naively estimate

    returns = returns.dropna()
    wins = returns[returns >0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    win_prob = len(wins)/len(returns)
    win_loss_ratio = wins.mean() / abs(losses.mean())

    return kelly_sizing(win_prob, win_loss_ratio, frac=frac)

def volatility_scaled_size(target_risk, asset_vol, max_size=1.0):

    #target risk:desired risk as part of the portfolio. asset_vol: assets volatility, max_size: most volatility the asset can have in the portfolio

    if asset_vol <= 0 or np.isnan(asset_vol):
        return 0.0
    size = target_risk/asset_vol

    return min(size, max_size)

def volatility_scaled_portfolio(returns_df, target_risk=0.02, lookback=63, max_size=0.25):

    vol = returns_df.rolling(lookback).std() * np.sqrt(252)
    sizes = target_risk/vol
    sizes = sizes.clip(upper=max_size).fillna(0)
    return sizes

