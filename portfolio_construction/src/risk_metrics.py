import numpy as np 
import pandas as pd

from sklearn.covariance import LedoitWolf

def build_returns_matrix(tickers, period="5y", interval="1d"):
    #fetch price data for multiple tickers and reutrn a dataframe of aligned daily returns, it is important that these are aligned for the covariance matrix

    import sys
    from data_loader import fetch_price_data

    returns = {}
    for ticker in tickers:
        df = fetch_price_data(ticker, period=period, interval=interval)
        returns[ticker] = df['Close'].pct_change()

    returns_df = pd.DataFrame(returns).dropna()
    return returns_df

def annualised_stats(returns_df):
    #compute anuallsised returns and covariance matrix from daily returns dataframe
    mu = returns_df.mean() * 252  
    cov = returns_df.cov() * 252
    return mu, cov 

def correlation_matrix(returns_df):
    #gives pairwise correlation between tickers
    corr = returns_df.corr()
    return corr

def shrunk_covariance(returns_df):
    #we use Ledoit-Wolf shrinkage to esimate a stable covariance matrix, which is important as seen in notebook.

    lw = LedoitWolf().fit(returns_df.values)
    cov_shrunk = lw.covariance_ * 252 
    return cov_shrunk, lw.shrinkage_

def risk_contributions(weights, cov):
    #compute each assets contribution to total portfolio risk.
    weights = np.array(weights)
    port_vol = np.sqrt(weights.T @ cov @ weights)
    marginal = cov @ weights / port_vol
    contributions = weights * marginal
    return contributions, contributions / port_vol #absolute and percentage contributions
