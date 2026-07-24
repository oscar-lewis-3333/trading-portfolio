import numpy as np
import pandas as pd
from scipy import stats

def compute_returns(df):
    """Add daily returns and log returns columns."""
    df = df.copy() #working on a copy of the dataframe, not the actual one - just avoids errors 
    df['Return']     = df['Close'].pct_change() #we define a column 'return' to be the percentage change from interval to interval (often day to day)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))  
    df.dropna(inplace=True) #importantly, acts on the copy
    return df

def summary_statistics(df):
    """Compute key statistics on daily returns."""
    r = df['Return'] 
    return {
        'mean_daily_return':  r.mean(),
        'std_daily_return':   r.std(),
        'annualised_return':  r.mean() * 252, # ~252 trading days per year
        'annualised_vol':     r.std() * np.sqrt(252),
        'sharpe_ratio':       (r.mean() / r.std()) * np.sqrt(252), #sharpe ratio for the year, "return per unit risk" > 1 good, > 2 excellent.
        'skewness':           stats.skew(r), #whether distrubution leans left or right. <0 implies big losses more common than large gains
        'kurtosis':           stats.kurtosis(r),
        'max_drawdown':       max_drawdown(df['Close']),
        'var_95':             np.percentile(r, 5),   #5th percentile of returns. on 95% of days you wont lose more than this
    }

def max_drawdown(prices):
    """Maximum peak-to-trough decline."""
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown.min()

def normality_test(df):
    """Test whether returns are normally distributed."""
    r = df['Return'].dropna()
    stat, p = stats.normaltest(r)
    return {
        'statistic': stat,
        'p_value':   p,
        'is_normal': p > 0.05
    }

def rolling_stats(df, window=30): #compute rolling monthly (unless specified otherwise) return means/volatility
    """Add rolling mean and volatility columns."""
    df = df.copy()
    df[f'Rolling_Mean_{window}'] = df['Return'].rolling(window).mean()
    df[f'Rolling_Vol_{window}']  = df['Return'].rolling(window).std() * np.sqrt(252)
    return df


#The above is all valid for a single ticker. We now add more than one ticker.

def compare_assets(tickers, period="2y", interval="1d"):
    from data_loader import fetch_price_data
    results = []
    for ticker in tickers:
        print(f"fetching{ticker}...")
        try:
            df = fetch_price_data(ticker, period=period, interval=interval)
            df = compute_returns(df)
            s = summary_statistics(df)
            s['ticker'] = ticker
            results.append(s) #add to results all data we need for a comparison of 2 tickers
        except Exception as e: #if fails for a ticker, we return an error safely
            print(f"Failed for {ticker}: {e}")    
    ranking = pd.DataFrame(results) #turn such result into a dataframe
    ranking = ranking.set_index('ticker') #row-label is ticker, column labels are all different criterion from summary_statistics
    ranking = ranking.sort_values('sharpe_ratio', ascending=False)
    return ranking
