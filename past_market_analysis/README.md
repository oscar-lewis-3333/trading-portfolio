# MARKET DATA ANALYSIS PROJECT:

## OVERVIEW
Given a ticker, or array of tickers, a time period and an interval, we use quantitative analysis to produce statistic metrics from real-time market data and uses such metrics to compare a collection of tickers.

## FEATURES 
- Import real-time market data using Yahoo Finance
- Compute daily returns, log returns, cumulative returns, rolling volatility
- Calculate key statistical metrics such as: Sharpe-ratio, skewness, max-drawdown, VaR
- Output contains multi-panel visualations of assets
- Output contains multi-asset comparision ranked by Sharpe ratio descending
- Output contains statistical normality testing

## LIBRARIES USED
- yfinance - market data
- pandas - data manipulation, specifically placing market data into dataframes for analysis
- numpy - numerical computation
- scipy - statistical analysis
- matplotlib - visualisation of market data, key statistics

## PROJECT STRUCTURE
   src/ 
    data_loader.py - downloads, saves, reads real-time market data
    analysis.py - performs detailed, quantitative, multi-asset statistical analysis
    plotting.py - visualising functions to be used in ticker_analysis.ipynb
   notebooks/
    ticker_analysis.ipynb - main analysis, use of all previous python files
   data/ - saved market data

## USAGE
```python

from src.data_loader import fetch_price_data
from src.analysis import compute_returns, summary_statistics, compare_assets
from src.plotting import plot_price_history, plot_returns_analysis

# Single Asset:
df = fetch_price_data("AAPL", period="2y")
df = compute_returns(df)
stats = summary_statistics(df)

#Multi asset comparison

tickers = ["GLD", "BTC-USD", "AAPL", "AMZN", "GS", "GOOGL"]
ranking = compare_assets(tickers, period="3y")
```
## FINDINGS
Across the past 4 years, JP Morgan and Goldman Sachs stocks have consistently peformed well when judging by sharpe ratio. This ratio measure the ratio of the mean returns and the standard deviation of the returns and hence can be interpreted as the return per unit risk. JP Morgan had lower annual returns, but also lower volatility compared with Goldman Sachs. Another interesting observation is that the S&P 500 had a significantly smaller max drawdown (difference between largest peak and largest trough) compared with other tickers (Gold, Google, JP Morgan etc.) whilst still having a sharpe ratio greater than 1, making it a safe investment over the past 4 years.
The same trend follows with S&P 500 when we limit to 2 years. Cryptocurrencies have performed badly over the past 2 years, with SOL, ETH having negative Sharpe ratio and BTC having a Sharpe ratio (0.175) close to 0. On the contrary, JNJ (Healthcare) has performed exceptionally well in many metrics over that same period, with Sharpe ratio 1.63 and maximum drawdown of -0.144, the smallest drawdown of all tickers tested. Comparing to the 4 year period, JP Morgan and Goldman Sachs have effectively swapped places, with Goldman Sachs having a noticeably better Sharpe ratio (1.44 compared to 1.23 respectively) coming from a significantly higher return, albeit a higher volatility. This shows how the 'better' investment can vary significantly based on what time period you test over, a concept called lookback bias.