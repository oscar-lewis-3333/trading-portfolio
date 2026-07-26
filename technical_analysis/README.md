# TECHNICAL ANALYSIS AND SIGNAL GENERATION

## OVERVIEW
Project takes tickers and uses their real-world market data to indicate buy/sell signals. Comes with a backtest signal, where we compare buying and selling according to our signals, with just buying and holding.

## FEATURES
- Computes SMA (Simple moving average) and EMA (exponential moving average) of the ticker price over different windows
- Computes RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence) and Bollinger Bands to tell us about the momentum of stock price relative to its recent prices. 
- Computes a variety of signals to give indications as to whether it's a good moment to buy or sell. These include MACD, RSI, SMA and Bollinger crossover signals
- Combines the above signals with a trend signal, weighted with the ADX (Average Direction Index), to weight the trend more when its stronger, less when its weaker
- Applies trend regime and volume filters alongside a cooldown to combat false signals
- Uses Bullish/Bearish Divergence alongside this composite signal to improve accuracy
- Plots professional visuals comparing the 2 straegies over time

## KEY DESIGN DECISIONS
- Chose to have continuously changing upper and lower thresholds when computing upper, lower bollinger bands %B instead of classically fixed boundaries of 0 and 1 (as a result of %B landing between 0 and 1 95% of the time). Instead, we look at the 10%, 90% quantiles and use those as our lower and upper thresholds respectively.
- Chose to do similar with the RSI. Classically, RSI > 70 implies overbought, and <30 implies oversold. We changed this to a percentile model based on the RSI rolling value, choosing the 20%,80% quantiles.
- Chose to weight different signals differently when constructing the composite signal, specifically if the ticker was in a strong trend (up or down) we weighted the trend signal significantly more than other signals. This was to avoid false sell signals and a lack of buy signals when a ticker was in a strong upturn (and vice versa)

## BACKTEST RESULTS

| Ticker  | Strategy_Return_% | BuyHold_Return_% | Outperformance_% |
|---------|-------------------|------------------|-------------------|
| SPY     | 20.36              | 40.35             | -19.99            |
| JNJ     | 73.65              | 71.89             | 1.76              |
| SOL-USD | -26.95             | -59.06            | 32.11             |
| TSLA    | 5.74               | 45.15             | -39.41            |


## KEY FINDINGS
- On the tickers tested, our strategy generally seems to predict the buy/sell moments to a pretty good degree. Whilst there are some incorrect signals, a visual estimate of 80-90%, of signals are accurate after taking lag into account, which will be addressed below. The system is designed to detect large shifts in the market as they happen and often after a signal, there is a noticeable change in that direction (i.e the signal is near a turning point).
- On tickers that are more volatile, such as cryptocurrencies or more developing markets, our system seems to outperform the buy and hold strategies. This is positive as it tells us, even with the lag, the system still correctly picks good moments to buy and sell.
- On tickers that are less volatile, such as the S&P 500 and recently NVIDIA, the system seems to underperform the buy and hold strategy. This is for a few reasons. One is that the system is designed not to purchase the ticker when the period starts, but to wait for a buy opportunity. In the S&P 500 example given in the attached notebook, half of the 'losses' (compared to buy and hold) occur in the first 6 months where no purchase has taken place. Another is that, since we sell all holdings at the first sell signal and we wait for a significant upturn for a buy signal, of which not many happen as the stock is not volatile (see past_market_analysis for further details), we are holding cash whilst the asset is increasing, so losing money comparatively. The lag also plays a big factor here, if our buys/sells were 2-3 days prior, we would have significantly more, but this is an inherent problem in the system that cannot be fixed unless we implement a predictive model. This is an aim for a later project.
- Volatility isnt the only factor which affects this though. Recall JNJ from past_market_analysis had sharpe 1.63 and low volatility, but here our system beats the buy and hold method by 1.76%, showing that low volatility doesn't automatically mean our system underperforms, but the nature of the changes is what matters.

## LIMITATIONS
- No modelling of transactions costs, dividends/staking which are both common with tickers
- System only full buys or full sells, no in between. We either have all cash in the ticker, or none, no partial positions
- All signals used depend on past data, so we get an element of inherent lag. This is due to it taking time for the data to reach the point of causing a sell signal
- Lack of testing across bear markets
- System is designed to catch large shifts in the market, so when a stock increases pretty consistently, our strategy seems to fall behind the buy and hold strategy

## LIBRARIES USED
- yfinance - importing real-time market data
- pandas - data manipulation for computing many indicators for each day
- numpy - numerical computation
- matplotlib - visualisation of our strategy, buy/sell signals, comparison with buy and hold

## PROJECT STRUCTURE
    src/ 
        indicators.py - compute various indicators to be used in signals
        signals.py - designs individual signals, filters to be combined robust_composite_signal
        plotting.py - visualising both strategies for comparisons in technical.ipynb
    notebooks/
        technical.ipynb - main analysis, use of all previous python files
   

## USAGE
``` python
from data_loader import fetch_price_data  #from past_market_analysis project

from indicators import add_all_indicators
from signals import backtest_signals, robust_composite_signal
from plotting import plot_backtest, plot_signals

ticker = "SPY" 
df = fetch_price_data(ticker, period="2y", interval="1d")
df = add_all_indicators(df)
df = robust_composite_signal(df)
df = backtest_signals(df)
```