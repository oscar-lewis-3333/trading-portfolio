## Original portfolio

Begin with building up quantitative finance and machine learning knowledge across various topics to be used later. From project 7 onwards, signal testing and risk management approaches were tested. Technical indicators showed no significance, with machine learning not adding anything. A momentum effect held up, but only on a subset of the market, and dissappeared when expanded to the full S&P500, so universe limited to that subset. Risk management strategies were tried, but all underperformed the equity curve significantly, so none were implemented.

As a result, the trading bot enacts the 63d momentum approach on the smaller universe through Alpaca paper trading. Execution is daily through cron, but rebalancing only occurs every 21 trading days.

### 1. Past market analysis
Historical price analysis, return important statistics, and multi-asset Sharpe ratio ranking across stocks, ETFs, and cryptocurrencies.

[View project](past_market_analysis/)

### 2. Technical analysis, signal generation and backtesting
Adaptive RSI, Bollinger thresholds, ADX-weighted composite trading signals and backtesting across various tickers

[View project](technical_analysis/)

### 3. Options pricing, stochastic modelling and implied volatility
Black-Scholes analytical pricing with SymPy, Monte Carlo simulation with SciPy, and implied volatility calculations.

[View project](options_pricing/)

### 4. Time series forecasting
ARIMA return modelling and GARCH/GJR volatility forecasting, with walk-forward backtesting and comparison against market implied volatility.

[View project](time_series_forecasting/)

### 5. Portfolio construction
Efficient frontier plotting for both long-only, short-allowed portfolios, alongside out-of-sample testing for max-Sharpe, min-variance and risk parity portfolios.

[View project](portfolio_construction/)

### 6. Machine learning fundamentals
Overfitting, feature importance, calibration and cross validation, established on well-behaved data before being applied to anything financial.

[View project](ml_fundamentals/)

### 7. Machine learning trading signals and momentum validation
Rigorous validation of ML on technical indicators, which found no significance, followed by a cross-sectional momentum signal that did hold up and that ML could not improve on. Ends with a forward-looking test of value and quality fundamentals.

[View project](ml_trading_signals/)

### 8. Risk management and position sizing
Kelly optimised, volatility scaled and CVaR-constrained sizing applied to the momentum signal. All four risk mechanisms came up short of the unadjusted strategy.

[View project](risk_management/)

### 9. Autonomous trading system
Automated execution of the universe and momentum signal from project 7, on Alpaca paper trading, run daily by cron, with a weekly summary after each simulated run.

[View project](trading_bot/)
