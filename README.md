# Quantitative Trading Portfolio

Quantitative finance projects written in Python, grouped by the market they trade. Assumptions, results and limitations live in the individual project READMEs rather than here.

## Layout

| Folder | Contents |
| --- | --- |
| [`original_portfolio/`](./original_portfolio/) | Nine projects on US equities, running from market statistics through to the paper-trading bot |
| [`uk_portfolio/`](./uk_portfolio/) | GBP-listed asset research, with a trading system coming soon |
| [`crypto_portfolio/`](./crypto_portfolio/) | Crypto research, with a crypto trading system in the works|

## Method

Later projects follow a similar pattern. A rule/strategy is chosen based off of an economic hypothesis. Development and holdout periods are fixed, with the holdout period preserved entirely for testing the strategy developed. Transaction costs are modelled, with fees only being charged on net assets traded. All strategies are compared with a suitable benchmark, with HAC testing, block bootstrapping, or both to test whether there is a statistical edge to our strategy. Equity curves are calculated and plotted, with a variety of risk management approaches being tested to see if there is positive effects.

Strategies that are not useful are kept for research purposes, but not implemented in practice.

## UK portfolio

Research concerning assets listed in GBP, with transaction costs based upon Trading212 offerings.

### Reversal signals

**Selected for implementation**

Daily cross-sectional reversal on UK single stocks, with a holdout of 2024-2025. Modelled with 10bp costs each side and 3.8% AER on idle cash. Risk management chosen to cap weights at 5%, with remainder cash. Consistently outperformed its benchmark, but failed to pass any significance test. Final verdict is promising results, but no statistically proven edge.

[View project](./uk_portfolio/reversal_signals/)

### ETF trend 

**Not selected for implementation**

Monthly long/cash trend following approach across 10 GBP funds covering equities, gilts, corporate bonds, gold and commodities. 6 month lookback beat a cash-composite benchmark over a 2023-2026 holdout period, with a borderline one-sided significance result. The strategy took on more exposure and risk than its benchmark, and was outperformed by a fully invested, equal weight buy and hold approach.

[View project](./uk_portfolio/etf_trend/)

A UK trading bot operating through Trading212 is coming soon, alongside further projects researching GBP-listed assets.

## Crypto portfolio

Research concerning cryptocurrencies, with transaction costs based upon Revolut X offerings.

### Crypto momentum 

**Selected for implementation**

Weekly absolute momentum on BTC, ETH against GBP. Each asset is held while its trailing 30d returns are positive, with cash being held otherwise. Over the 2024-2025 holdout, returned 96.50% after modelled transaction costs. Bootstrap confidence intervals include 0, so no statistically significant edge shown comparitively to its benchmark. Limited by only having a 2 asset universe, 2 year holdout and untested trading frequency.

[View project](./crypto_portfolio/crypto_momentum/)

A crypto trading bot through Revolut X is coming soon, alongside further projects researching cryptocurrencies.

## Original portfolio

Begin with building up quantitative finance and machine learning knowledge across various topics to be used later. From project 7 onwards, signal testing and risk management approaches were tested. Technical indicators showed no significance, with machine learning not adding anything. A momentum effect held up, but only on a subset of the market, and dissappeared when expanded to the full S&P500, so universe limited to that subset. Risk management strategies were tried, but all underperformed the equity curve significantly, so none were implemented.

As a result, the trading bot enacts the 63d momentum approach on the smaller universe through Alpaca paper trading. Execution is daily through cron, but rebalancing only occurs every 21 trading days.

### 1. Past market analysis
Historical price analysis, return important statistics, and multi-asset Sharpe ratio ranking across stocks, ETFs, and cryptocurrencies.

[View project](./original_portfolio/past_market_analysis/)

### 2. Technical analysis, signal generation and backtesting
Adaptive RSI, Bollinger thresholds, ADX-weighted composite trading signals and backtesting across various tickers

[View project](./original_portfolio/technical_analysis/)

### 3. Options pricing, stochastic modelling and implied volatility
Black-Scholes analytical pricing with SymPy, Monte Carlo simulation with SciPy, and implied volatility calculations.

[View project](./original_portfolio/options_pricing/)

### 4. Time series forecasting
ARIMA return modelling and GARCH/GJR volatility forecasting, with walk-forward backtesting and comparison against market implied volatility.

[View project](./original_portfolio/time_series_forecasting/)

### 5. Portfolio construction
Efficient frontier plotting for both long-only, short-allowed portfolios, alongside out-of-sample testing for max-Sharpe, min-variance and risk parity portfolios.

[View project](./original_portfolio/portfolio_construction/)

### 6. Machine learning fundamentals
Overfitting, feature importance, calibration and cross validation, established on well-behaved data before being applied to anything financial.

[View project](./original_portfolio/ml_fundamentals/)

### 7. Machine learning trading signals and momentum validation
Rigorous validation of ML on technical indicators, which found no significance, followed by a cross-sectional momentum signal that did hold up and that ML could not improve on. Ends with a forward-looking test of value and quality fundamentals.

[View project](./original_portfolio/ml_trading_signals/)

### 8. Risk management and position sizing
Kelly optimised, volatility scaled and CVaR-constrained sizing applied to the momentum signal. All four risk mechanisms came up short of the unadjusted strategy.

[View project](./original_portfolio/risk_management/)

### 9. Autonomous trading system
Automated execution of the universe and momentum signal from project 7, on Alpaca paper trading, run daily by cron, with a weekly summary after each simulated run.

[View project](./original_portfolio/trading_bot/)

## Libraries

Python, NumPy, pandas, Matplotlib, SciPy, SymPy, statsmodels, arch, scikit-learn, yfinance,
pandas_market_calendars, alpaca-py.

## About

Trading Portfolio built by Oscar Lewis, 4th year MMath student at the University of Warwick, to showcase skills in python, quantative finance and machine learning. For key details on individual projects see README's.

[LinkedIn](https://www.linkedin.com/in/oscar-lewis-aba333230) · [GitHub](https://github.com/oscar-lewis-3333)
