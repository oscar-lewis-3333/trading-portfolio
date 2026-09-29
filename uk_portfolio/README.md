# UK portfolio

Research concerning assets listed in GBP, with transaction costs based upon Trading212 offerings.

## Reversal signals

**Selected for implementation**

Daily cross-sectional reversal on UK single stocks, with a holdout of 2024-2025. Modelled with 10bp costs each side and 3.8% AER on idle cash. Risk management chosen to cap weights at 5%, with remainder cash. Consistently outperformed its benchmark, but failed to pass any significance test. Final verdict is promising results, but no statistically proven edge. An extension trying machine learning methods has been tried, with both an adaptive walk-forward and a fixed configuration are kept for further testing.

[View project](reversal_signals/)

## ETF trend 

**Not selected for implementation**

Monthly long/cash trend following approach across 10 GBP funds covering equities, gilts, corporate bonds, gold and commodities. 6 month lookback beat a cash-composite benchmark over a 2023-2026 holdout period, with a borderline one-sided significance result. The strategy took on more exposure and risk than its benchmark, and was outperformed by a fully invested, equal weight buy and hold approach. An extension using inverse volatility weightings was attempted, but was outperfomed on Sharpe, with a strictly negative 95% confidence interval when comparing equal weight and inverse volatility portfolios.

[View project](etf_trend/)

## Equity momentum

**Not selected for implementation**

Monthly cross-sectional momentum on shares listed in GBP. Parameters chosen annually using the previous 3 years' net Sharpe ratio. Over the 2024-2025 holdout, approach returned -6.65% with 10bp transaction costs each side, as opposed to its equal-weight benchmark returning -0.43%. The momentum strategy was more volatile, and had higher drawdowns. All Bootstrap intervals contain 0, hence no statistical edge shown. Limited by survivourship bias, alongside poor results comparitively to its benchmark.

[View project](equity_momentum/)

## ETF relative momentum

**Not selected for implementation**

Annually selected, monthly rebalanced momentum approach on GBP listen ETFs. Over 01/2024-08/2026, strategy returns £17,254.97 from £10,000 compared to £12,766.26 for its equal weight, rebalanced monthly, benchmark. However, volatility was increased, and 99.58% of the net profit came from one asset, gold. 95% confidence interval for net returns against the benchmark contains 0, and p-value was 0.0943, leading to no edge being shown. For the gold-concentration reason, and poor performance over the development period, this strategy is unlikely to be implemented in the future. Similarly to other projects, survivorship bias limits the findings.

[View project](etf_relative_momentum/)

A UK trading bot operating through Trading212 is coming soon, alongside further projects researching GBP-listed assets.

