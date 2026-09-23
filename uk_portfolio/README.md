# UK portfolio

Research concerning assets listed in GBP, with transaction costs based upon Trading212 offerings.

## Reversal signals

**Selected for implementation**

Daily cross-sectional reversal on UK single stocks, with a holdout of 2024-2025. Modelled with 10bp costs each side and 3.8% AER on idle cash. Risk management chosen to cap weights at 5%, with remainder cash. Consistently outperformed its benchmark, but failed to pass any significance test. Final verdict is promising results, but no statistically proven edge.

[View project](reversal_signals/)

## ETF trend 

**Not selected for implementation**

Monthly long/cash trend following approach across 10 GBP funds covering equities, gilts, corporate bonds, gold and commodities. 6 month lookback beat a cash-composite benchmark over a 2023-2026 holdout period, with a borderline one-sided significance result. The strategy took on more exposure and risk than its benchmark, and was outperformed by a fully invested, equal weight buy and hold approach.

[View project](etf_trend/)

## Equity momentum

**Not selected for implementation**

Monthly cross-sectional momentum on shares listed in GBP. Parameters chosen annually using the previous 3 years' net Sharpe ratio. Over the 2024-2025 holdout, approach returned -6.65% with 10bp transaction costs each side, as opposed to its equal-weight benchmark returning -0.43%. The momentum strategy was more volatile, and had higher drawdowns. All Bootstrap intervals contain 0, hence no statistical edge shown. Limited by survivourship bias, alongside poor results comparitively to its benchmark.

[View project](equity_momentum/)

A UK trading bot operating through Trading212 is coming soon, alongside further projects researching GBP-listed assets.

