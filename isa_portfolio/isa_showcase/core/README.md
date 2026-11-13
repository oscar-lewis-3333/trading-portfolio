# Core

The core is the books default setting, from which every other holding is funded from. It is held 100% of the time that a rebound/defensive basket is not being held.

## What it holds

The core holds the following 3 different ETFs, with equal weighting of 1/3 per ETF. This is rebalanced monthly, and calculated after knowing what weighting it has in the book (see [Allocator](../allocator/))

- **IWFQ**: MSCI World Sector Neutral Quality
- **IWFM**: MSCI World Momentum
- **MINV**: MSCI World Minimum Volatility

It was chosen over a single world tracker (VWRP) because the factors tend to do well at different times: momentum in trends, quality and minimum volatility in sell-offs. It ended higher than VWRP in each of three stress periods fixed in advance (2008, Covid, 2022), although its full-period edge over VWRP is not statistically significant.

This core was chosen over a single world tracer (VWRP) due to its theoretical advantage of different factors performing better in different market conditions, namely momentum tending to perform well in trends, and minumum volatility/quality performing better in sell-offs. The results were better than VWRP over the full period, with specifically better performances in 2008, COVID19, 2022, but over the full period the difference was statistically insignificant.

## Results

From £10,000, after modelled costs:

| window | CAGR | Sharpe | max drawdown |
|---|---|---|---|
| 2001–2026 | 8.6% | 0.55 | −35.7% |
| 2006–2026 | 10.4% | 0.75 | −26.0% |
| 2020–2026 | 10.8% | 0.72 | −12.3% |

The main weaknesses are (2001–03 and 2008–09), which is what the [defensive layer](../defensive/) is designed to handle.
