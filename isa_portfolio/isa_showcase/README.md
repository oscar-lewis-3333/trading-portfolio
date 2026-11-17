# ISA system showcase

The long-only, ISA-designed system for holding assets in a UK Stocks and Shares ISA. The current chosen broker is Trading 212, with paper trading beginning at the start of November 2026. The system contains five primary parts, with a sixth to combine and optimise the others:

## Results

Backtested, begin in February 2001 with £10,000, end in September 2026. The book is after modelled costs, with the indices not including fund fees. Final results yield the following table:

| | CAGR | Sharpe | max drawdown | £10,000 becomes |
|---|---|---|---|---|
| core alone | 8.7% | 0.56 | −35.7% | £85,000 |
| whole book | 18.9% | 1.43 | −12.0% | £845,300 |
| MSCI World (net, GBP) | 7.8% | 0.45 | −39.9% | £68,900 |

### Longer history, 1979–2026

| period | book Sharpe | core-path Sharpe | book return/vol | core-path return/vol | book max drawdown | core-path max drawdown | cash rate |
|---|---|---|---|---|---|---|---|
| 1979–2026 | 0.89 | 0.63 | 1.30 | 1.01 | −31% | −38% | 5.9% |
| 1979–1989 | 0.51 | 0.62 | 1.14 | 1.18 | −31% | −38% | 12.6% |
| 1990–2000 | 0.52 | 0.24 | 1.06 | 0.72 | −14% | −33% | 8.0% |
| 2001–2026 | 1.43 | 0.92 | 1.65 | 1.13 | −12% | −20% | 2.4% |

### Forward estimate

A forward Sharpe of 0.9-1.3 is estimated going forward, the lower end if the previous regime of higher interest rates returns (and the system does not change accordingly), and the higher end if the current regime stays relatively consistent. Work on classifying growth and inflation regimes, to lean into the assets that suit each one, is in progress and aims to keep the book towards the upper end, but this is not completed yet. For more detail, see: [Allocator](allocator/README.md) or [Sleeves](sleeves/README.md)


## The layers

| folder | what it is | what is shown |
|---|---|---|
| [Core](core/) | Equal-weight MSCI World Sector Neutral Quality, Momentum and Minimum Volatility ETF core, held 100% of time outside of a defensive/rebound period | all |
| [Defensive](defensive/) | The defensive basket designed for recessions, alongside an exit/re-entry market recession trigger | trigger dates, final results shown. specific trigger and holdings private |
| [Rebound](rebound/) | A basket designed for best performance in the recovery period after a recession | only results |
| [Overlay](overlay/) |Assets that are held under certain market conditions, at month-endings | combined overlays, not individual ones |
| [Sleeves](sleeves/) | Assets that are held under certain market conditions. Funding for such holdings is from the core | combined sleeves, not individual ones |
| [Allocator](allocator/) | Combining all layers into one system, testing portfolio optimisation techniques, comparing final portfolio against major indices, how each layer adds to the final product | big-picture results |

## Research process

- Each idea began with an economic hypothesis for why it could hold, before testing began
- Each rule is then fixed with a hashed protocol before any returns data is computed/seen. Any changes after seeing results are hashed amendments
- Candidates are measured against what they add to the existing system, primarily the change in the book's Sharpe ratio from a small weight. This is done using block-bootstrap significance testing, Holm-adjusted p-values, placebo tests, separating best/worst trades, and results per time-period
- Transaction costs are spreads measured on the ISA's lines.


## Limitations

- Core ETFs history begins in 11/2012 and 10/2014 (MINV, then IWFQ, IWFM), so for preceding history, index data was used.
- The backtest is in-sample, so true performance depends on its forward live performance
- Some assets used had a bull period since 2000, so their corresponding sleeves may not hold up if they were to have a bear period
- The defensive trigger has too few events in this window to be tested extensively