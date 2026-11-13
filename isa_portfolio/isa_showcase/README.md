# ISA system showcase

The long-only, ISA-designed system for holding assets in a UK Stocks and Shares ISA. The current chosen broker is Trading 212, with paper trading beginning at the start of November 2026. The system contains five primary parts, with a sixth to combine and optimise the others:

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
- Candidates are measured against what they add to the existing system, primarily the change in the book's Sharpe ratio from a small weight. This is done using block-bootstrap significance testing, Holm-adjusted p-values, placebo tests, seperating best/worst trades, and results per time-period
- Transactions costs are spreads measured on the ISA's lines.

## Results

Backtested, begin in February 2001 with £10,000, end in September 2026. The book is after modelled costs, with the indices not including fund fees. Final results yield the following table:

| | CAGR | Sharpe | max drawdown | £10,000 becomes |
|---|---|---|---|---|
| core alone | 8.6% | 0.55 | −35.7% | £83,500 |
| whole book | 18.5% | 1.53 | −10.6% | £778,600 |
| MSCI World (net, GBP) | 7.8% | 0.45 | −39.9% | £68,900 |
