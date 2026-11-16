# The ISA book and its allocator

Combine all aspects of the system into one, the core, defensive/rebound layers, sleeves and overlays, into one cohesive book using portfolio optimisation methods. The main concern is how much capital to move out of the core and into each sleeve/overlay. I ended up fixing a method which scales shrinkage sleeve weights by a common factor, capping the largest sleeve weight at 20%, and the minimum core weight at 30%.

## The book against major indices

Begin on 1st February 2001 with £10,000 and trade according to the allocator. The below comparison is with major stock indices, with the indices having 0bp assumed fund fee, and our system assuming modelled transaction costs.

| | CAGR | Sharpe | max drawdown | £10,000 becomes |
|---|---|---|---|---|
| ISA book | 17.9% | 1.48 | −9.9% | £684,100 |
| MSCI World | 7.8% | 0.45 | −39.9% | £68,900 |
| MSCI USA (≈ S&P 500) | 8.8% | 0.49 | −41.4% | £86,300 |
| MSCI UK | 5.8% | 0.32 | −40.3% | £42,900 |

## Weighting methods

The chosen rule, as stated above, scales each sleeve's shrinkage weight by a constant, with caps and floors. This was compared with various other portfolio optimisation techniques such as risk parity, inverse volatility, hierarchical risk parity, minimum CVaR and the Markowitz tangency portfolio with shrunk means/covariance, each tested with varying caps/floors. None beat the method I fixed above, with the closest being risk parity (−0.05 Sharpe [−0.20, +0.11]). [`src/engine.py`](src/engine.py) gives book construction and metrics, and [`src/solvers.py`](src/solvers.py) gives the optimisers.

## Before 2001: how much of this survives?

The book was built in-sample on 2001-2026. So, I re-ran the rule on 1979-2000 using index or futures histories where ETFs did not yet exist as substitutes for their real counterparts. The core itself has no usable history pre-2001, so for this period, the core is fixed as a standard world equity (US market to mid-1990, then the developed-market index), while keeping rebound, defensive features fixed, leaving the earlier years to test sleeves/overlays and the defensive/rebound layers, not the core. "Core path" is defined to be the book without the sleeves/overlays. 

| period | book Sharpe | core-path Sharpe | book return/vol | core-path return/vol | book max drawdown | core-path max drawdown | cash rate |
|---|---|---|---|---|---|---|---|
| 1979–2026 | 0.83 | 0.63 | 1.27 | 1.01 | −31% | −38% | 5.9% |
| 1979–1989 | 0.51 | 0.62 | 1.14 | 1.18 | −31% | −38% | 12.6% |
| 1990–2000 | 0.29 | 0.24 | 0.88 | 0.72 | −14% | −33% | 8.0% |
| 2001–2026 | 1.48 | 0.92 | 1.73 | 1.13 | −10% | −20% | 2.4% |

Sharpe is defined to be the returns over cash per unit risk, so the high interest rates from 1979-2000 kill the Sharpes of those eras, hence a 'return/vol' column is added to show the difference.


Before 2001 the sleeves and overlays didn't add to the core path's Sharpe (−0.03 [−0.21, +0.15], against +0.62 [+0.39, +0.86] since 2001, both measured on the original fund histories), though they still reduce the drawdowns. Primarily one asset's edge depends on the low-interest-rate economic conditions of the past 25 years, and breaks in an era with higher interest rates, while some of the economic conditions the overlays were founded upon did not apply in that era. A forward Sharpe of 0.9-1.3 is estimated going forward, the lower end if the previous regime of higher interest rates returns (and the system does not change accordingly), and the higher end if the current regime stays relatively consistent. Work on classifying growth and inflation regimes, to lean into the assets that suit each one, is in progress and aims to keep the book towards the upper end, but this is not completed yet.
