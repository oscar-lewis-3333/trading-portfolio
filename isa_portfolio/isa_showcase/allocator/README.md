# The ISA book and its allocator

Combine all aspects of the system into one, the core, defensive/rebound layers, sleeves and overlays, into one cohesive book using portfolio optimsation methods. The main concern is how much capital to move out of the core and into each sleeve/overlay. I ended up fixing a method which scales shrinkage sleeve weights by a common factor, capping the largest sleeve weight at 20%, and the minimum core weight at 30%.

## The book against major indices

Begin on 1st February 2001 with £10,000 and trade according to the allocator. The below comparison is with major stock indices, with the indices having 0bp assumed fund fee, and our system assuming modelled transaction costs.

| | CAGR | Sharpe | max drawdown | £10,000 becomes |
|---|---|---|---|---|
| ISA book | 18.5% | 1.53 | −10.6% | £778,600 |
| MSCI World | 7.8% | 0.45 | −39.9% | £68,900 |
| MSCI USA (≈ S&P 500) | 8.8% | 0.49 | −41.4% | £86,300 |
| MSCI UK | 5.8% | 0.32 | −40.3% | £42,900 |

## Weighting methods

The chosen rule, as stated above, scales each sleeve's shrinkage weight by a constant, with caps and floors. This was compared with various other portfolio optimisation techniques such as risk parity, inverse volatility, hierarchial risk parity, minimum CVaR and the markowitz tangency portfolio with shrunk means/covariance, each tested with varying caps/floors. None beat the method I fixed above, with the closest being risk parity (−0.05 Sharpe [−0.20, +0.11]). [`src/engine.py`](src/engine.py) gives book construction and metrics, and [`src/solvers.py`](src/solvers.py) gives the optimisers
