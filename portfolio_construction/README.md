# PORTFOLIO CONSTRUCTION

## OVERVIEW
This project discusses the most efficient ways to optimise a portfolio of tickers, a process called Markowitz optimisation. We plot the efficient frontier, discuss the tangency portfolio, the limitations it has (instability, turnover), and the methods of overcoming this (risk parity). We end with a discussion on the benefits and risks of shorting.

## FEATURES
- Correlation matrix estimator across tickers
- Construction and visualisation of the efficient frontier
- Maximum sharpe, minimum variance portfolio constructions
- Ledoit-Wolf shrinkage covariance estimation 
- Construction of the risk parity portfolio without returns data 
- Weight stability testing across sub-periods
- Out-of-sample testing of each portfolio-type (risk parity, max sharpe etc.)
- Turnover and transaction cost analysis across portfolio-types
- Weight caps
- Long-only vs shorting portfolio analysis

## KEY DESIGN DECISIONS
- SLSQP scipy solver used as its one of the only numerical approximators which solves alongside allowing bounds and constraints on the function. This is needed for us, as we cannot hold more than 100% of an asset (without shorting), and we need the constraints that all weights (elements of the vector) sum to 1, and satisfy the target return condition.
- We set a non-zero lower bound of 0.001 for our risk parity calculations, this is due to the weighting, if bounds could be set abritrarily small, then we would be dividing by a number very close to 0, which is unstable.
- Turnover halved as we want to define the process of liquidating in one asset and fully buying another asset to be 1.0 (100%) not 2.0 (200%)
## RESULTS

### Correlation matrix

|      | AAPL  | NVDA   | SPY   | JNJ    | GLD   | TSLA   |
|------|------|------|------|------|------|------|
| AAPL | 1.000 | 0.479  | 0.725 | 0.137  | 0.056 | 0.466  |
| NVDA | 0.479 | 1.000  | 0.708 | -0.113 | 0.081 | 0.463  |
| SPY  | 0.725 | 0.708  | 1.000 | 0.161  | 0.148 | 0.582  |
| JNJ  | 0.137 | -0.113 | 0.161 | 1.000  | 0.048 | -0.042 |
| GLD  | 0.056 | 0.081  | 0.148 | 0.048  | 1.000 | 0.055  |
| TSLA | 0.466 | 0.463  | 0.582 | -0.042 | 0.055 | 1.000  |

### Portfolio comparison (in-sample, full period)

| Strategy | Return (%) | Volatility (%) | Sharpe |
|---|---|---|---|
| Equal weight | 24.37 | 21.95 | 1.110 |
| Min variance | 14.35 | 11.36 | 1.263 |
| Risk parity | 18.88 | 13.98 | 1.351 |
| Max Sharpe | 23.99 | 14.60 | 1.643 |

### Weight instability across sub-periods (long-only max Sharpe)

| Period | AAPL | NVDA | SPY | JNJ | GLD | TSLA |
|-----|-----|-----|-----|-----|-----|-----|
| 2021-08 to 2022-10 | 27.6 | 0.0 | 0.0 | 63.1 | 0.0 | 9.4 |
| 2022-10 to 2024-01 | 0.0 | 40.0 | 0.0 | 0.0 | 60.0 | 0.0 |
| 2024-02 - 2025-05 | 0.0 | 8.0 | 0.0 | 10.3 | 78.2 | 3.4 |
| 2025-05 to 2026-07 | 7.5 | 12.3 | 28.2 | 49.0 | 3.0 | 0.0 |
| **Std dev (pp)** | **13.0** | **17.4** | **14.1** | **30.2** | **39.8** | **4.4** |

### Estimated annual mean returns by sub-period (%)

| Period | AAPL | NVDA | SPY | JNJ | GLD | TSLA |
|------|------|------|------|------|------|------|
| 1 | 10.9 | -11.2 | -5.7 | 5.2 | -7.2 | 15.8 |
| 2 | 17.0 | 132.1 | 20.0 | -3.5 | 17.6 | 0.3 |
| 3 | 16.8 | 65.6 | 14.7 | 2.5 | 38.1 | 56.3 |
| 4 | 33.5 | 53.7 | 25.4 | 45.3 | 21.5 | 20.6 |
| **Std dev (pp)** | **9.7** | **58.7** | **13.6** | **22.2** | **18.7** | **23.7** |

### Out-of-sample performance (60/40 split)

| Strategy | IS Sharpe | OOS Sharpe | OOS Return (%) | OOS Vol (%) |
|---|---|---|---|---|
| Max Sharpe | 1.451 | 1.574 | 33.16 | 21.06 |
| Max Sharpe (shrunk) | 1.450 | 1.567 | 33.36 | 21.29 |
| Min variance | 0.735 | 1.881 | 26.33 | 14.00 |
| Risk parity | 0.971 | 1.930 | 27.56 | 14.28 |
| Equal weight | 0.935 | 1.388 | 29.40 | 21.19 |

### Turnover when re-optimised each period

| Strategy | Avg turnover (%) | Max turnover (%) | Cost per rebalance (%) |
|---|---|---|---|
| Max Sharpe | 68.9 | 100.0 | 0.069 |
| Min variance | 19.5 | 34.3 | 0.020 |
| Risk parity | 10.6 | 17.0 | 0.011 |
| Equal weight | 0.0 | 0.0 | 0.000 |

### Effect of a maximum weight cap (max Sharpe)

| Cap | Sharpe | Return (%) | Vol (%) | AAPL | NVDA | SPY | JNJ | GLD | TSLA |
|---|---|---|---|---|---|---|---|---|---|
| None | 1.635 | 24.11 | 14.75 | 0.0 | 20.7 | 0.0 | 39.9 | 39.4 | 0.0 |
| 50% | 1.635 | 24.11 | 14.75 | 0.0 | 20.7 | 0.0 | 39.9 | 39.4 | 0.0 |
| 35% | 1.604 | 25.55 | 15.92 | 6.9 | 23.1 | 0.0 | 35.0 | 35.0 | 0.0 |
| 25% | 1.460 | 26.36 | 18.05 | 9.3 | 25.0 | 15.7 | 25.0 | 25.0 | 0.0 |

### Long-only vs shorting allowed (max Sharpe)

| Strategy | Return (%) | Vol (%) | Sharpe | Gross exposure (%) | Largest short (%) |
|---|---|---|---|---|---|
| Long only | 23.99 | 14.60 | 1.643 | 100.0 | 0.0 |
| Shorting allowed | 32.36 | 18.71 | 1.730 | 227.5 | -63.2 |

| Strategy | AAPL | NVDA | SPY | JNJ | GLD | TSLA |
|---|---|---|---|---|---|---|
| Long only | 0.0 | 20.2 | 0.0 | 40.8 | 39.0 | 0.0 |
| Shorting allowed | 17.0 | 35.6 | -63.2 | 59.3 | 51.8 | -0.5 |


## KEY FINDINGS
- The process of markowitz optimisation (finding an optimal portfolio) is unstable as shown by the stability test, with the issue not being the covariance matrix (as shown by Ledoit-Wolf optimisation changing minimal) but the historical means, not the method. This makes it difficult to choose the best weights as they are changing significantly over different periods of time for max-sharpe portfolio causing ~0.07% per restructuring per our estimation. This is due to the max-sharpe portfolio eliminating the assets that do not perform well, and puts all its eggs in 1/2 baskets. This is what causes the instabilty as there is no portfolio diversification
- When risk parity portfolio is chosen, the process is significantly more stable as the portfolio is more diversified. This reduces turnover per restructure and this portfolio performs the best of tested portfolio-types when the out-of-sample test was performed, with the max sharpe portfolio getting slightly better (better market period) but getting trumped by the risk parity portfolio.
- When judging by all statistical metrics against other main portfolio-types (max-sharpe, minimum variance, risk parity), they all outperform the equal weight portfolio. This includes the out-of-sample test, the standard test over the past 5 years and even shorting portfolios. This is evidenced by its position relative to the efficient frontier.
- The shorting portfolio does increase the returns per risk on past data compared to other optimal portfolios, which is evidenced by the plot of their 2 efficient frontiers. This comes at a price which cannot be seen on these plots, the gross exposure. The max-sharpe portfolio allowing shorting does increase the sharpe ratio comparitively to its counterpart (long-only max-sharpe) by ~5%, but its gross exposure is 227%, meaning we have to take on 2.27x leverage for the portfolio, without even considering short costs and the potentially unbounded losses that could come from a short. In practice, this is why this portfolio is unlikely to be chosen.

## LIMITATIONS
- Historical means used as expected returns data throughout project, which is where a large uncertainty arises from, as discussed in the section around stability in the notebook. When splitting by period, the max-sharpe portfolio shifts positions by an average of 69% per period change, implying that the portfolio is chosen to fit data - may not be reliable in the future.
- Costs of trading modelled as flat and symmetric 0.1% rate, no modelling of borrowing costs when considering shorting, or even availability of shorts.
## LIBRARIES USED
- numpy — matrix operations, quadratic forms, weight arithmetic
- pandas — returns alignment, covariance/correlation estimation, results tables
- scipy.optimize — constrained optimisation (SLSQP) for all portfolio solvers
- scikit-learn — Ledoit-Wolf shrinkage covariance numerical approximator 
- matplotlib — efficient frontier (comparison to arbitrary portfolios and specific portfolios) and comparisons of short vs long efficient frontiers
- yfinance — historical price data (via past_market_analysis project)

## PROJECT STRUCTURE
    src/
        risk_metrics.py — returns matrix construction, annualised statistics, correlation, Ledoit-Wolf shrinkage, risk contributions
        optimisation.py — min variance, max Sharpe, risk parity, efficient frontier, stability testing, out-of-sample comparison, turnover analysis, shorting comparison
        plotting.py     — efficient frontier with arbitrary marked portfolios and specific 'optimal' portfolios, long-only vs shorting efficient frontier comparison
    notebooks/
        portfolio.ipynb — main analysis of src .py files

## USAGE
```python
from risk_metrics import build_returns_matrix, annualised_stats, risk_contributions
from optimisation import max_sharpe_portfolio, min_variance_portfolio, risk_parity_portfolio, efficient_frontier, stability_test, stability_test_shrunk, out_of_sample_test, turnover_analysis, portfolio_stats
from plotting import plot_efficient_frontier

tickers = ["AAPL", "NVDA", "SPY", "JNJ", "GLD", "TSLA"]
returns_df = build_returns_matrix(tickers, period="5y")
mu, cov = annualised_stats(returns_df)
mu_arr, cov_arr = mu.values, cov.values

# 'optimal' portfolios
w_sharpe = max_sharpe_portfolio(mu_arr, cov_arr)
w_minvar = min_variance_portfolio(mu_arr, cov_arr)
w_rp     = risk_parity_portfolio(cov_arr)

ret, vol, sharpe = portfolio_stats(w_sharpe, mu_arr, cov_arr)

#plotting and visualising the efficient frontier
vols, rets, _ = efficient_frontier(mu_arr, cov_arr, n_points=50)
plot_efficient_frontier(mu_arr, cov_arr, tickers, vols, rets,
                        portfolios={'Max Sharpe': (w_sharpe, 'red', '*'),
                                    'Risk parity': (w_rp, 'blue', 'P')})

# Diagnostics
stability_test(returns_df, n_periods=4, allow_shorting=False)
out_of_sample_test(returns_df, train_frac=0.6)
turnover_analysis(returns_df, n_periods=4, cost_bps=10)

#weight cap feature
w_capped = max_sharpe_portfolio(mu_arr, cov_arr, max_weight=0.35)
```