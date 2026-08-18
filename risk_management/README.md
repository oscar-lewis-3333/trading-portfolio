# RISK MANAGEMENT & POSITION SIZING

## OVERVIEW
This project concerns itself with risk management. It's all well and good having a prediction strategy that is accurate, but how do we weight how confident we are? How do we know how much to invest in a trade? This project discusses position sizing approaches such Kelly sizing, volatility sizing, CVaR-constraining, and various risk mechanisms tested against an equity curve which implements the momentum strategy. Only one approach was found to actually improve performance - circuit breaker, which yielded excellent performance even when transaction costs modelled.

## FEATURES
- Sizing based upon kelly criterion (both on theoretical data and actual datqa)
- Volatility scaled sizing (using realised volatility and GARCH)
- Exposure circuit breaker using maximum drawdown (including soft/hard limits and linear scaling inbetween)
- VaR, CVaR computation, CVaR-constrainted sizing
- Combined position sizing (minimum across methods)
- Transaction cost modelling with stability analysis
- Volatility regime scaling (tested, found inferior)
- Correlation-based exposure scaling (tested, found not-applicable/useful for this strategy)
- Position-level triple-barrier stops (tested, found inferior)
- Final combined risk policy

## KEY DESIGN DECISIONS
- The Kelly-sizing approach is often seen as an 'aggressive' or 'optimistic' estimate, so quarter Kelly was implemented instead
- When created a composite size, the aim is to satisfy all criterion (CVaR constrained, Kelly and volatilty), so a minimum of these must be taken
- Chose to use fold returns for Kelly sizing rather than the theoretical values since the theoretical value implied each loss was the same size as each win, which was not true on this fold at all (~2.4).
- Linear scaling between upper/lower thresholds chosen in circuit breaker system. Quadratic/polynomial order and even exponential scaling possible, but often decreases exposure too quickly when approaching the hard-threshold, but still being considerably far away

## RESULTS

### Kelly sizing

| Method | Value |
|---|---|
| Theoretical Kelly (p=0.6571, b=1.0) | 0.3142 |
| Quarter Kelly (theoretical) | 0.0786 |
| Half Kelly (theoretical) | 0.1571 |
| Kelly from actual fold returns (full) | 0.4177 |
| Recommended (quarter Kelly, actual) | 0.1044 |

### Volatility-scaled sizing

| Method | Position size |
|---|---|
| Trailing 63-day realised vol | 0.0633 |
| GARCH day-1 forecast | 0.0745 |

### Drawdown circuit breaker

| Metric | Value |
|---|---|
| Percentage of time in reduced-exposure state | 36.85% |
| Raw final equity | 14.04 |
| Managed final equity | 27.88 |

### Transaction cost sensitivity

| Cost (bps) | Final equity |
|---|---|
| 5 | 27.48 |
| 10 | 27.10 |
| 25 | 25.97 |
| 50 | 24.19 |
| 100 | 20.99 |

### Volatility regime scaling

| Metric | Value |
|---|---|
| Days in high-vol regime | 627 / 2491 |
| Days with reduced regime exposure | 2222 |
| Regime-managed max drawdown | -23.30% |
| Regime-managed final equity | 7.33 |


### Summary — all methods, 63-day rebalance basis

| Method | Final equity | Max drawdown |
|---|---|---|
| Raw (no risk management) | 14.04 | -40.09% |
| Drawdown circuit breaker | 27.88 | -15.76% |
| Volatility regime scaling | 7.33 | -23.30% |
| Position-level stops (63d) | 2.77 | -21.19% |

## KEY FINDINGS
- There was significant difference between the theoretical Kelly and actual Kelly. This was due to a difference in predicted 'win' values. In theory, we always assume that each win is the same size as a loss (coinflip). In our momentum strategy, we encountered an approximate 2.4x payoff asymmetry, meaning that each 'up' is approximately 2.4x each 'down' (in their respective directions).
- Across the 4 risk mechanisms tested in this project, only the circuit breaker approach, where we linearly decrease our exposure to the market within soft/hard limits based upon the current drawdown, improved returns and reduced the total maximum drawdown. The raw momentum strategy on our equity curve returned $14.04 from 1$ over 10 years, whereas the circuit breaker approach returned $27.88 without transaction costs and $27.10 when 0.1% transaction fees are applied per rebalance, doubling our returns over the same time-frame. This approach also brought the max drawdown from ~ -40.1% to -15.76%, over halving the difference between peaks and troughs. The other approaches were not nearly as successful, with regime volatility scaling, returns went from $14.04 to $7.33, almost halving our returns, and taking the max drawdown to -23.30%, which is still beaten by the circuit breaker. This is not of huge surprise, as volatility is a price neutral metric, hence when the market is spiraling upwards, this mechanism scales back due to the high volatility. For the position-level stop mechanism, returns went from the stated $14.04 momentum raw strategy, to $2.77 under this strategy, and max drawdown down to -21.19%, still beaten by the circuit breaker. This could be due to the rebalancing period being too long for an approach like this. When we reach the profit/stop boundar, we cease trading until the next rebalance (in this case, next quarter), so hypothetically if the strategy performs badly for the first 2 weeks of a quarter, but responds well afterwards, this mechanism has already ceased trading, so loses out. If the rebalancing was reduced to 1-4 weeks instead of quarterly, intuitively this approach would perform significantly better than it does currently. This is a line of exploration for the future, and would require significant statistical testing before it became reliable.
- Correlation exposure scaling became unnecessary as the 'baskets' of assets selected (top 25% of given universe when judged by momentum), already averaged a momentum of ~0, and a maximum of 0.2 over a 10 year period. This meant the effects of this method would be very limited, and implies the momentum method already selects assets with minimal correlation automatically 
- When modelling transaction costs and applying to the circuit breakers, effects were stable. As expected, performance worsened as transaction costs were varied from 0.05% to 1% per rebalance, but it was stable and even with 1% costs per rebalance, performance of the circuit breaker approach outperformed the raw method by ~50%

## LIMITATIONS
- Project only really test the momentum strategy from ml_trading_signals. A wide range of strategies have not been tested with these risk management approaches, so findings may change based on strategy.
- Regime method only used realised volatility - a price neutral metric, hence its sub-par performance is not totally unexpected. Possible if used in conjunction with another classifier which isn't price neutral, then performance may improve
- Chose circuit breaker limits based on intuition surrounding drawdown range of the portfolio instead of being optimised, hence there is a possibility that there are more effective thresholds which could improve performance. Alternatively, these thresholds may perform better on this equity curve than other curves due to in-sample bias.

## LIBRARIES USED
- numpy, pandas — position sizing calculations, equity curve construction
- scipy — statistical significance testing (t-test)
- matplotlib — max-drawdown, exposure, equity curves, comparison of methods

## PROJECT STRUCTURE
    src/
        position_sizing.py — volatility and Kelly sizing methods
        risk_limits.py      — drawdown circuit breaker, VaR/CVaR, modelling transaction costs, volatility regime, correlation limiting exposure, triple-barrier position-level stops
        plotting.py          — plotting various risk management approaches
    notebooks/
        risk_management.ipynb — main analysis

## USAGE
```python
from position_sizing import kelly_sizing, kelly_from_returns, volatility_scaled_size
from risk_limits import compute_drawdown, drawdown_trading_exposure, compute_var,compute_cvar, cvar_constrained_size, combined_position_size, estimate_momentum_turnover, classify_regime, regime_scaled_exposure, correlation_penalty, apply_position_stops
from plotting import plot_equity_drawdown_exposure, plot_position_sizing_comparison, plot_three_way_comparison, plot_all_risk_methods

#position sizing 
f_kelly = kelly_from_returns(mom_wf['excess'], fraction=0.25)
size_vol = volatility_scaled_size(target_risk=0.02, asset_vol=garch_vol)
size_cvar = cvar_constrained_size(returns, cvar_budget=-0.03)
base_size = combined_position_size(f_kelly, size_vol, size_cvar)

#circuit breaker management scheme
drawdown = compute_drawdown(equity_curve)
exposure = drawdown_trading_exposure(drawdown, soft_limit=-0.10, hard_limit=-0.25)
managed_returns = portfolio_returns * exposure
plot_equity_drawdown_exposure(raw_equity, managed_equity, drawdown, exposure)

#estimating transaction costs
turnover = estimate_momentum_turnover(pooled_v2, rebalance_days=63)

#position stops (stop trading when hit profit/stop multiplier)
stopped_equity = apply_position_stops(tickers, stop_mult=1.0, profit_mult=2.0, rebalance_days=10)

#comparing all methods tested in notebook
plot_all_risk_methods(raw_equity, drawdown_managed, regime_managed, stopped_equity)
```