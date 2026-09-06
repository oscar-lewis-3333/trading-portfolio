# RISK MANAGEMENT & POSITION SIZING

## OVERVIEW
This project concerns itself with risk management. It's all well and good having a prediction strategy that is accurate, but how do we weight how confident we are? How do we know how much to invest in a trade? This project discusses position sizing approaches such Kelly sizing, volatility sizing, CVaR-constraining, and various risk mechanisms tested against an equity curve which implements the 63 day momentum strategy from [Machine Learning Trading Signals](./ml_trading_signals/). 

## FEATURES
- Sizing based upon kelly criterion (both on theoretical data and actual data)
- Volatility scaled sizing (using realised volatility and GARCH)
- Exposure circuit breaker using maximum drawdown (including soft/hard limits and linear scaling inbetween, found inferior)
- VaR, CVaR computation, CVaR-constrainted sizing
- Combined position sizing (minimum across methods)
- Transaction cost modelling with stability analysis
- Volatility regime scaling (tested, found inferior)
- Correlation-based exposure scaling (tested, found not-applicable/useful for this strategy)
- Position-level triple-barrier stops (tested, found inferior)
- 11 test functions to ensure implementation of various approaches are correct

## KEY DESIGN DECISIONS
- The Kelly-sizing approach is often seen as an 'aggressive' or 'optimistic' estimate, so quarter Kelly was implemented instead
- When created a composite size, the aim is to satisfy all criterion (CVaR constrained, Kelly and volatilty), so a minimum of these must be taken
- Linear scaling between upper/lower thresholds chosen in circuit breaker system. Quadratic/polynomial order and even exponential scaling possible, but often decreases exposure too quickly when approaching the hard-threshold, but still being considerably far away

## RESULTS

### Kelly sizing from historical strategy returns

| Metric | Value |
|---|---:|
| Return observations | 2448 |
| Non-zero returns | 2448 |
| Winning days | 1362 |
| Losing days | 1086 |
| Historical win probability | 0.5564 |
| Historical win/loss ratio | 0.9843 |
| Mean win | 1.4298% |
| Mean loss | -1.4527% |
| Full Kelly estimate | 0.1057 |
| Half Kelly | 0.0528 |
| Quarter Kelly | 0.0264 |

### Volatility-scaled sizing

| Volatility estimate | Estimated volatility | Position size |
|---|---:|---:|
| Trailing 63-day realised volatility | 0.3799 | 0.0526 |
| GARCH one-day forecast | 0.1859 | 0.1076 |

### Drawdown overlay performance

| Metric | Value |
|---|---:|
| Time in reduced-exposure state | 39.32% |
| Raw maximum drawdown | -43.7003% |
| Managed maximum drawdown | -28.5043% |
| Raw final equity | 24.28 |
| Managed final equity | 10.66 |

### VaR and CVaR risk estimates

| Metric | Value |
|---|---:|
| VaR (95%) | -3.0425% |
| CVaR (95%) | -4.7130% |
| CVaR/VaR ratio | 1.549 |
| CVaR-constrained size (3% budget) | 0.6365 |

### Combined position-sizing constraint

| Sizing method | Position size |
|---|---:|
| Quarter Kelly | 0.0264 |
| Volatility-scaled | 0.1076 |
| CVaR-constrained | 0.6365 |
| Final recommended size (minimum) | 0.0264 |

### Drawdown-overlay turnover and 10-bps costs

| Metric | Value |
|---|---:|
| Mean turnover when trading occurred | 11.83% |
| Turnover events | 764 |
| Managed final equity, gross | 10.66 |
| Managed final equity, net of 10-bps costs | 9.74 |

### Transaction-cost sensitivity

| Cost | Final equity |
|---|---:|
| 5 bps | 10.19 |
| 10 bps | 9.74 |
| 25 bps | 8.50 |
| 50 bps | 6.78 |
| 100 bps | 4.31 |

### Volatility-regime overlay performance

| Metric | Value |
|---|---:|
| Days at or above high-volatility threshold | 648 |
| Days with reduced applied exposure | 2252 |
| Regime-managed maximum drawdown | -31.33% |
| Regime-managed final equity | 5.25 |

### Correlation-exposure diagnostic

| Metric | Value |
|---|---:|
| Signal date | 2026-08-17 |
| Selected basket | JPM, AMD, JNJ, HD |
| Average pairwise correlation | -0.0294 |
| Correlation-adjusted exposure scale | 1.0000 |

### Position-barrier performance

| Metric | Value |
|---|---:|
| Raw final equity | 24.28 |
| Raw maximum drawdown | -43.70% |
| Position-barrier final equity | 1.32 |
| Position-barrier maximum drawdown | -19.80% |
| Stop-loss exits | 299 |
| Profit-target exits | 164 |
| Periods containing cash | 2384 |

### All risk methods — 63-day momentum with 21-day rebalancing

| Method | Start | End | Obs. | Final equity | CAGR (%) | Ann. vol. (%) | Max DD (%) | Sharpe | Calmar | Avg. exposure (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw momentum | 2016-12-06 | 2026-09-03 | 2449 | 24.283 | 38.742 | 32.257 | -43.700 | 1.180 | 0.887 | 100.000 |
| Drawdown overlay | 2016-12-06 | 2026-09-03 | 2449 | 10.657 | 27.494 | 26.601 | -28.504 | 1.049 | 0.965 | 75.147 |
| Volatility overlay | 2016-12-06 | 2026-09-03 | 2449 | 5.249 | 18.555 | 15.705 | -31.327 | 1.165 | 0.592 | 55.798 |
| Position barriers | 2016-12-06 | 2026-09-03 | 2449 | 1.324 | 2.920 | 9.324 | -19.804 | 0.356 | 0.147 | 13.304 |

## KEY FINDINGS
- No risk management approach was effective when used in conjunction with the momentum strategy. This does not mean they are not useful when used alongside other approaches, but when tested with this 63d momentum strategy, none were effective. 
- The maximum sharpe ratio was achieved by the raw strategy, although the volatility overlay came close despite having ~20% of the final equity. Said volatility overlay's final equity was ~$5.25, and its poor performance on this front could be explained due to volatility being a price-neutral metric. It is possible that the performance of this method could be improved if it was used in conjunction with a metric which identifies a well-performing stocks from a badly-performing one, but this is left for future work.
- Circuit breaker approached reduced final equity from ~$24 to ~$10 even without costs, with it reducing to ~$8.50 with 0.25% transaction costs per rebalance. The max drawdown was however reduced from -43.7% to ~-28.5%, which may appeal to cautious investors.
- Position stops method was massively unsuccessful, due to trading being ceased often during strong periods for the strategy. The approach may perform better if it was applied over shorter time-frames, but this is left for future projects.
- Correlation exposure scaling became unnecessary as the 'baskets' of assets selected (top 20% of given universe when judged by 63d momentum), already averaged a momentum of ~0, and a maximum of 0.2 over a 10 year period. This meant the effects of this method would be very limited, and implies the momentum method already selects assets with minimal correlation automatically.
- When modelling transaction costs and applying to the circuit breakers, effects were stable. As expected, performance worsened as transaction costs were varied from 0.05% to 1% per rebalance, but stability was shown despite poor results.

## LIMITATIONS
- Project only really tests the momentum strategy from ml_trading_signals. A wide range of strategies have not been tested with these risk management approaches, so findings may change based on strategy. This is to be tested in future projects surrounding different strategies on different universes.
- Regime method only used realised volatility - a price neutral metric, hence its sub-par performance is not totally unexpected. Possible if used in conjunction with another classifier which isn't price neutral, then performance may improve.
- Chose circuit breaker limits based on intuition surrounding drawdown range of the portfolio instead of being optimised, hence there is a possibility that there are more effective thresholds which could improve performance. Alternatively, these thresholds may perform better on this equity curve than other curves due to in-sample bias.

## LIBRARIES USED
- numpy, pandas — position sizing calculations, equity curve construction
- matplotlib — max-drawdown, exposure, equity curves, comparison of methods

## PROJECT STRUCTURE
    src/
        position_sizing.py — volatility and Kelly sizing methods
        risk_limits.py — drawdown and volatility overlays, VaR/CVaR, transaction costs, correlation exposure and position barriers
        performance_evaluation.py — return, exposure and performance comparison tables
        plotting_risk_management.py — visual comparisons of sizing and risk-management methods
    notebooks/
        risk_management.ipynb — main analysis
    tests/
        test_drawdown_overlay.py — test that the drawdown overlay acts as intended (day t drawdown changes only day t+1 exposure)
        test_position_stops.py — test that the position profit/stop check acts as it's supposed to
        test_transaction_costs.py — test that transaction costs only apply at rebalances (and apply correctly)
        test_volatility_overlay.py — ensure that volatility works as intended (same principle as drawdown)

## USAGE
```python
from features import build_multi_ticker_dataset, build_momentum_equity_curve
from labels import excess_return_target
from position_sizing import kelly_from_returns, volatility_scaled_size
from risk_limits import apply_drawdown_overlay, apply_transaction_costs, apply_position_stops, apply_volatility_overlay, combine_turnover_overlay, cvar_constrained_size, combined_position_size
from plotting_risk_management import plot_equity_drawdown_exposure, plot_position_sizing_comparison, plot_three_way_comparison, plot_all_risk_methods
from performance_evaluation import build_returns_by_method, build_exposure_by_method, compare_risk_methods

#build pooled data using same strategy mentioned throughout
pooled = build_multi_ticker_dataset(tickers, period="10y", horizon=21, ranking_horizon=21)
pooled_v2 = excess_return_target(pooled)
daily_equity = build_momentum_equity_curve(pooled_v2, lookback_col='return_63d', top_frac=0.2, rebalance_days=21)
strategy_returns = daily_equity.loc[~daily_equity['is_initial'], 'portfolio_return'].dropna().astype(float)

#position sizing 
f_kelly = kelly_from_returns(strategy_returns, frac=0.25)
trailing_vol = strategy_returns.rolling(63).std().iloc[-1] * (252 ** 0.5)
size_vol = volatility_scaled_size(target_risk=0.02, asset_vol=trailing_vol)
size_cvar = cvar_constrained_size(strategy_returns, cvar_budget=-0.03)
base_size = combined_position_size(kelly_size=f_kelly, vol_scaled_size=size_vol, cvar_size=size_cvar)
plot_position_sizing_comparison(f_kelly, size_vol, size_cvar, base_size)

#circuit breaker management scheme
drawdown_result = apply_drawdown_overlay(daily_equity['portfolio_return'], soft_limit=-0.10, hard_limit=-0.25, initial_capital=1.0)
drawdown = drawdown_result['reference_drawdown']
exposure = drawdown_result['exposure']
managed_returns = drawdown_result['managed_return']
managed_equity = drawdown_result['managed_equity']
plot_equity_drawdown_exposure(daily_equity, managed_equity, drawdown, exposure)

#estimating transaction costs
turnover_breakdown = combine_turnover_overlay(base_turnover=daily_equity['turnover'], exposure=exposure)
cost_result = apply_transaction_costs(returns=managed_returns, turnover=turnover_breakdown['total_turnover'], cost_bps=10, initial_capital=1.0)

#volatility regime management scheme
volatility_result = apply_volatility_overlay(daily_equity['portfolio_return'],vol_window=21, low_vol_threshold=0.15, high_vol_threshold=0.35, min_exposure=0.20, initial_capital=1.0)
regime_managed_returns = volatility_result['managed_return']
regime_managed_equity = volatility_result['managed_equity']

#position stops (stop trading when hit profit/stop multiplier)
stopped_equity, stop_events = apply_position_stops(pooled_v2, lookback_col='return_63d', top_frac=0.2, rebalance_days=21, stop_mult=1.0, profit_mult=2.0, min_universe=5, initial_capital=1.0)

#comparing all methods tested in notebook
plot_all_risk_methods(daily_equity['equity'], managed_equity, regime_managed_equity, stopped_equity['equity'])

#performance comparison table
returns_by_method = build_returns_by_method(raw_returns=daily_equity['portfolio_return'], drawdown_returns=managed_returns, volatility_returns=regime_managed_returns, barrier_returns=stopped_equity['portfolio_return'])
exposure_by_method = build_exposure_by_method(returns_by_method=returns_by_method, drawdown_exposure=exposure, volatility_exposure=volatility_result['exposure'], barrier_cash_weight=stopped_equity['cash_weight'])
performance_table = compare_risk_methods(returns_by_method, exposure_by_method)
```