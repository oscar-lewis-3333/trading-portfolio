import sys
import numpy as np
import pandas as pd

def generate_trading_decisions(tickers, lookback_col='return_21d', top_frac=0.25, rebalance_days=63, soft_limit=-0.10, hard_limit=-0.25, kelly_fraction=0.25, target_risk=0.02, max_position=0.25, period="10y"):

    #effectively combining the recommendations from ml_trading_signals and risk_management into one pipeline

    sys.path.append('../../ml_trading_signals/src')
    sys.path.append('../../risk_management/src')
    sys.path.append('../../time_series_forecasting/src')
    sys.path.append('../../past_market_analysis/src')

    from data_loader import fetch_price_data
    from position_sizing import kelly_from_returns, volatility_scaled_size
    from risk_limits import compute_drawdown, drawdown_trading_exposure

    #1: get price data, momentum rank.
    price_data = {}
    for t in tickers:
        df = fetch_price_data(t, period=period)
        price_data[t] = df['Close']
    prices = pd.DataFrame(price_data).dropna(how='all')
    returns = prices.pct_change()

    momentum = prices.pct_change(21).iloc[-1].dropna()
    if len(momentum) < 5:
        raise ValueError("Not enough tickers with valid momentum data")

    ranked = momentum.sort_values(ascending=False)
    n = max(1, int(len(momentum) * 0.25))
    selected = ranked.head(n)

    #2: position sizing - volatility scaling per asset. initialise final decisions dataframe with relative weights from the sizing
    vol_63d = returns.rolling(63).std().iloc[-1] * np.sqrt(252)
    sizes = {}
    for ticker in selected.index:
        asset_vol = vol_63d[ticker]
        sizes[ticker] = volatility_scaled_size(target_risk, asset_vol, max_size=max_position)

    decisions = []
    for ticker in selected.index:
        decisions.append({
            'ticker': ticker,
            'momentum_21': selected[ticker],
            'base_size': sizes[ticker],
            'current_price': prices[ticker].iloc[-1],
        })
    decisions_df = pd.DataFrame(decisions)

    raw_total = decisions_df['base_size'].sum() #choose to scale weights so that the sum of weight is 1. else have a situation where only 15% of capital invested, when this is the whole system
    if raw_total > 0:
        decisions_df['relative_weight'] = decisions_df['base_size'] / raw_total
    else:
        decisions_df['relative_weight'] = 0
    
    #3: applying circuit breaker. use equal weight universe as baseline (both are equity curves like risk_management)
    equal_weight_returns = returns[selected.index].mean(axis=1)
    equity = (1 + equal_weight_returns.dropna()).cumprod()
    drawdown = compute_drawdown(equity)
    exposure_series = drawdown_trading_exposure(drawdown, soft_limit, hard_limit)
    current_exposure = exposure_series[-1] if len(exposure_series) > 0 else 1.0

    #4: 'append' decisions with final sizes
    decisions_df['circuit_breaker_exposure'] = current_exposure
    decisions_df['final_size'] = decisions_df['relative_weight'] * current_exposure

    return decisions_df, current_exposure, drawdown.iloc[-1]
