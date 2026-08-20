import numpy as np
import pandas as pd

def compute_drawdown(equity_curve):
    #compute drawdown series from equity curve - how far below running peak value stock sits at each time

    equity_curve = pd.Series(equity_curve)
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak
    return drawdown

def drawdown_trading_exposure(drawdown, soft_limit, hard_limit):
    #as drawdown increases we limit exposure. above soft limit, full exposure, between hard and soft limit, linearly scaled down, hard_limit - stop trading (no exposure)
    #circuit breaker, not prediction. make assumption that trading through a bad patch is bad. stops the process of turning a manageable loss to a really bad one

    drawdown = np.asarray(drawdown)
    exposure = np.ones_like(drawdown, dtype=float)

    in_between = (drawdown < soft_limit) & (drawdown >= hard_limit)
    scale = (drawdown[in_between] - hard_limit) / (soft_limit - hard_limit)
    exposure[in_between] = scale

    exposure[drawdown < hard_limit] = 0.0

    return exposure

def compute_var(returns, confidence=0.95):

    #value at risk - loss threshold that wont be exceeded with given confidence.
    #if VaR(0.95)= -0.03, then on 95% of days we wont lose more than 3%. 

    return np.percentile(returns.dropna(), (1 - confidence)*100)

def compute_cvar(returns, confidence=0.95):
    #compute conditional VaR (expected shortfall) - given we're in the (1-confidence)'th percentile, what is the expected loss?

    returns = returns.dropna()
    var = compute_var(returns, confidence=confidence)
    losses = returns[returns <= var]
    return losses.mean() if len(losses) > 0 else var

def cvar_constrained_size(returns, cvar_budget=-0.05, confidence=0.95, max_size=1.0):
    #aim to size a position so that the cvar is never exceeded, where cvar_budget is max accepted loss

    asset_cvar = compute_cvar(returns, confidence)
    if asset_cvar >= 0: #if no losses, then always max size
        return max_size

    size = cvar_budget/asset_cvar

    return min(size, max_size)

def combined_position_size(kelly_size, vol_scaled_size, cvar_size):
    #combine all positions across the two computational src .py files. For a trade, we aim to satisfy all 3 position criterion simultaenously

    return min(kelly_size, vol_scaled_size, cvar_size)

def apply_transaction_costs(returns, turnover, cost_bps=10):

    #apply transactions costs to a series, turnover gives the fraction of the portfolio being traded at that point, cost_bps in basis points (100bps = 1%)
    #we only deduct costs when rebalancing occurs (~every quarter)

    returns = returns.copy()
    fraction_cost = cost_bps/10000
    cost_drag = turnover * fraction_cost
    return returns - cost_drag

def estimate_momentum_turnover(pooled_df, lookback_col='return_21d', top_frac=0.25, rebalance_days=63, min_universe=5):

    #estimate turnover at each rebalance
    
    clean = pooled_df.dropna(subset = [lookback_col])
    all_dates = sorted(clean.index.unique())

    prev_selection = None
    turnover_events = []

    for i, date in enumerate(all_dates):
        if i % rebalance_days !=0:
            continue

        day = clean.loc[clean.index == date]
        if len(day) < min_universe:
            continue

        ranked = day.sort_values(lookback_col, ascending=False)
        n = max(1, int(top_frac * len(day)))
        current_selection = set(ranked.head(n)['ticker'])

        if prev_selection is not None:
            changed = len(current_selection - prev_selection)
            turnover_events.append({
                'date': date,
                'turnover': changed / len(current_selection)
            })

        prev_selection = current_selection

    return pd.DataFrame(turnover_events).set_index('date')

def classify_regime(returns, vol_window=21, vol_threshold_high=0.30, dispersion_window=21):
    #we classify high-volatility regimes, which will be a new way to size positions. Aim to compare to other approach above.

    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
    regime = pd.Series('normal', index=returns.index)
    regime[rolling_vol > vol_threshold_high] = 'high_vol'
    return regime, rolling_vol

def regime_scaled_exposure(rolling_vol, low_vol_threshold=0.15, high_vol_threshold=0.35, min_exposure=0.2):
    #we try a way of limiting exposure in high volatility periods. Compare to other approach later. Scale exposure linearly based on volatility with a minimum

    exposure = np.ones(len(rolling_vol))
    vol = rolling_vol.values

    scale_zone = (vol > low_vol_threshold) & (vol < high_vol_threshold)
    scale = 1 - (vol[scale_zone] - low_vol_threshold) / (high_vol_threshold - low_vol_threshold)

    exposure[scale_zone] = (1 - min_exposure) * scale + min_exposure
    exposure[vol >= high_vol_threshold] = min_exposure
    exposure[np.isnan(vol)] = 1.0

    return pd.Series(exposure, index=rolling_vol.index)


def correlation_penalty(pooled_df, selection, lookback_col='return_1d', window=63, max_penalty=0.5):

    #measure pairwise correlation between assets. scale down exposure when assets are highly correlated. limit at 50% by standard

    returns_wide = pooled_df[pooled_df['ticker'].isin(selection)].pivot_table(index=pooled_df.index.name or 'index', columns='ticker', values=lookback_col)
    recent = returns_wide.tail(window)
    corr_matrix = recent.corr()

    n = len(selection)
    upper_tri = corr_matrix.values[np.triu_indices(n, k=1)] #upper triangle gives all corr values without the 1 diagonal (lower tri also does this)
    avg_corr = np.nanmean(upper_tri)

    penalty_scale = 1 - (max_penalty * max(0, avg_corr))
    return penalty_scale, avg_corr

def position_stop_check(entry_price, current_price, entry_vol, stop_mult=1.0, profit_mult=2.0):
    #same as triple barriers from ml_trading_signals
    upper = entry_price * (1 + profit_mult * entry_vol)
    lower = entry_price * (1 - stop_mult * entry_vol)

    if current_price >= upper:
        return 'profit'
    elif current_price <= lower:
        return 'stop'
    else:
        return 'hold'

def apply_position_stops(tickers, lookback_col='return_21d', period="10y", top_frac=0.25, rebalance_days=21, profit_mult=0.2, stop_mult=1.0, min_universe=5, vol_window=21):

    #simulating momentum strategy with above 'labels' applied. if profit/stop free capital to sit until next quarter

    import sys
    from data_loader import fetch_price_data

    price_data = {}
    for t in tickers:
        df = fetch_price_data(t, period=period)
        price_data[t] = df['Close']

    prices = pd.DataFrame(price_data).dropna(how='all')
    returns = prices.pct_change()
    vol = returns.rolling(vol_window).std()

    momentum = prices.pct_change(21)

    all_dates = prices.index
    daily_rows = []
    positions = {}

    for i, date in enumerate(all_dates):
        if i < vol_window:
            continue

        if i % rebalance_days == 0:
            today_mom = momentum.loc[date].dropna()

            if len(today_mom) < min_universe:
                positions = {}
            else:
                ranked = today_mom.sort_values(ascending=False)
                n = max(1, int(len(ranked) * top_frac))
                selected = ranked.head(n).index

                positions = {}
                for t in selected:
                    positions[t] = {
                        'entry_price': prices.loc[date, t],
                        'entry_vol': vol.loc[date, t]
                        if not np.isnan(vol.loc[date, t]) else 0.02
                    }
        
        day_positions_returns = []
        for t in list(positions.keys()):
            pos = positions[t]
            current_price = prices.loc[date, t]
            if np.isnan(current_price):
                continue
            outcome = position_stop_check(pos['entry_price'], current_price, pos['entry_vol'], stop_mult, profit_mult)
            day_positions_returns.append(returns.loc[date, t] if not np.isnan(returns.loc[date, t]) else 0.0)

            if outcome in ['profit', 'stop']:
                del positions[t]

        portfolio_return = np.mean(day_positions_returns) if day_positions_returns else 0.0
        daily_rows.append({
            'date': date,
            'portfolio_return': portfolio_return
        })

    daily_df = pd.DataFrame(daily_rows).set_index('date').sort_index()
    daily_df['equity'] = (1 + daily_df['portfolio_return']).cumprod()
    return daily_df

def final_risk_policy(base_size, drawdown_exposure):
    #after analysis in the notebook, we give our final risk policy - size positions by kelly/cvar/vol-scaling, then scale with circuit breaker

    return base_size * drawdown_exposure
