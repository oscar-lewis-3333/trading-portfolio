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

def apply_transaction_costs(returns, turnover, cost_bps=10, initial_capital=1.0):

    #apply transaction costs on intervals where rebalancing occurs, with turnover being expressed as a fraction of portfolio value
    #cost_bps in basis points

    if not isinstance(returns, pd.Series) or not isinstance(turnover, pd.Series):
        raise TypeError("Returns and turnover must be pandas series'")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Transaction costs must be finite and non-negative")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be positive and finite")
    

    returns = pd.to_numeric(returns.copy(), errors='raise').astype(float)
    turnover = pd.to_numeric(turnover.copy(), errors='raise').astype(float)

    if returns.empty:
        raise ValueError("Returns cannot be empty")
    if not returns.index.equals(turnover.index):
        raise ValueError("Returns and turnover must have identical indexes")
    if returns.index.has_duplicates:
        raise ValueError("Return dates cannot be duplicated")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Returns must be sorted chronologically")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns must be finite")
    if not np.isfinite(turnover.to_numpy()).all():
        raise ValueError("Turnover must be finite")
    if (returns < -1).any():
        raise ValueError("Returns cannot be below -100%")
    if (turnover < 0).any():
        raise ValueError("Turnover cannot be negative")
    
    cost_rate = cost_bps/10000
    cost_frac = turnover * cost_rate

    if (cost_frac >= 1).any():
        raise ValueError("Transaction costs cannot consume the entire portfolio")

    gross_growth = 1 + returns
    net_growth = (1 - cost_frac) * gross_growth
    net_return = net_growth - 1

    gross_equity = initial_capital * gross_growth.cumprod()
    net_equity = initial_capital * net_growth.cumprod()

    return pd.DataFrame({
        'gross_return': returns,
        'turnover': turnover,
        'cost_fraction': cost_frac,
        'net_return': net_return,
        'gross_equity': gross_equity,
        'net_equity': net_equity,
    })


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


def correlation_penalty(pooled_df, selection, signal_date, lookback_col='return_1d', window=63, min_obs=20, max_penalty=0.5):

    #measure pairwise correlation between assets. scale down exposure when assets are highly correlated. limit at 50% by standard. use only returns available by signal date

    required = {'ticker', lookback_col}
    missing = required.difference(pooled_df.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    forbidden = {
        'mom_return_1d',
        'ranking_return',
        'fwd_return',
        'label',
        'excess_return',
    }

    if lookback_col in forbidden:
        raise ValueError("Correlation cannot use a future-looking column")

    selected = list(selection)
    if len(selected) < 2 or len(selected) != len(set(selected)):
        raise ValueError("Select at least two unique tickers")

    if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or window < 2:
        raise ValueError("Window must be an integer and at least 2")

    if not 2 <= min_obs <= window:
        raise ValueError("Minimum observations must be between 2 and the window")

    if not np.isfinite(max_penalty) or not 0 <= max_penalty <= 1:
        raise ValueError("Maximum penalty must be between 0 and 1 (and hence finite)")

    dates = pd.to_datetime(pooled_df.index, errors='raise')
    signal_date = pd.Timestamp(signal_date)

    if not dates.is_monotonic_increasing:
        raise ValueError("Pooled data must be chronological")
    if not (dates == signal_date).any():
        raise ValueError("Signal date is not present in pooled data")

    date_ticker = pd.MultiIndex.from_arrays([dates, pooled_df['ticker']])
    if date_ticker.has_duplicates:
        raise ValueError("Duplicate date-ticker observations found")

    history = pooled_df[['ticker', lookback_col]].copy()
    history['_date'] = dates
    history[lookback_col] = pd.to_numeric(history[lookback_col], errors='raise')
    history = history.loc[(history['_date'] <= signal_date) & history['ticker'].isin(selected)]

    returns_wide = history.pivot(index='_date', columns='ticker', values=lookback_col).reindex(columns = sorted(selected)).tail(window)
    counts = returns_wide.notna().sum()

    if (counts < min_obs).any():
        raise ValueError(f"Insufficient History: {counts.to_dict()}")

    corr_matrix= returns_wide.corr(min_periods=min_obs)
    n = len(selected)
    upper = np.triu_indices(n, k=1)
    pairwise = corr_matrix.to_numpy()[upper]

    if not np.isfinite(pairwise).all():
        raise ValueError("Pairwise correlations could not be calculated")

    avg_corr = float(np.clip(pairwise.mean(), -1, 1))
    penalty_scale = 1 - (max_penalty * max(0, avg_corr))
    return penalty_scale, avg_corr

def position_stop_exit(entry_price, entry_vol, day_open, day_high, day_low, next_open, stop_mult=1.0, profit_mult=2.0):

    values = np.asarray([entry_price,
        entry_vol,
        day_open,
        day_high,
        day_low,
        next_open,
        stop_mult,
        profit_mult], dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("All inputs must be finite")
    if min(entry_price, day_open, day_high, day_low, next_open) <= 0:
        raise ValueError("Prices must be positive")
    if entry_vol <= 0 or stop_mult <= 0 or profit_mult <= 0:
        raise ValueError("Volatility and barrier multipliers must be positive")
    if not day_low <= day_open <= day_high:
        raise ValueError("Daily OHLC values are inconsistent, upper bound must be day_high, lower bound must be day_low")

    upper = entry_price * (1 + profit_mult*entry_vol)
    lower = entry_price * (1 - stop_mult*entry_vol)

    if day_open <= lower:
        return day_open, 'stop', 'open'
    if day_open >= upper:
        return day_open, 'profit', 'open'

    upper_hit = day_high >= upper
    lower_hit = day_low <= lower

    if upper_hit and lower_hit:
        return lower, 'stop', 'intraday' #if both hit, daily data not sufficient to distinguish which one was hit first, so take the safe side of stop
    if lower_hit:
        return lower, 'stop', 'intraday'
    if upper_hit:
        return upper, 'profit', 'intraday'

    if next_open <= lower:
        return next_open, 'stop', 'next_open'
    if next_open >= upper:
        return next_open, 'profit', 'next_open'

    return None, 'hold', None

def apply_position_stops(pooled_df, lookback_col='return_63d', top_frac=0.2, rebalance_days=21, profit_mult=2.0, stop_mult=1.0, min_universe=5, initial_capital=1.0):
    #rank at close t, trade at open t+1 and value at open t+2
    #proceeds from barrier exits remain cash until scheduled rebalance (next quarter, so expect long-term well-performing stocks to perform poorly long term as no equity until next quarter)

    required = {
        'ticker',
        lookback_col,
        'stop_vol',
        'mom_entry_date',
        'mom_exit_date',
        'mom_entry_open',
        'mom_high_1d',
        'mom_low_1d',
        'mom_exit_open'
    }
    missing = required.difference(pooled_df.columns)

    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")
    if pooled_df.empty:
        raise ValueError("Pooled data cannot be empty")
    if not 0 < top_frac <= 1:
        raise ValueError("Top fraction must be greater than 0 and at most 1")
    if not isinstance(rebalance_days, (int, np.integer)) or isinstance(rebalance_days, bool) or rebalance_days < 1:
        raise ValueError("Rebalance days must be a positive integer")
    if not isinstance(min_universe, (int, np.integer)) or isinstance(min_universe, bool) or min_universe < 2:
        raise ValueError("Minimum universe must be an integer of at least 2")

    parameters = np.asarray([
        profit_mult,
        stop_mult,
        initial_capital], dtype=float)

    if not np.isfinite(parameters).all():
        raise ValueError("Parameters must be finite")
    if profit_mult <= 0 or stop_mult <= 0:
        raise ValueError("Barrier multipliers must be positive")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive")

    data = pooled_df.copy()
    data.index = pd.to_datetime(data.index, errors='raise')
    data = data.sort_index()

    data['mom_entry_date'] = pd.to_datetime(data['mom_entry_date'], errors='coerce')
    data['mom_exit_date'] = pd.to_datetime(data['mom_exit_date'], errors='coerce')

    numeric_cols = [
        lookback_col,
        'stop_vol',
        'mom_entry_open',
        'mom_high_1d',
        'mom_low_1d',
        'mom_exit_open']
    
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='raise')

    if data['ticker'].isna().any():
        raise ValueError("Ticker values cannot be missing")

    date_ticker = pd.MultiIndex.from_arrays([data.index, data['ticker']])
    if date_ticker.has_duplicates:
        raise ValueError("Duplicate date-ticker observations found")

    signal_dates = data.index.unique().sort_values()
    if len(signal_dates) < 3:
        raise ValueError("At least three trading days are required")

    positions = {}
    cash = float(initial_capital)
    started = False
    since_rebalance = 0
    rows = []
    events = []

    for date_i in range(len(signal_dates) - 2): #as said at start, need 3 days to correctly rank, trade and value.
        signal_date = signal_dates[date_i]
        entry_date = signal_dates[date_i + 1]
        exit_date = signal_dates[date_i + 2]

        day = data.loc[data.index == signal_date]
        day_by_ticker = day.set_index('ticker', drop=False)

        def ticker_row(ticker): #get information for that ticker on that day, and ensure dates align. raise value errors if not
            if ticker not in day_by_ticker.index:
                raise ValueError(f"Missing {ticker} on {signal_date.date()}")

            row = day_by_ticker.loc[ticker]

            dates_aligned = row['mom_entry_date'] == entry_date and row['mom_exit_date'] == exit_date
            if not dates_aligned:
                raise ValueError(f"Dates are misaligned for {ticker} after "f"{signal_date.date()}")
            return row

        current_values = {}
        for ticker, position in positions.items():
            row = ticker_row(ticker)
            current_open = float(row['mom_entry_open'])

            if not np.isfinite(current_open) or current_open <= 0:
                raise ValueError(f"Invalid opening price for {ticker}")

            current_values[ticker] = position['shares'] * current_open

        value_before_rebalance = cash + sum(current_values.values())

        if (not np.isfinite(value_before_rebalance) or value_before_rebalance <= 0):
            raise ValueError("Portfolio value must remain positive and finite")

        rebalance_due = (not started or since_rebalance >= rebalance_days)
        rebalanced = False
        rebalance_turnover = 0.0

        if rebalance_due:
            eligible_mask = (np.isfinite(day[lookback_col]) & np.isfinite(day['stop_vol']) & day['stop_vol'].gt(0))
            eligible = day.loc[eligible_mask, ['ticker', lookback_col]]

            if len(eligible) < min_universe:
                if not started:
                    continue
                raise ValueError(f"Only {len(eligible)} eligible stocks on " f"{signal_date.date()}")

            ranked = eligible.sort_values([lookback_col, 'ticker'], ascending=[False, True],)

            n = max(1, int(len(ranked) * top_frac))
            selected = tuple(ranked.head(n)['ticker'])
            for ticker in selected:
                row = ticker_row(ticker)
                price_values = row[[
                        'mom_entry_open',
                        'mom_high_1d',
                        'mom_low_1d',
                        'mom_exit_open']].to_numpy(dtype=float)

                if not np.isfinite(price_values).all():
                    raise ValueError(f"Missing execution prices for {ticker}")
                if (price_values <= 0).any():
                    raise ValueError(f"Execution prices must be positive for {ticker}")

            cash_weight = cash / value_before_rebalance
            all_tickers = set(current_values).union(selected)
            weight_changes = sum(abs(current_values.get(ticker, 0.0) / value_before_rebalance - (1.0 / n if ticker in selected else 0.0)) for ticker in all_tickers)
            rebalance_turnover = 0.5 * (abs(cash_weight) + weight_changes)
            sleeve_value = value_before_rebalance / n
            new_positions = {}

            for ticker in selected:
                row = ticker_row(ticker)
                entry_price = float(row['mom_entry_open'])
                new_positions[ticker] = {
                    'shares': sleeve_value / entry_price,
                    'entry_price': entry_price,
                    'entry_vol': float(row['stop_vol']),
                    'entry_date': entry_date,
                }

            positions = new_positions
            cash = 0.0
            started = True
            since_rebalance = 0
            rebalanced = True

        if not started:
            continue

        start_values = {}

        for ticker, position in positions.items():
            row = ticker_row(ticker)
            start_values[ticker] = position['shares'] * float(row['mom_entry_open'])

        start_value = cash + sum(start_values.values())

        if not np.isfinite(start_value) or start_value <= 0:
            raise ValueError("Interval starting value must be positive and finite")

        if not rows:
            initial_weights = {ticker: value / start_value for ticker, value in start_values.items()}

            rows.append({
                'date': entry_date,
                'signal_date': signal_date,
                'entry_date': entry_date,
                'exit_date': entry_date,
                'portfolio_return': 0.0,
                'equity': start_value,
                'cash': cash,
                'cash_weight': cash / start_value,
                'rebalanced': False,
                'rebalance_turnover': 0.0,
                'barrier_turnover': 0.0,
                'turnover': 0.0,
                'n_stop_exits': 0,
                'n_profit_exits': 0,
                'n_holdings': len(positions),
                'weights': initial_weights,
                'is_initial': True,
            })

        barrier_notional = 0.0
        n_stop_exits = 0
        n_profit_exits = 0

        for ticker in sorted(list(positions)):
            position = positions[ticker]
            row = ticker_row(ticker)

            price_values = row[[
                    'mom_entry_open',
                    'mom_high_1d',
                    'mom_low_1d',
                    'mom_exit_open']].to_numpy(dtype=float)

            if not np.isfinite(price_values).all():
                raise ValueError(f"Missing interval prices for {ticker}")

            exit_price, outcome, execution = position_stop_exit(
                entry_price=position['entry_price'],
                entry_vol=position['entry_vol'],
                day_open=float(row['mom_entry_open']),
                day_high=float(row['mom_high_1d']),
                day_low=float(row['mom_low_1d']),
                next_open=float(row['mom_exit_open']),
                stop_mult=stop_mult,
                profit_mult=profit_mult,
            )

            if outcome == 'hold':
                continue

            proceeds = position['shares'] * float(exit_price)
            cash += proceeds
            barrier_notional += proceeds

            if outcome == 'stop':
                n_stop_exits += 1
            else:
                n_profit_exits += 1

            event_date = exit_date if execution == 'next_open' else entry_date

            events.append({
                'event_date': event_date,
                'signal_date': signal_date,
                'position_entry_date': position['entry_date'],
                'ticker': ticker,
                'outcome': outcome,
                'execution': execution,
                'entry_price': position['entry_price'],
                'exit_price': float(exit_price),
                'shares': position['shares'],
                'proceeds': proceeds,
                'realised_return': float(exit_price)/ position['entry_price'] - 1,
            })

            del positions[ticker]

        end_values = {}

        for ticker, position in positions.items():
            row = ticker_row(ticker)
            end_values[ticker] = position['shares']* float(row['mom_exit_open'])

        end_value = cash + sum(end_values.values())
        if not np.isfinite(end_value) or end_value <= 0:
            raise ValueError("Portfolio ending value must be positive and finite")

        portfolio_return = end_value / start_value - 1
        barrier_turnover = barrier_notional / start_value
        total_turnover = rebalance_turnover + barrier_turnover

        weights = {ticker: value / end_value for ticker, value in end_values.items()}

        rows.append({
            'date': exit_date,
            'signal_date': signal_date,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'portfolio_return': portfolio_return,
            'equity': end_value,
            'cash': cash,
            'cash_weight': cash / end_value,
            'rebalanced': rebalanced,
            'rebalance_turnover': rebalance_turnover,
            'barrier_turnover': barrier_turnover,
            'turnover': total_turnover,
            'n_stop_exits': n_stop_exits,
            'n_profit_exits': n_profit_exits,
            'n_holdings': len(positions),
            'weights': weights,
            'is_initial': False,
        })
        since_rebalance += 1

    if not rows:
        raise ValueError("No valid stopped portfolio intervals were constructed")
    
    daily = pd.DataFrame(rows).set_index('date').sort_index()
    if daily.index.has_duplicates:
        raise ValueError("Duplicate equity-curve dates were produced")

    event_columns = [
        'event_date',
        'signal_date',
        'position_entry_date',
        'ticker',
        'outcome',
        'execution',
        'entry_price',
        'exit_price',
        'shares',
        'proceeds',
        'realised_return']
    event_log = pd.DataFrame(events, columns=event_columns)

    if not event_log.empty:
        event_log = event_log.sort_values(['event_date', 'ticker']).reset_index(drop=True)

    return daily, event_log

def apply_drawdown_overlay(portfolio_returns, soft_limit=-0.10, hard_limit=-0.25, initial_capital=1.0):

    #we apply exposures to the return day t using the drawdwon 'on' day t - 1
    #the previous strategy applied things immediately, which is not how things happen in reality
    #raw strategy remains our reference so overlay can re-enter after reaching zero exposure.

    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("Portfolio returns must be a pandas Series")

    returns = pd.to_numeric(portfolio_returns.copy(), errors='raise').astype(float)

    if returns.empty:
        raise ValueError("Portfolio returns cannot be empty")
    
    if returns.index.has_duplicates:
        raise ValueError("Portfolio return dates cannot be duplicated")
    
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Portfolio returns must be sorted chronologically")
    
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Portfolio returns must be finite")
    
    if (returns < -1).any():
        raise ValueError("Portfolio returns cannot be below -100%")

    limits = np.asarray([soft_limit, hard_limit, initial_capital], dtype=float)
    if not np.isfinite(limits).all():
        raise ValueError("Limits and initial capital must be finite")
    
    if not -1 <= hard_limit < soft_limit < 0:
        raise ValueError("Limits must be both be negative, with the hard limit being less than the soft limit")

    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive")

    reference_equity = initial_capital * (1 + returns).cumprod()

    #including initial capital as a possible peak handles a negative first return correctly.
    running_peak = reference_equity.cummax().clip(lower=initial_capital)
    reference_drawdown = reference_equity / running_peak - 1

    signal_exposure = pd.Series(drawdown_trading_exposure(reference_drawdown, soft_limit=soft_limit, hard_limit=hard_limit,), index=returns.index, dtype=float)

    #day t drawdown determines day t+1 exposure
    applied_exposure = signal_exposure.shift(1).fillna(1.0)

    managed_return = applied_exposure * returns
    managed_equity = initial_capital * (1 + managed_return).cumprod()

    return pd.DataFrame({
        'raw_return': returns,
        'reference_equity': reference_equity,
        'reference_drawdown': reference_drawdown,
        'signal_exposure': signal_exposure,
        'exposure': applied_exposure,
        'managed_return': managed_return,
        'managed_equity': managed_equity,
    })

def combine_turnover_overlay(base_turnover, exposure):
    #combine stock-selection turnover with turnover caused by changing exposure due to the drawdown

    if not isinstance(base_turnover, pd.Series) or not isinstance(exposure, pd.Series):
        raise TypeError("Turnover and exposure must be pandas Series")

    base_turnover = pd.to_numeric(base_turnover.copy(), errors='raise').astype(float)
    exposure = pd.to_numeric(exposure.copy(), errors='raise').astype(float)

    if base_turnover.empty:
        raise ValueError("Turnover cannot be empty")
    if not base_turnover.index.equals(exposure.index):
        raise ValueError("Turnover and exposure must have identical indexes")
    if base_turnover.index.has_duplicates:
        raise ValueError("Dates cannot be duplicated")
    if not base_turnover.index.is_monotonic_increasing:
        raise ValueError("Dates must be sorted chronologically")
    if not np.isfinite(base_turnover.to_numpy()).all():
        raise ValueError("Turnover must be finite")
    if not np.isfinite(exposure.to_numpy()).all():
        raise ValueError("Exposure must be finite")
    if (base_turnover < 0).any():
        raise ValueError("Turnover cannot be negative")
    if not exposure.between(0, 1).all():
        raise ValueError("Exposure must be between 0 and 1")

    previous_exposure = exposure.shift(1).fillna(exposure.iloc[0]) 
    common_exposure = pd.concat([previous_exposure, exposure], axis=1).min(axis=1)

    strategy_turnover = base_turnover * common_exposure
    overlay_turnover = (exposure - previous_exposure).abs()
    total_turnover = strategy_turnover + overlay_turnover

    return pd.DataFrame({
        'base_turnover': base_turnover,
        'previous_exposure': previous_exposure,
        'exposure': exposure,
        'strategy_turnover': strategy_turnover,
        'overlay_turnover': overlay_turnover,
        'total_turnover': total_turnover,
    })

def apply_volatility_overlay(portfolio_returns, vol_window=21, low_vol_threshold=0.15, high_vol_threshold=0.35, min_exposure=0.2, initial_capital=1.0):
    #calculate volatility on day t and apply its exposure on day t+1

    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("Portfolio returns must be a pandas Series")

    if not isinstance(vol_window, (int, np.integer)) or isinstance(vol_window, bool) or vol_window < 2:
        raise ValueError("Volatility window must be an integer of at least 2")

    parameters = np.asarray([
        low_vol_threshold,
        high_vol_threshold,
        min_exposure,
        initial_capital], dtype=float)

    if not np.isfinite(parameters).all():
        raise ValueError("Overlay parameters must be finite")
    if not 0 <= low_vol_threshold < high_vol_threshold:
        raise ValueError("Volatility thresholds must be non-negative and increasing")
    if not 0 <= min_exposure <= 1:
        raise ValueError("Minimum exposure must be between 0 and 1")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive")

    returns = pd.to_numeric(portfolio_returns.copy(), errors='raise').astype(float)

    if returns.empty:
        raise ValueError("Portfolio returns cannot be empty")
    if returns.index.has_duplicates:
        raise ValueError("Portfolio return dates cannot be duplicated")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Portfolio returns must be chronological")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Portfolio returns must be finite")
    if (returns < -1).any():
        raise ValueError("Returns cannot be below -100%")

    rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)

    signal_exposure = regime_scaled_exposure(rolling_vol, low_vol_threshold=low_vol_threshold, high_vol_threshold=high_vol_threshold, min_exposure=min_exposure)

    applied_exposure = signal_exposure.shift(1).fillna(1.0)
    managed_return = applied_exposure * returns
    managed_equity = initial_capital * (1 + managed_return).cumprod()

    return pd.DataFrame({
        'raw_return': returns,
        'rolling_vol': rolling_vol,
        'signal_exposure': signal_exposure,
        'exposure': applied_exposure,
        'managed_return': managed_return,
        'managed_equity': managed_equity,
    })