import numpy as np
import pandas as pd
import yfinance as yf
from labels import triple_barrier_labels
from datetime import datetime
import time

def build_features(df, spy_df=None):
    #we attach features for our dataframe for analysis reasons (most if not all features seen in technical_analysis, with changes being to make them ratios, as general close prices are not stationary as shown in ARIMA/GARCH project, whilst ratios are)

    out = pd.DataFrame(index=df.index)
    close = df['Close']
    high, low, volume = df['High'], df['Low'], df['Volume']

    returns = close.pct_change()

    #momentum
    for window in [1, 5, 10, 21, 63]:
        out[f'return_{window}d'] = close.pct_change(window)

    #volatility
    for window in [10, 21, 63]:
        out[f'vol_{window}d'] = returns.rolling(window).std() *np.sqrt(252)

    out['vol_ratio'] = out['vol_10d'] / out['vol_63d']

    #mean reversion (distance from moving averages)
    for window in [10, 21, 50, 200]:
        sma = close.rolling(window).mean()
        out[f'dist_sma_{window}']  = (close - sma)/sma

    #RSI
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(span=14, adjust=False).mean()
    avg_loss = losses.ewm(span=14, adjust=False).mean()
    out['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))

    #bollinger bands %B
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    out['bb_pctb'] = (close - (sma_20 - 2*std_20)) / (4 * std_20)
    out['bb_width'] = (4 * std_20) / sma_20

    #MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out['macd_hist'] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    #volume
    out['volume_ratio'] = volume / volume.rolling(21).mean()
    out['volume_trend'] = volume.rolling(5).mean() / volume.rolling(21).mean()
    #relative strength to the market
    if spy_df is not None:
        spy_returns = spy_df['Close'].pct_change()
        aligned = spy_returns.reindex(df.index)
        for window in [5, 21, 63]:
            out[f'rel_strength_{window}d'] = (close.pct_change(window) - spy_df['Close'].pct_change(window).reindex(df.index))
            out['beta_63d'] = (returns.rolling(63).cov(aligned) /aligned.rolling(63).var())

    return out

def build_multi_ticker_dataset(tickers, period="10y", horizon=10, ranking_horizon=21, vol_window=21, profit_mult=2.0, stop_mult=1.0, delay=0.2):

    if horizon < 1 or ranking_horizon < 1:
        raise ValueError("Horizon and ranking horizon must be at least one")

    #build pooled feature dataset across multiple tickers
    from data_loader import fetch_price_data


    spy = fetch_price_data("SPY", period=period) 
    all_data = []
    failed = []

    for ticker in tickers:
        try:
            df = fetch_price_data(ticker, period=period)
            if len(df) < 300:
                failed.append(ticker)
                continue
            feats = build_features(df, spy_df=spy)
            labels, hold_days, fwd_return = triple_barrier_labels(df, horizon=horizon, profit_mult=profit_mult, stop_mult=stop_mult)

            ranking_entry = df['Open'].shift(-1)
            ranking_exit = df['Close'].shift(-ranking_horizon)

            combined = feats.copy()
            combined['ranking_return'] = (ranking_exit / ranking_entry) - 1
            combined['mom_return_1d'] = df['Open'].shift(-2) / df['Open'].shift(-1) - 1  #to see the daily returns, the momentum ranking happens at close on day 1, can then execute at open on day 2, and can track daily returns from day 3 onwards

            trade_dates = pd.Series(df.index, index=df.index)
            combined['mom_entry_date'] = trade_dates.shift(-1)
            combined['mom_exit_date'] = trade_dates.shift(-2)

            combined['label'] = labels
            combined['fwd_return'] = fwd_return
            combined['ticker'] = ticker

            daily_return = df['Close'].pct_change()
            combined['stop_vol'] = daily_return.rolling(vol_window).std()
            combined['mom_entry_open'] = df['Open'].shift(-1)
            combined['mom_high_1d'] = df['High'].shift(-1)
            combined['mom_low_1d'] = df['Low'].shift(-1)
            combined['mom_exit_open'] = df['Open'].shift(-2)

            all_data.append(combined)
            time.sleep(delay)

        except Exception as e:
            failed.append(f"{ticker} failed: {e}")
                          
    if not all_data:
        raise ValueError("No ticker datasets were constructed successfully")
    
    pooled = pd.concat(all_data)
    pooled = pooled.sort_index() #sort by date, not ticker

    print(f"Successfully loaded: {len(all_data)} / {len(tickers)} tickers")
    if failed:
        print(f"Failed or insufficient data: {len(failed)} " f"({failed[:10]}{'...' if len(failed) > 10 else ''})")

    return pooled

#part 3

def build_fundamental_features(tickers):

    #we only have current fundamental ratios from yfinance, so this is only valid for today looking forward - if we were to lookback, then theres a lookahead bias

    rows = []
    for ticker in tickers: 
        try: #collect all information we need. protect against primarily import errors.
            info = yf.Ticker(ticker).info
            rows.append({
                'ticker': ticker,
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'roe': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'profit_margin': info.get('profitMargins'),
                'earnings_growth': info.get('earningsGrowth'),
            })
        except Exception as e:
            print(f"{ticker}: failed — {e}")

    return pd.DataFrame(rows).set_index('ticker')

#now get a function to track this going forward.
def start_forward_tracking(tickers, fund_df, output_path='../data/forward_test_log.csv'):
    #rank things today by value/quality and record starting prices. 

    import sys
    from data_loader import fetch_price_data

    fund = fund_df.dropna(subset=['pe_ratio', 'roe']).copy()

    fund['pe_rank'] = fund['pe_ratio'].rank(ascending=True) #lower pe ratio means better 
    fund['roe_rank'] = fund['roe'].rank(ascending=False) #higher roe means better
    fund['composite_rank'] = fund['pe_rank'] + fund['roe_rank']
    fund = fund.sort_values('composite_rank')

    rows = []
    for ticker in fund.index:
        try:
            price = fetch_price_data(ticker, period="5d")['Close'].iloc[-1]
            rows.append({
                'ticker': ticker,
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'start_price': price,
                'pe_ratio': fund.loc[ticker, 'pe_ratio'],
                'roe': fund.loc[ticker, 'roe'],
                'composite_rank': fund.loc[ticker, 'composite_rank'],
                'group': 'top_quintile' if fund.loc[ticker, 'composite_rank'] <=
                         fund['composite_rank'].quantile(0.2) else
                         'bottom_quintile' if fund.loc[ticker, 'composite_rank'] >=
                         fund['composite_rank'].quantile(0.8) else 'middle',
            })
        except Exception as e:
            print(f"{ticker}: failed — {e}")
    log = pd.DataFrame(rows)
    log.to_csv(output_path, index=False)
    print(f"Logged {len(log)} tickers to {output_path}")
    print(f"\nTop quintile (cheap + profitable): {log[log['group']=='top_quintile']['ticker'].tolist()}")
    print(f"Bottom quintile: {log[log['group']=='bottom_quintile']['ticker'].tolist()}")
    return log

def forward_evaluation_tracking(log_path='../data/forward_test_log.csv'):

    #update the data previously defined above

    import sys
    from data_loader import fetch_price_data

    log = pd.read_csv(log_path)
    current_returns = []

    for _, row in log.iterrows():
        try:
            current_price= fetch_price_data(row['ticker'], period="5d")['Close'].iloc[-1]
            ret = current_price/row['start_price'] - 1
            current_returns.append(ret)
        except Exception:
            current_returns.append(np.nan)

    log['current_return'] = current_returns
    days_elapsed = (pd.Timestamp.now() - pd.to_datetime(log['start_date'].iloc[0])).days
    print(f"Days since start date: {days_elapsed}")
    print(log.groupby('group')['current_return'].agg(['mean', 'count']))
    return log

#we build a momentum equity curve that is used for risk_management project. added after this project completed, but suitable place to have it.

def build_momentum_equity_curve(pooled_df, lookback_col, top_frac, rebalance_days, min_universe=5, initial_capital=1.0):

    required = {
        'ticker',
        lookback_col,
        'mom_return_1d',
        'mom_entry_date',
        'mom_exit_date'
    }
    missing = required.difference(pooled_df.columns)
    if missing:
        raise ValueError(f"Missing columns from pooled dataframe: {sorted(missing)}")

    if pooled_df.empty:
        raise ValueError("Pooled data cannot be empty")

    if not 0 < top_frac <= 1:
        raise ValueError("Top fraction must be greater than 0, and at most 1")
    
    if not isinstance(rebalance_days, (int, np.integer)) or isinstance(rebalance_days, bool) or rebalance_days < 1:
        raise ValueError("Rebalance days must be a positive integer")

    if not isinstance(min_universe, (int, np.integer)) or isinstance(min_universe, bool) or min_universe < 2:
        raise ValueError("Minimum universe must be an integer of at least 2")

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be positive and finite")

    data = pooled_df.copy()
    data.index = pd.to_datetime(data.index)
    data['mom_entry_date'] = pd.to_datetime(data['mom_entry_date'])
    data['mom_exit_date'] = pd.to_datetime(data['mom_exit_date'])
    data['mom_return_1d'] = pd.to_numeric(data['mom_return_1d'], errors='raise')
    data[lookback_col] = pd.to_numeric(data[lookback_col], errors='raise')


    if data['ticker'].isna().any():
        raise ValueError("Ticker values cannot be missing")

    date_ticker = pd.MultiIndex.from_arrays([data.index, data['ticker']]) #two index dataframe, one for dates, one for ticker
    
    if date_ticker.has_duplicates:
        raise ValueError("Duplicate date-ticker observation found")

    signal_dates = sorted(data.index.unique())

    if len(signal_dates) < 3:
        raise ValueError("At least 3 trading dates required")

    positions = {}
    cash = float(initial_capital)
    rows = []
    since_rebalance = 0

    for date_i in range(len(signal_dates) - 2): #from the signal, we need 2 extra days to see the 1d day return (signal at close day 1, can buy at open day 2, then see the return by day 3 open)
        signal_date = signal_dates[date_i]
        entry_date = signal_dates[date_i + 1]
        exit_date = signal_dates[date_i + 2]

        day = data.loc[data.index == signal_date]
        first_choice = not positions

        rebalance_due = first_choice or since_rebalance >= rebalance_days

        rebalanced = False
        turnover = 0.0

        if rebalance_due:
            eligible = day.loc[np.isfinite(day[lookback_col]), ['ticker', lookback_col]]

            if len(eligible) < min_universe:
                if first_choice:
                    continue
                raise ValueError(f"Only {len(eligible)} eligible stocks on " f"{signal_date.date()}")

            ranked = eligible.sort_values([lookback_col, 'ticker'], ascending=[False, True])
            n = max(1, int(len(ranked) * top_frac))
            selected = tuple(ranked.head(n)['ticker'])

            total_value = cash + sum(positions.values())
            if total_value <= 0:
                raise ValueError("Equity has been depleted")

            cash_weight = cash / total_value
            all_tickers = set(positions).union(selected)

            weight_changes = sum(abs(positions.get(ticker, 0.0) / total_value - (1.0 / n if ticker in selected else 0.0)) for ticker in all_tickers)
            turnover = 0.5 * (abs(cash_weight) + weight_changes)
            sleeve_value = total_value/n
            positions = {ticker: sleeve_value for ticker in selected}
            cash = 0.0
            since_rebalance = 0
            rebalanced = True

        before_return = cash + sum(positions.values())
        if not np.isfinite(before_return) or before_return <= 0:
            raise ValueError("Portfolio value must stay positive and finite")

        if not rows: 
            initial_weights = {ticker: value / before_return for ticker, value in positions.items()}
            rows.append({
                'date': entry_date,
                'signal_date': signal_date,
                'entry_date': entry_date,
                'exit_date': entry_date,
                'portfolio_return': 0.0,
                'equity': before_return,
                'rebalanced': False,
                'turnover': 0.0,
                'turnover_date': pd.NaT,
                'n_holdings': len(positions),
                'weights': initial_weights,
                'is_initial': True,
            })

        for ticker in positions:
            ticker_rows = day.loc[day['ticker'] == ticker]

            if len(ticker_rows) != 1:
                raise ValueError(
                    f"Expected one row for {ticker} on "
                    f"{signal_date.date()}"
                )

            ticker_row = ticker_rows.iloc[0]
            ticker_return = ticker_row['mom_return_1d']

            dates_aligned = (
                ticker_row['mom_entry_date'] == entry_date
                and ticker_row['mom_exit_date'] == exit_date
            )

            if not np.isfinite(ticker_return) or not dates_aligned:
                raise ValueError(
                    f"Missing or misaligned return for {ticker} "
                    f"after signal date {signal_date.date()}"
                )

            if ticker_return < -1:
                raise ValueError(
                    f"Return below -100% for {ticker} after "
                    f"{signal_date.date()}"
                )

            positions[ticker] *= 1 + float(ticker_return)

        after_return = cash + sum(positions.values())

        if not np.isfinite(after_return) or after_return < 0:
            raise ValueError("Portfolio value became invalid")

        portfolio_return = after_return / before_return - 1

        current_weights = (
            {
                ticker: value / after_return
                for ticker, value in positions.items()
            }
            if after_return > 0
            else {}
        )

        rows.append({
            'date': exit_date,
            'signal_date': signal_date,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'portfolio_return': portfolio_return,
            'equity': after_return,
            'rebalanced': rebalanced,
            'turnover': turnover,
            'turnover_date': (
                entry_date if rebalanced else pd.NaT
            ),
            'n_holdings': len(positions),
            'weights': current_weights,
            'is_initial': False,
        })

        since_rebalance += 1

    if not rows:
        raise ValueError("No valid portfolio intervals were constructed")

    result = pd.DataFrame(rows)
    result = result.set_index('date').sort_index()

    if result.index.has_duplicates:
        raise ValueError("Duplicate equity-curve dates were produced")

    return result
        




            


    