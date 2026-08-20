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

def build_multi_ticker_dataset(tickers, period="10y", horizon=10, profit_mult=2.0, stop_mult=1.0, delay=0.2):

    #build pooled feature dataset across multiple tickers
    import sys
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

            combined = feats.copy()
            combined['label'] = labels
            combined['fwd_return'] = fwd_return
            combined['ticker'] = ticker

            all_data.append(combined)
            time.sleep(delay)

        except Exception as e:
            failed.append(ticker)

            
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

def build_momentum_equity_curve(pooled_df, lookback_col='return_21d', top_frac=0.25, rebalance_days=63, min_universe=5):

    #rank, select top fraction each day using that days single-day return compounded continuously 

    clean = pooled_df.dropna(subset=[lookback_col, 'return_1d'])
    all_dates = sorted(clean.index.unique())
    
    daily_returns = []
    current_selection = None
     
    for i, date in enumerate(all_dates):
        day = clean.loc[clean.index == date]
        if len(day) < min_universe:
            continue

        if i % rebalance_days == 0:
            days = day.sort_values(lookback_col, ascending=False)
            n = max(1, int(len(day) * top_frac))
            current_selection = set(days.head(n)['ticker'])

        if current_selection is None:
            continue    

        held = day[day['ticker'].isin(current_selection)]
        if len(held) == 0:
            continue

        daily_returns.append({
            'date': date,
            'portfolio_return': held['return_1d'].mean()
        })

    daily_df = pd.DataFrame(daily_returns).set_index('date').sort_index()
    daily_df['equity'] = (1 + daily_df['portfolio_return']).cumprod()

    return daily_df
