import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

#The aim here is to see whether we should buy or sell based on the stats. We will implement 4 methods, each returning: 1 if we should buy, -1 if we should sell, 0 if inconclusive/hold
#We will combine all 4 signals into one composite signal at the end


#begin with moving average crossover. the rule is here, if SMA_50 (simple moving average over 50 days) crosses above SMA_200, then 1 (buy), and if it crosses below SMA_50, then sell.

def ma_crossover_signal(df):
    df = df.copy()

    df['MA_Above'] = df['SMA_50'] > df['SMA_200']

    #we now detect the crossover value (when df['MA_Above'] changes value, either way)

    df['MA_Signal'] = 0
    df.loc[df['MA_Above'] & ~df['MA_Above'].shift(1).fillna(False).infer_objects(copy=False), 
           'MA_Signal'] = 1    #location of where ma_above true, but the row before it was false - crossed over positively

    df.loc[~df['MA_Above'] & df['MA_Above'].shift(1).fillna(True).infer_objects(copy=False),
           'MA_Signal'] = -1  #location of where ma_above false, but row before it was true - crossed over negatively
    return df

#Idea is to use the RSI from indicators.py, and instead of hard-coding overbought=70, oversold=30, we use the recent values of RSI to determine whats overbought/oversold

def rsi_signal(df, window=252, lower_pct=20, upper_pct=80):
    df = df.copy()
    #defining upper, lower thresholds
    lower_thresh = df['RSI'].rolling(window).quantile(lower_pct/100)
    upper_thresh = df['RSI'].rolling(window).quantile(upper_pct/100)

    df['RSI_Signal'] = 0

    df.loc[(df['RSI'] > lower_thresh) & (df['RSI'].shift(1) <= lower_thresh), 'RSI_Signal'] = 1 #locations of a crossover from below lower threshold to above lower threshold
    df.loc[(df['RSI'] < upper_thresh) & (df['RSI'].shift(1) >= upper_thresh), 'RSI_Signal'] = -1

    return df

#similarly to above, we return a buy signal when we cross above the bottom bollinger band, a sell signal when we cross below the top bollinger band. Often (~95% of the time), this will return 0, but in the cases it applies, it is a significant signal
#along the same lines, we set upper, lower thresholds based on the recent behaviour of the ticker

def bollinger_signal(df, window=252, lower_pct=10, upper_pct=90):
    df = df.copy()

    lower_thres = df['BB_PctB'].rolling(window).quantile(lower_pct/100)
    upper_thres = df['BB_PctB'].rolling(window).quantile(upper_pct/100)

    df['BB_Signal'] = 0

    df.loc[(df['BB_PctB'] > lower_thres) & (df['BB_PctB'].shift(1) <= lower_thres),'BB_Signal'] = 1 #if %B was less than 0, now above 0, then buy

    df.loc[(df['BB_PctB'] < upper_thres) & (df['BB_PctB'].shift(1) >= upper_thres),'BB_Signal'] = -1 #if %B was above 1, now less than 1, then sell

    return df

#analogously, if MACD line crosses above signal line, then buy, if crosses below, then sell.

def macd_signal(df):
    df = df.copy()

    df['MACD_Cross_Signal'] = 0

    df.loc[(df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)), 'MACD_Cross_Signal'] = 1 #if MACD line above signal line, and was below, then buy
    df.loc[(df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)), 'MACD_Cross_Signal'] = -1 #if MACD line below signal line, but was above, then crossover below so sell

    return df

def trend_signal(df):
    df = df.copy()
    df['Trend_Signal'] = 0

    sma_50_rising = df['SMA_50'] > df['SMA_50'].shift(5)

    df.loc[(df['Close'] > df['SMA_50']) & sma_50_rising, 'Trend_Signal'] = 1
    df.loc[(df['Close'] < df['SMA_50']) & ~sma_50_rising, 'Trend_Signal'] = -1
    return df

#these are the 4 (buy/sell) signals corresponding to our indicators in indicators.py, alongside a trend signal, which indicates the general trend of the ticker. Not one of these is perfect, so we construct a composite signal which takes all signals into account

#we choose a 2 threshold. this means that the sum of all signals has to be greater than (or equal to) 2 for us to generate a signal, i.e,  muliple different signals telling us to buy/sell
def composite_signal(df, threshold=2):
    df = df.copy()

    df = ma_crossover_signal(df)
    df = rsi_signal(df)
    df = bollinger_signal(df)
    df = macd_signal(df)
    df = trend_signal(df)

    df['Signal_Score'] = (df['MA_Signal'] + df['RSI_Signal'] + df['BB_Signal'] + df['MACD_Cross_Signal'] + df['Trend_Signal'])

    df['Signal'] = 0
    df.loc[df['Signal_Score'] >= threshold, 'Signal'] = 1
    df.loc[df['Signal_Score'] <= -threshold, 'Signal'] = -1

    return df
#when testing the above signal with an example ticker ('AAPL') which had a upwards trend, we saw some contradictions. There were only sell-signals, and whilst ~70% were accurate, there were 15-20% of sell signal which preceded sharp upturns in price. To combat this, we use various filters and other methods to improve results and create a robust filter.

def trend_regime_filter(df):
    df = df.copy()
    #filter out signals which oppose the trend of the ticker
    df['Bull_Regime'] = df['Close'] > df['SMA_200']
    return df

#high volume confirms an instinct behind a move

def volume_filter(df, window=20):
    df = df.copy()
    df['Volume_Avg'] = df['Volume'].rolling(window).mean()
    df['High_Volume'] = df['Volume'] > df['Volume_Avg']

    return df
#we use 2 terms, bullish/bearish divergence. bearish divergence occurs when we have a higher high price, but lower high RSI. bullish being the opposite

def rsi_divergence(df, window=14):
    df = df.copy()

    price_high = df['Close'].rolling(window).max() == df['Close']
    price_low = df['Close'].rolling(window).min() == df['Close']

    # if the price is a high, its higher than 14 days ago and the RSI is less than 14 days ago, then bearish divergence. analogous for bullish divergence

    df['Bearish_Divergence'] = (price_high & (df['Close'] > df['Close'].shift(window)) & (df['RSI'] < df['RSI'].shift(window)))
    df['Bullish_Divergence'] = (price_low & (df['Close'] < df['Close'].shift(window)) & (df['RSI'] > df['RSI'].shift(window)))
    return df
#when ran originally, had a lot of sell signals in close proximity often corresponding to the same event, so introduce a cooldown.

def apply_cooldown(df, signal_col='Signal', cooldown=5):
    df = df.copy()
    last_signal_day = -999 #set last_signal_day original to be way before to ensure the first day doesnt get cancelled by cooldown
    filtered = []
    for i, sig in enumerate(df[signal_col]):
        if sig !=0 and i - last_signal_day >= cooldown:
            filtered.append(sig)
            last_signal_day = i
        else:
            filtered.append(0)
    df[signal_col] = filtered
    return df


#we can now create our robust signal

def robust_composite_signal(df, threshold=1, cooldown=5):
    df = df.copy()

    #import all signals/filters 
    df = ma_crossover_signal(df)
    df = rsi_signal(df)
    df = bollinger_signal(df)
    df = macd_signal(df)
    df = trend_signal(df)
    df = trend_regime_filter(df)
    df = volume_filter(df)
    df = rsi_divergence(df)

    #we use the ideas from our old composite signal, but change it slightly. We use a weighted trend model. We weight the trend signal more/less depending on whether the ADX is high or not, capping at 3 to avoid huge weighting.
    adx_weight = (df['ADX'] / 25).clip(0,3)
    df['Weighted_Trend'] = adx_weight * df['Trend_Signal']


    df['Signal_Score'] = (df['MA_Signal'] + df['RSI_Signal'] + df['BB_Signal'] + df['MACD_Cross_Signal'] + df['Weighted_Trend'])

    raw_buy = df['Signal_Score'] >= threshold
    raw_sell = df['Signal_Score'] <= -threshold

    #we now filter out low volume, high ADX and take into account whether its a bull/bear regime
    
    buy_ok = raw_buy & df['Bull_Regime'] & (df['ADX'] < 30) & df['High_Volume']
    sell_ok = raw_sell & ~df['Bull_Regime'] & (df['ADX'] < 30) & df['High_Volume']

    #bullish/bearish divergence is clearly a really strong condition, so that itself is enough of a buy/signal to influence us

    buy_ok = buy_ok | df['Bullish_Divergence']
    sell_ok = sell_ok | df['Bearish_Divergence']


    df['Signal'] = 0
    df.loc[buy_ok, 'Signal'] = 1
    df.loc[sell_ok, 'Signal'] = -1

    #applying the cooldown to avoid repeat signals

    df = apply_cooldown(df, cooldown=cooldown)
    return df

#after analysis on a few tickers, we have encountered lag. we backtest to see if our system is actually better than just buying and sticking

def backtest_signals(df, initial_capital=10000):
    #plan is, buy on 1, sell on -1, hold otherwise

    df = df.copy()
    position = 0 #position will describe our position, 0 = no position (cash), 1 = holding asset
    cash = initial_capital
    shares=0
    portfolio = []

    for i, row in df.iterrows():
        if row['Signal'] == 1 and position == 0: #if buy signal, and haven't bought, then spend all cash on shares
            shares = cash / row['Close']
            cash = 0
            position = 1
        elif row['Signal'] == -1 and position == 1: #if sell signal and holding shares, then sell all shares
            cash = shares * row['Close']
            shares = 0
            position = 0

        portfolio.append(cash + shares * row['Close'])

    df['Portfolio_Value'] = portfolio
    df['Buy_Hold_Value'] = initial_capital * (df['Close'] / df['Close'].iloc[0])

    return df





