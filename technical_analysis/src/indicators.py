import numpy as np
import pandas as pd

#This notebook is purely designed to calculate key statistical metrics in evaluating when to buy and when to sell, such as simple moving average and the exponential moving average

#Unless specified, we look at 20, 50, 200 days for SMA. This looks at short, medium, long term trading

def add_sma (df, windows=[20,50,200]):
    #we create a function to compute the simple moving average over a period of time (the mean of the price over the past n days). We make this a column

    df = df.copy()
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean() #in past_market_analysis we used this to compute rolling volatility (rolling STD)
    return df

#We look at different windows for EMA, since the window means a different thing. from now on, EMA_n is th Exponential moving average over the past n days

def add_ema(df, windows=[12, 26]):
    #we now create a similar function to compute the exponential moving average which is similar to the simple moving average, but weights newer days more.

    df = df.copy()

    for window in windows:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean() #instead of rolling, we use ewm (exponential weighted mean). Old values are never truly forgotten span (window) controls how fast older values decay 
    return df


#We now look at RSI (Relative Strength Index) which is a good metric for measuring the momentum of a ticker (whether it is overbought or oversold)
#RSI > 70 implies overbought, potential sell signal. RSI < 30 implies oversold, potential buy signal

def add_rsi(df, window=14):
    df = df.copy()

    delta = df['Close'].diff() #column of price changes

    gains = delta.clip(lower=0)  #clip sets all values (here lower than 0) to 0. left with all
    losses = -delta.clip(upper=0) #delta.clip(upper=0) contains all negative values. relative strength is a positive value, so here we have 'positive' losses

    ema_gains = gains.ewm(span=window, adjust=False).mean() #relative strength is not a ratio of averages, but EMA instead
    ema_losses = losses.ewm(span=window, adjust=False).mean()

    rs = ema_gains/ema_losses #if a product increases every day, then ema_losses=0, so RSI = 100. sign that it could be overbought

    df['RSI'] = 100 - (100 / (1 + rs)) #
    return df

#We now look at bollinger bands, which measure volatility and tell us whether the price is high or low relative to recent prices

#unless inputted differently, we 'lookback' over a 20 day period (i.e use that as our time period for 'recent prices'). 2x std is used commonly as, for a normal distrubution, 95% of the time the price will lay between these 2 bounds

def add_bollinger_bands(df, window=20, num_std=2):
    df = df.copy()

    df['BB_Middle'] = df['Close'].rolling(window=window).mean() #middle band is the SMA

    rolling_std = df['Close'].rolling(window=window).std() #rolling variance

    df['BB_Upper'] = df['BB_Middle'] + (num_std*rolling_std)
    df['BB_Lower'] = df['BB_Middle'] - (num_std*rolling_std)

    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] # measure the 'squeeze'

    df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) #called %B, is between 0 and 1 95% of the time and tells us 'how far up' the bands we are. 0 implies lower, 1 upper, and 0.5 middle

    return df

#MACD measures the difference between EMA's of different periods of time, for example EMA_12 - EMA-26. In theory, EMA_12 is the fast line, where only the previous 12 days are weighted heavily, and EMA_26 is the slow line (more data days taken into account).

#if MACD, >0: then we have upwards momentum, < 0: then downwards momentum. We will use a signal line (EMA_9 of the MACD line) to see the general trajectory, then plot a histogram of the MACD - signal lines.

#in terms of conclusions, if MACD > signal, then thats a buy signal, opposite is a sell-signal. if MACD crosses above 0, then turning bullish, opposite bearish. histogram growing then momentum accelerating, opposit momentum fading

def add_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    df['MACD'] = ema_fast - ema_slow

    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    return df

# we end with ADX (average directional index), which measures the 'strength' of a trend, in either direction. if ADX >25, then reversal signals less reliable, <20 then reversal signal more reliable

def add_adx(df, window=14):
    df = df.copy()

    high, low, close = df['High'], df['Low'], df['Close']

    #we look at the directional movement of the asset (day to day)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0 #if the asset is decreasing in value, we only care about minus_dm, so set plus_dm to 0. vice versa for minus_dm
    minus_dm[minus_dm < 0] = 0

    #we look at the biggest of these values, then look at the SMA of this

    tr = pd.concat([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)

    dx = 100 * np.abs(plus_di - minus_di)/(plus_di + minus_di)
    df['ADX'] = dx.rolling(window).mean()
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di

    return df


# we combine all indicators into one function for ease of importing
def add_all_indicators(df):

    df = df.copy()

    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_adx(df)

    return df
