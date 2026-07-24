import yfinance as yf
import pandas as pd
import os

def fetch_price_data(ticker, period="1y", interval="1d"): #if no period, interval specified, then we default to 1y, 1d
    """
    we collect data from the web (live with crypto, 15min delay with stocks), ticker is the "thing" you want, period the time period you want to look over, interval is time between data points. if interval <1D, then can only use 60D period by yfinance restrictions
    ticker  : e.g. 'AAPL', 'BTC-USD', 'ETH-USD'
    period  : '1d','5d','1mo','3mo','6mo','1y','2y','5y' (possible inputs)
    interval: '1m','5m','15m','1h','1d','1wk','1mo'
    """

    df = yf.download(ticker, period=period, interval=interval)
    
    # Encountered an issue where df was returning a multi-index, so in that case, we take the first level and get only take a list
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.dropna(inplace=True) #removes rows with missing values, while modifying the original dataframe
    return df

def save_data(df, ticker, folder="../data"): #save the dataframe created above to a file
    """Save DataFrame to CSV."""
    os.makedirs(folder, exist_ok=True) #creates folder, doesn't return error if folder already exists
    path = f"{folder}/{ticker.replace('-','_')}.csv" #hyphon file names can cause issues, call the file the ticker.csv
    df.to_csv(path) #convert to a csv
    print(f"Saved to {path}")

def load_data(ticker, folder="../data"): #read the data we have just saved above 
    """Load previously saved CSV."""
    path = f"{folder}/{ticker.replace('-','_')}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True) #index_col=0 tells first column is the index (dates), parse_dates=true converts the strings into dates so we can plot correct