import yfinance as yf
import pandas as pd
import os

def fetch_price_data(ticker, period="1y", interval="1d", *, start=None, end=None, auto_adjust=True, keepna=False):
    #fetch historical data, defaulting to daily prices and yearly periods.
    if start is not None or end is not None:
        period = None

    df = yf.download(ticker, period=period, interval=interval, start=start, end=end, auto_adjust=auto_adjust, keepna=keepna, group_by="column", multi_level_index=False, progress=False)

    if df is None or df.empty:
        raise RuntimeError("No price data were downloaded")

    #remove missing rows by default, but preserve them when requested
    if not keepna:
        df.dropna(inplace=True)

    return df

def save_data(df, ticker, folder="../data"): #save the dataframe created above to a file

    os.makedirs(folder, exist_ok=True) #creates folder, doesn't return error if folder already exists
    path = f"{folder}/{ticker.replace('-','_')}.csv" #hyphon file names can cause issues, call the file the ticker.csv
    df.to_csv(path) #convert to a csv
    print(f"Saved to {path}")

def load_data(ticker, folder="../data"): #read the data we have just saved above 

    path = f"{folder}/{ticker.replace('-','_')}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True) #index_col=0 tells first column is the index (dates), parse_dates=true converts the strings into dates so we can plot correct

#implement a function to 'pull' all current s&p 500 tickers. not used in this project, but in future projects (ml_trading_signals and autonomous_trading_system)

def get_sp500_tickers():

    import requests
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    sp500_details = pd.read_html(response.text)
    sp500_table = sp500_details[0]
    tickers = sp500_table['Symbol'].tolist()
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers