import yfinance as yf
import pandas as pd
import numpy as np


#start by collecting the data needed for reversal signals. This is different from fetch_price_data since we want to handle adjusted prices consistently across all tickers, whilst keeping spy outside the universe.
def fetch_reversal_data(ticker, start, end):

    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, actions=True, keepna=True, multi_level_index=False, progress=False, threads=False)

    if df is None or df.empty:
        raise ValueError(f"No data found for ticker {ticker} between {start} and {end}")

    return df.sort_index().rename_axis("Date")

def audit_price_coverage(panel, tickers, sessions):

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    missing_columns = set(required + ["ticker"]).difference(panel.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    records = []
    for ticker in tickers:
        df = panel.loc[panel["ticker"] == ticker]
        observed_dates = df.index.unique()

        records.append({
            "ticker": ticker,
            "rows_returned": len(df),
            "expected_sessions": len(sessions),
            "duplicate_extra_rows": int(df.index.duplicated().sum()),
            "missing_sessions": len(sessions.difference(observed_dates)),
            "unexpected_dates": len(observed_dates.difference(sessions)),
            "incomplete_rows": int(df[required].isna().any(axis=1).sum()),
        })

    return pd.DataFrame(records).set_index("ticker")

def align_price_panel(panel, tickers, sessions):

    sessions = pd.DatetimeIndex(sessions, name="Date")
    tickers = list(tickers)

    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError("Provide a nonempty list of unique tickers.")
    if (sessions.empty or sessions.hasnans or not sessions.is_unique or not sessions.is_monotonic_increasing):
        raise ValueError("Sessions must be valid, unique and sorted.")

    frames = []

    for ticker in tickers:
        df = panel.loc[panel["ticker"] == ticker].copy()

        if df.index.has_duplicates:
            raise ValueError(f"{ticker} has duplicate dates.")
        if not df.index.isin(sessions).all():
            raise ValueError(f"{ticker} contains dates outside the calendar.")

        observed_dates = df.index
        df = df.reindex(sessions)
        df["ticker"] = ticker
        df["observed_row"] = sessions.isin(observed_dates)
        frames.append(df)

    return pd.concat(frames).sort_index()

#do our own adjusting of prices to ensure consistency across all tickers and avoid yfinance auto_adjust issues. Important here so that prices aren't adjusted by splits or divedends.
def add_adjusted_prices(panel):

    required = ["Open", "High", "Low", "Close", "Adj Close"]

    missing_columns = set(required).difference(panel.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    values = panel[required].astype(float)
    #allow missing observations, but ensure all observed prices are positive and finite.
    invalid = values.notna() & ((values <= 0) | ~np.isfinite(values))
    if invalid.any().any():
        raise ValueError("Observed prices must be positive and finite.")

    out = panel.copy()
    factor = values["Adj Close"] / values["Close"]
    out["adjustment_factor"] = factor

    for column in ["Open", "High", "Low"]:
        out[f"adj_{column.lower()}"] = values[column] * factor
    out["adj_close"] = values["Adj Close"]

    return out

def fetch_quote_snapshot(tickers):
    #preserve failed requested and missing fields 
    fields = ["symbol", "currency", "marketState", "bid", "ask", "bidSize", "askSize", "regularMarketTime"]
    records = []

    for ticker in tickers:
        row = {
            "ticker": ticker,
            "source": "Yahoo Finance",
            "yfinance_version": yf.__version__,
            "fetch_status": "failed",
            "error": "",
            **{field: None for field in fields}
        }
        try:
            info = yf.Ticker(ticker).get_info()
            row.update({field: info.get(field) for field in fields})
            row["fetch_status"] = "ok"
        except Exception as exc:
            row["error"] = str(exc)

        row["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC")
        records.append(row)
        
    return pd.DataFrame(records).set_index("ticker")


    

