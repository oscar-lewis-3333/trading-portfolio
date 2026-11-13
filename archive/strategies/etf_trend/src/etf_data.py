import json
from pathlib import Path

import pandas as pd
import yfinance as yf
import numpy as np
import pandas_market_calendars as mcal


def load_etf_history(symbol, start, end, cache_dir, *, refresh=False, repair=False):
    #load cached daily history or download it. end date exclusive
    start = pd.Timestamp(start).strftime("%Y-%m-%d")
    end = pd.Timestamp(end).strftime("%Y-%m-%d")

    if start >= end:
        raise ValueError("start must be earlier than end.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    version = "repaired" if repair else "raw"
    stem = f"{symbol}_{start}_{end}_{version}"
    price_path = cache_dir / f"{stem}.csv"
    metadata_path = cache_dir / f"{stem}.json"

    if price_path.exists() and metadata_path.exists() and not refresh:
        prices = pd.read_csv(price_path, index_col="Date", parse_dates=True)
        metadata = json.loads(metadata_path.read_text())
        return prices, metadata

    ticker = yf.Ticker(symbol)
    if repair:
        ticker.get_history_metadata()

    prices = ticker.history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        back_adjust=False,
        actions=True,
        repair=repair,
        keepna=True,
        rounding=False,
        raise_errors=True)

    if prices is None or prices.empty:
        raise ValueError(f"No history returned for {symbol}.")

    required = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {sorted(missing)}")

    #preserve exchange-local calendar date when removing timezone
    prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index().rename_axis("Date")
    if prices.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate dates.")

    history_metadata = ticker.get_history_metadata()
    metadata = {
        "symbol": symbol,
        "currency": history_metadata.get("currency"),
        "exchange_timezone": history_metadata.get("exchangeTimezoneName"),
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "yfinance_version": yf.__version__
    }

    prices.to_csv(price_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return prices, metadata

def audit_etf_history(prices, jump_threshold=0.20):
    #summarising potential problems (no observation change)
    price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    close = prices["Close"]
    dividends = prices["Dividends"]
    factor = prices["Adj Close"] / close

    #compare adjustment with recorded dividend yield
    implied_yield = 1 - factor.shift(1) / factor
    recorded_yield = dividends / close.shift(1)

    dividend_days = (dividends.gt(0) & prices["Stock Splits"].eq(0))
    adjustment_ratio = (implied_yield / recorded_yield).where(dividend_days).replace([np.inf, -np.inf], np.nan)

    close_returns = close.pct_change(fill_method=None)
    return pd.Series({
        "duplicate_dates": int(prices.index.duplicated().sum()),
        "incomplete_price_rows": int(prices[price_columns].isna().any(axis=1).sum()),
        "zero_volume_rows": int(prices["Volume"].eq(0).sum()),
        "large_close_moves": int(close_returns.abs().gt(jump_threshold).sum()),
        "dividend_events": int(dividends.gt(0).sum()),
        "first_dividend_date": prices.index[dividends.gt(0)].min(),
        "median_adjustment_ratio": adjustment_ratio.median() if adjustment_ratio.notna().any() else np.nan
    })

def build_lse_schedule(start, end):

    #build LSE trading schedule (end exclusive)
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    if start >= end:
        raise ValueError("start must be earlier than end.")

    schedule = mcal.get_calendar("LSE").schedule(start_date=start, end_date=end - pd.Timedelta(days=1))
    #correct the (misplaced) 2022 spring holiday in older calendar versions
    session = pd.Timestamp("2022-05-30")

    if start <= session < end and session not in schedule.index:
        schedule.loc[session, ["market_open", "market_close"]] = [
            pd.Timestamp("2022-05-30 08:00", tz="Europe/London").tz_convert("UTC"),
            pd.Timestamp("2022-05-30 16:30", tz="Europe/London").tz_convert("UTC")]

    return schedule.sort_index().rename_axis("Date")

def build_monthly_prices(histories, schedule):

    #select adjusted closes on final session of each month
    if not histories or schedule.empty:
        raise ValueError("Provide nonempty histories and a trading schedule.")

    sessions = pd.DatetimeIndex(schedule.index, name="Date")

    #extend calendar to identify whether final month complete
    calendar_end = sessions[-1].to_period("M").end_time.normalize() + pd.Timedelta(days=1)
    full_schedule = build_lse_schedule(sessions[0], calendar_end)
    calendar_dates = full_schedule.index

    month_ends = calendar_dates.to_series().groupby(calendar_dates.to_period("M")).max()
    month_ends = month_ends.loc[month_ends <= sessions[-1]]
    month_end_dates = pd.DatetimeIndex(month_ends.to_numpy(), name="Date")

    columns = {}

    for ticker, frame in histories.items():
        if not frame.index.equals(sessions):
            raise ValueError(f"{ticker}: history does not match the schedule.")

        columns[ticker] = frame["Adj Close"].reindex(month_end_dates)

    return pd.DataFrame(columns, index=month_end_dates).rename_axis(columns="ticker")

def build_daily_prices(histories, schedule):

    #build aligned (adjusted) opens and closes (whilst not filling missing quotes)
    if not histories or schedule.empty:
        raise ValueError("Provide nonempty histories and a trading schedule.")

    sessions = pd.DatetimeIndex(schedule.index, name="Date")
    opens = {}
    closes = {}

    for ticker, frame in histories.items():
        if not frame.index.equals(sessions):
            raise ValueError(f"{ticker}: history does not match the schedule.")

        prices = frame[["Open", "Close", "Adj Close"]].astype(float)
        invalid = prices.notna() & ((prices <= 0) | ~np.isfinite(prices))
        if invalid.any().any():
            raise ValueError(f"{ticker}: observed prices must be positive and finite.")

        adjustment = prices["Adj Close"] / prices["Close"]
        opens[ticker] = prices["Open"] * adjustment
        closes[ticker] = prices["Adj Close"]

    adjusted_open = pd.DataFrame(opens, index=sessions).rename_axis(columns="ticker")
    adjusted_close = pd.DataFrame(closes, index=sessions).rename_axis(columns="ticker")

    return adjusted_open, adjusted_close

def prepare_valuation_marks(open_prices, close_prices, excluded_quotes, *, execution_dates):
    #substitute previous-session close for reviewed, isolated quote gaps
    if (open_prices.empty or not open_prices.index.is_unique or not open_prices.index.is_monotonic_increasing or not open_prices.columns.is_unique):
        raise ValueError("Provide prices with ordered, unique dates and assets.")

    for frame in (close_prices, excluded_quotes):
        if (not frame.index.equals(open_prices.index) or not frame.columns.equals(open_prices.columns)):
            raise ValueError("Prices and exclusion flags must have matching labels.")

    #policy supports reviewed gaps where both quotes were excluded
    if (not open_prices.isna().equals(excluded_quotes) or not close_prices.isna().equals(excluded_quotes)):
        raise ValueError("Missing quotes must match the reviewed exclusions exactly.")

    execution_dates = pd.DatetimeIndex(execution_dates)
    if (execution_dates.hasnans or not execution_dates.is_unique or not execution_dates.isin(open_prices.index).all()):
        raise ValueError("Execution dates must be unique and present in the prices.")

    stale_days = excluded_quotes.any(axis=1)
    if stale_days.loc[open_prices.index.isin(execution_dates)].any():
        raise ValueError("Stale quotes cannot be substituted on execution dates.")
    if stale_days.iloc[-1]:
        raise ValueError("The final portfolio valuation requires observed quotes.")

    previous_close = close_prices.shift(1)
    valuation_open = open_prices.mask(excluded_quotes, previous_close)
    valuation_close = close_prices.mask(excluded_quotes, previous_close)

    #also rejects consecutive gaps, or gap without valid close
    for marks in (valuation_open, valuation_close):
        if not (np.isfinite(marks) & marks.gt(0)).all().all():
            raise ValueError("Valuation marks require positive prices and a valid previous close.")

    return valuation_open, valuation_close, excluded_quotes.copy()