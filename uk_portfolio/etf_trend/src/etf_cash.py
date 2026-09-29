#BOE interest rate history
import json
import shutil
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import etf_backtest

BANK_RATE_SERIES = "IUDBEDR"
_BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    "?csv.x=yes&Datefrom={start}&Dateto={end}&SeriesCodes={series}"
    "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)


def boe_series_url(series_code, start, end):
    #BoE date format is 01/Jan/2016
    start = pd.Timestamp(start).strftime("%d/%b/%Y")
    end = pd.Timestamp(end).strftime("%d/%b/%Y")
    return _BOE_URL.format(start=start, end=end, series=series_code)


def parse_boe_csv(text, series_code=BANK_RATE_SERIES):
    #parsing BoE database CSV to get a series of decimal interest rate
    if "<html" in text[:500].lower():
        raise ValueError("The BoE response is an HTML page, not CSV. Download the CSV manually.")

    frame = pd.read_csv(StringIO(text))
    frame.columns = [str(column).strip() for column in frame.columns]
    if series_code not in frame.columns:
        raise ValueError(f"Column {series_code} not found; columns are {list(frame.columns)}.")

    date_column = frame.columns[0]
    raw_dates = frame[date_column].astype(str).str.strip()
    dates = pd.to_datetime(raw_dates, format="%d %b %Y", errors="coerce")
    if dates.isna().any():
        dates = pd.to_datetime(raw_dates, dayfirst=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("Some BoE dates could not be parsed.")

    values = pd.to_numeric(frame[series_code], errors="coerce")
    if values.isna().any():
        raise ValueError("Some BoE values are missing or non-numeric.")

    rates = pd.Series(values.to_numpy() / 100, index=pd.DatetimeIndex(dates, name="Date"), name="bank_rate")
    rates = rates.sort_index()
    if rates.index.has_duplicates:
        raise ValueError("Duplicate BoE dates.")
    if not np.isfinite(rates).all() or (rates < 0).any() or (rates > 0.25).any():
        raise ValueError("Bank Rate values lie outside 0-25%.")
    return rates


def load_bank_rate(cache_dir, *, start, end, refresh=False, source_file=None, series_code=BANK_RATE_SERIES):
    #load said csv created above
    start = pd.Timestamp(start).strftime("%Y-%m-%d")
    end = pd.Timestamp(end).strftime("%Y-%m-%d")
    if start >= end:
        raise ValueError("start must be earlier than end.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{series_code}_{start}_{end}_raw"
    csv_path = cache_dir / f"{stem}.csv"
    metadata_path = cache_dir / f"{stem}.json"

    if csv_path.exists() and metadata_path.exists() and not refresh:
        rates = parse_boe_csv(csv_path.read_text(), series_code)
        return rates, json.loads(metadata_path.read_text())

    url = boe_series_url(series_code, start, end)
    if source_file is not None:
        shutil.copyfile(source_file, csv_path)
        source = f"manual download saved as {Path(source_file).name}"
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            csv_path.write_bytes(response.read())
        source = "automatic download"

    rates = parse_boe_csv(csv_path.read_text(), series_code)
    metadata = {
        "series_code": series_code,
        "description": "Official Bank Rate, daily (Bank of England Interactive Database)",
        "units": "percent in the CSV; loader returns annual decimals",
        "url": url,
        "source": source,
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "first_date": rates.index[0].strftime("%Y-%m-%d"),
        "last_date": rates.index[-1].strftime("%Y-%m-%d"),
        "observations": int(len(rates))
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return rates, metadata


def audit_bank_rate(rates, sessions):
    #check for issues within the series
    sessions = pd.DatetimeIndex(sessions)
    changes = rates.loc[rates.diff().ne(0)].iloc[1:]
    gaps = rates.index.to_series().diff().dt.days
    return pd.Series({
        "first_date": rates.index[0],
        "last_date": rates.index[-1],
        "observations": int(len(rates)),
        "largest_gap_days": int(gaps.max()) if len(gaps) > 1 else 0,
        "rate_changes": int(len(changes)),
        "min_rate_pct": rates.min() * 100,
        "max_rate_pct": rates.max() * 100,
        "covers_first_session": bool(rates.index[0] <= sessions[0]),
        "days_before_last_session": int((sessions[-1] - rates.index[-1]).days)
    })


def rate_changes(rates):
    #dates which the interest changed
    changes = rates.loc[rates.diff().ne(0)].iloc[1:]
    return (changes * 100).rename("new_rate_pct")


def build_cash_rate_scenarios(bank_rate, *, spread=0.005):
    #conditions for the interest rate scenarios
    return {
        "zero": 0.0,
        "bank_rate": bank_rate.copy(),
        "bank_rate_minus_50bp": (bank_rate - spread).clip(lower=0.0),
    }


def monthly_cash_returns(cash_rate_annual, sessions):
    #cash-only portfolio, i.e what happens if we hold cash the entire time
    factors = etf_backtest.cash_accrual_factors(cash_rate_annual, sessions)
    return (factors.groupby(factors.index.to_period("M")).prod() - 1).rename("cash")