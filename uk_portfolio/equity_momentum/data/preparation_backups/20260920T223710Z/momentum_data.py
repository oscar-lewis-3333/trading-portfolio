from pathlib import Path
import time

import yfinance as yf
import pandas as pd
import numpy as np

def load_candidate_universe(universe_dir, snapshot_path):
    #load (frozen) candidate list.
    snapshot_path = Path(snapshot_path)

    if snapshot_path.exists():
        candidates = pd.read_csv(snapshot_path, dtype="string", keep_default_na=False)
    else:
        catalogue = pd.read_csv(Path(universe_dir) / "t212_isa_universe.csv", dtype="string", keep_default_na=False)
        candidate_mask = (catalogue["isa_eligible"].str.lower().eq("true")
            & catalogue["quote_unit"].isin(["GBP", "GBX"])
            & catalogue["exchange"].isin(["London Stock Exchange", "London Stock Exchange AIM"])
            & catalogue["instrument_type"].eq("ordinary_share")
            & catalogue["listing_status"].ne("delisted"))
        candidates = catalogue.loc[candidate_mask].sort_values("isin").reset_index(drop=True).copy()
        
    if candidates.empty:
        raise ValueError("The candidate universe is empty.")
    for column in ["api_ticker", "isin", "yf_symbol"]:
        if candidates[column].str.strip().eq("").any():
            raise ValueError(f"Missing identifier: {column}")
        if candidates[column].duplicated().any():
            raise ValueError(f"Duplicate identifier: {column}")

    if not snapshot_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(snapshot_path, index=False)

    return candidates

def load_cached_prices(cache_path, universe):
    #load existing local prices and report coverage for each candidate
    bundle = pd.read_pickle(Path(cache_path))
    source = bundle["prices"]
    if source.attrs.get("quote_currency") != "GBP":
        raise ValueError("Expected cached prices already converted to GBP.")

    prices = source.loc[source["ticker"].isin(universe["yf_symbol"])].set_index("ticker", append=True).sort_index().copy()
    
    #keep price metadata, while not inheriting reversal_signals data
    prices.attrs = {key: source.attrs[key] for key in ["quote_currency", "share_basis"]}
    if not prices.index.is_unique:
        raise ValueError("Duplicate stock-date rows in cached prices.")

    schedule = bundle["schedule"].copy()
    if not prices.index.get_level_values("Date").isin(schedule.index).all():
        raise ValueError("Cached prices contain dates outside the schedule.")

    valid_close = (prices["observed_row"].fillna(False) & prices["adj_close"].gt(0))
    observed = prices.loc[valid_close].reset_index()

    summary = observed.groupby("ticker").agg(cached_close_sessions=("Date", "size"), first_cached_session=("Date", "min"), last_cached_session=("Date", "max"))
    coverage = universe[["yf_symbol", "isin", "name"]].merge(summary, left_on="yf_symbol", right_index=True, how="left", validate="one_to_one")
    coverage["cached_close_sessions"] = coverage["cached_close_sessions"].fillna(0).astype(int)
    coverage["has_cached_prices"] = coverage["cached_close_sessions"].gt(0)

    return {
        "prices": prices,
        "schedule": schedule,
        "coverage": coverage,
    }

def download_price_cache(symbols, cache_dir, *, start, end):
    #cache raw yahoo prices and download outcomes.
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(set(symbols))
    report = []

    for number, symbol in enumerate(symbols, start=1):
        path = cache_dir / f"{symbol}.pkl"

        if path.exists():
            payload = pd.read_pickle(path)
            if (payload["symbol"] != symbol or payload["start"] != start or payload["end"] != end):
                raise ValueError(f"Cached request differs for {symbol}")

        else:
            payload = {"symbol": symbol,
                "start": start,
                "end": end,
                "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "yfinance_version": yf.__version__,
                "status": "failed",
                "prices": pd.DataFrame(),
                "quote_currency": None,
                "error": "",
                "metadata_error": ""
            }

            ticker = yf.Ticker(symbol)
            try:
                prices = ticker.history(start=start, end=end, interval="1d",auto_adjust=False,
                    actions=True, keepna=True, repair=False, raise_errors=True)
                if prices.empty:
                    raise ValueError("Yahoo returned no price rows.")

                payload["prices"] = prices.sort_index().rename_axis("Date")
                payload["status"] = "downloaded"
            except Exception as e:
                payload["error"] = f"{type(e).__name__}: {e}"

            if payload["status"] == "downloaded":
                try:
                    payload["quote_currency"] = ticker.history_metadata.get("currency")
                except Exception as e:
                    #preserve downloaded prices even if metadata fails
                    payload["metadata_error"] = f"{type(e).__name__}: {e}"

            temporary = path.with_suffix(".pkl.tmp")
            pd.to_pickle(payload, temporary)
            temporary.replace(path)
            time.sleep(1.0)

        report.append({
            "ticker": symbol,
            "status": payload["status"],
            "rows": len(payload["prices"]),
            "quote_currency": payload["quote_currency"],
            "error": payload["error"],
            "metadata_error": payload["metadata_error"]
        })

        if number % 10 == 0 or number == len(symbols):
            print(f"Processed {number}/{len(symbols)} symbols")

    return pd.DataFrame(report)


def prepare_downloaded_prices(symbols, cache_dir, schedule):
    #convert successful downloads to GBP without filling gaps
    cache_dir = Path(cache_dir)
    frames = []

    divisors = {"GBP": 1.0, "GBp": 100.0, "GBX": 100.0}
    price_columns = ["Open", "High", "Low", "Close", "Adj Close"]

    for symbol in sorted(set(symbols)):
        payload = pd.read_pickle(cache_dir / f"{symbol}.pkl")

        if payload["symbol"] != symbol:
            raise ValueError(f"Cached symbol mismatch: {symbol}")
        if payload["status"] != "downloaded":
            continue

        currency = payload["quote_currency"]
        if currency not in divisors:
            raise ValueError(f"Unrecognised quote currency for {symbol}: {currency}")

        frame = payload["prices"].copy()
        dates = pd.DatetimeIndex(frame.index)
        if dates.tz is not None:
            dates = dates.tz_convert("Europe/London").tz_localize(None)
        frame.index = dates.normalize().rename("Date")

        if not frame.index.is_unique:
            raise ValueError(f"Duplicate dates for {symbol}")
        if not frame.index.isin(schedule.index).all():
            raise ValueError(f"Dates outside the exchange schedule: {symbol}")

        #convert monetary columns only
        money_columns = price_columns + ["Dividends"]
        frame[money_columns] = frame[money_columns] / divisors[currency]

        frame["adjustment_factor"] = frame["Adj Close"] / frame["Close"]

        for column in ["Open", "High", "Low", "Close"]:
            frame[f"adj_{column.lower()}"] = frame[column] * frame["adjustment_factor"]

        frame["observed_row"] = True
        frame = frame.reindex(schedule.index)
        frame.index.name = "Date"

        frame["observed_row"] = frame["observed_row"].eq(True)
        frame["ticker"] = symbol
        frame["data_vintage"] = payload["retrieved_at_utc"]

        frames.append(frame.set_index("ticker", append=True))

    if not frames:
        raise ValueError("No successful downloads to prepare.")

    prices = pd.concat(frames).sort_index()
    prices.attrs = {"quote_currency": "GBP", "share_basis": "Yahoo split-adjusted; adj_* also reflect distributions"}

    return prices

def audit_price_quality(prices):
    #flag suspucious data observations for further review
    prices = prices.sort_index()
    if not prices.index.is_unique:
        raise ValueError("Duplicate stock-date rows.")

    observed = prices["observed_row"].eq(True)
    columns = ["Open", "High", "Low", "Close", "adj_close"]
    values = prices[columns]

    flags = pd.DataFrame(index=prices.index)
    flags["observed_sessions"] = observed
    flags["missing_price"] = observed & values.isna().any(axis=1)
    flags["invalid_price"] = observed & (values.notna() & (~np.isfinite(values) | values.le(0))).any(axis=1)

    volume = prices["Volume"]
    flags["missing_volume"] = observed & volume.isna()
    flags["invalid_volume"] = (observed & volume.notna() & (~np.isfinite(volume) | volume.lt(0)))
    flags["zero_volume"] = observed & volume.eq(0)

    complete_ohlc = prices[["Open", "High", "Low", "Close"]].notna().all(axis=1)

    flags["ohlc_inconsistent"] = observed & complete_ohlc & (prices["Low"].gt(prices["High"]) | prices[["Open", "Close"]].max(axis=1).gt(prices["High"] + 1e-8) | prices[["Open", "Close"]].min(axis=1).lt(prices["Low"] - 1e-8))
    previous_close = prices.groupby(level="ticker")["adj_close"].shift(1)
    ratio = prices["adj_close"] / previous_close

    #diagnostic thresholds, not rules for removing genuine returns
    flags["large_adj_move"] = observed & (ratio.gt(2.0) | ratio.lt(0.5))

    flags["suspected_unit_jump"] = observed & (np.isclose(ratio, 100.0, rtol=0.02, atol=0) | np.isclose(ratio, 0.01, rtol=0.02, atol=0))
    summary = flags.groupby(level="ticker").sum().astype(int)
    summary["zero_volume_pct"] = 100 * summary["zero_volume"] / summary["observed_sessions"].replace(0, np.nan)

    return flags, summary