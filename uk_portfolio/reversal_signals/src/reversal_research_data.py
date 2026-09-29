from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import reversal_data


def configure_imports():
    # Support the repository's grouped layout without changing hash-pinned modules.
    project = Path(__file__).resolve().parents[1]
    portfolio = project.parent.parent
    if str(portfolio.parent) not in sys.path:
        sys.path.append(str(portfolio.parent))
    import trading_portfolio
    if str(project.parent) not in list(trading_portfolio.__path__):
        trading_portfolio.__path__ = [*trading_portfolio.__path__, str(project.parent)]
    for path in portfolio.glob("*/risk_management/src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


configure_imports()
from reversal_preparation import sha256, verify_frozen_source


def load_spec(project):
    project = Path(project)
    spec = json.loads((project / "data/uk_holdout_2021_2023_v1/frozen_spec.json").read_text())
    for name, digest in spec["source_sha256"].items():
        verify_frozen_source(project / "src" / name, digest)
    raw = project / "data/uk_2015_2021_v1/raw_panel_v1.pkl"
    if sha256(raw) != spec["development_raw_sha256"]:
        raise ValueError("Original development snapshot changed")
    return spec


def prepare_development(project):
    #rebuild archive's 2015-2021 cleaning and execution inputs

    project = Path(project)
    spec = load_spec(project)
    cache_dir = project / "data/uk_2015_2021_v1"
    uk_raw_panel = pd.read_pickle(cache_dir / "raw_panel_v1.pkl")
    cached_tickers = sorted(uk_raw_panel["ticker"].unique().tolist())
    uk_tickers = sorted(spec["tickers"])

    #use original saved session calendar
    full_schedule = pd.read_pickle(project / "data/uk_holdout_2021_2023_v1/lse_schedule_2015_2023.pkl")
    uk_schedule = full_schedule.loc[full_schedule.index < spec["periods"]["development_end_exclusive"]].copy()
    uk_panel = reversal_data.align_price_panel(uk_raw_panel, cached_tickers, uk_schedule.index)

    price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    unit_corrections = pd.read_csv(cache_dir / "proposed_unit_corrections.csv", parse_dates=["Date"])

    #start from unchanged aligned panel on each run
    uk_unit_panel = uk_panel.copy()
    uk_unit_panel["unit_correction_provisional"] = False

    for correction in unit_corrections.itertuples(index=False):
        mask = (uk_unit_panel["ticker"].eq(correction.ticker) & (uk_unit_panel.index == correction.Date))
        if mask.sum() != 1:
            raise ValueError(f"Expected one row for {correction.ticker}.")

        original_close = uk_unit_panel.loc[mask, "Close"].iloc[0]
        if not np.isclose(original_close, correction.expected_original_close):
            raise ValueError(f"Unexpected original price for {correction.ticker} on {correction.Date.date()}.")
        
        uk_unit_panel.loc[mask, price_columns] *= correction.price_multiplier
        uk_unit_panel.loc[mask, "unit_correction_provisional"] = True

    split_corrections = pd.read_csv(cache_dir / "proposed_split_corrections.csv", parse_dates=["Date"])

    #ROAD also needs listing gap handled seperately
    simple_splits = split_corrections.loc[split_corrections["ticker"].isin(["BCE.L", "PXEN.L"])]

    uk_split_panel = uk_unit_panel.copy()
    uk_split_panel["split_correction_provisional"] = False
    for correction in simple_splits.itertuples(index=False):
        ticker_mask = uk_split_panel["ticker"].eq(correction.ticker)
        event_mask = ticker_mask & (uk_split_panel.index == correction.Date)

        if event_mask.sum() != 1:
            raise ValueError(f"Expected one event row for {correction.ticker}.")

        if correction.scope == "event_row":
            affected = event_mask
            uk_split_panel.loc[affected, price_columns] *= correction.factor
        elif correction.scope == "earlier_history":
            affected = ticker_mask & (uk_split_panel.index < correction.Date)
            uk_split_panel.loc[affected, price_columns + ["Dividends"]] *= correction.factor
            uk_split_panel.loc[affected, "Volume"] /= correction.factor
        else:
            raise ValueError(f"Unknown correction scope: {correction.scope}")

        uk_split_panel.loc[event_mask, "Stock Splits"] = 1 / correction.factor
        uk_split_panel.loc[affected | event_mask, "split_correction_provisional"] = True

    event_rows = [(row.Date, row.ticker) for row in simple_splits.itertuples(index=False)]

    uk_clean_panel = uk_split_panel.copy()

    road = uk_clean_panel["ticker"].eq("ROAD.L")
    dates = uk_clean_panel.index
    road_split_date = pd.Timestamp("2018-06-25")

    before_split = road & (dates < road_split_date)
    split_day = road & (dates == road_split_date)
    if split_day.sum() != 1:
        raise ValueError("Expected one ROAD consolidation row.")


    #put earlier observations on post-consolidation share basis
    uk_clean_panel.loc[before_split, price_columns + ["Dividends"]] *= 33
    uk_clean_panel.loc[before_split, "Volume"] /= 33
    uk_clean_panel.loc[split_day, "Stock Splits"] = 1 / 33
    uk_clean_panel.loc[before_split | split_day, "split_correction_provisional"] = True

    #distinguish unlisted period from trading on another venue
    road_unlisted = (road & (dates >= pd.Timestamp("2018-01-22")) & (dates < pd.Timestamp("2018-06-26")))
    road_nex = (road & (dates >= pd.Timestamp("2018-06-26")) & (dates < pd.Timestamp("2020-01-07")))

    uk_clean_panel["known_unlisted"] = road_unlisted
    uk_clean_panel["known_nex_listing"] = road_nex

    #exclude dates from LSE/AIM analysis without deleting rows
    uk_clean_panel.loc[road_unlisted | road_nex, price_columns + ["Volume"]] = np.nan

    uk_prices = reversal_data.add_adjusted_prices(uk_clean_panel)

    quote_metadata = pd.read_csv(cache_dir / "quote_metadata_report.csv", dtype=str, keep_default_na=False).set_index("ticker", verify_integrity=True).reindex(cached_tickers)
    gbp_scale = quote_metadata["currency"].map({"GBp": 0.01, "GBX": 0.01, "GBP": 1.0})
    if quote_metadata["status"].ne("ok").any() or gbp_scale.isna().any():
        raise ValueError("Missing or unsupported price-currency metadata.")

    uk_activity = uk_prices[["ticker", "Close", "Volume"]].copy()
    uk_activity["traded_value_proxy_gbp"] = uk_activity["Close"] * uk_activity["Volume"] * uk_activity["ticker"].map(gbp_scale)

    #unknown volume stays unknown, not defaulted to zero
    uk_activity["reported_zero_volume"] = uk_activity["Volume"].eq(0).astype(float).where(uk_activity["Volume"].notna())

    uk_known_suspensions = pd.DataFrame([("PREM.L", "2017-07-03", "2017-07-12"),
            ("SQZ.L", "2017-11-21", "2017-11-30"),
            ("EAH.L", "2021-01-04", "2021-02-04")], 
            columns=["ticker", "start", "resume"])

    for column in ["start", "resume"]:
        uk_known_suspensions[column] = pd.to_datetime(uk_known_suspensions[column])

    uk_execution_market = uk_prices[["ticker", "Open", "Close", "Dividends", "Volume"]].copy()

    #convert monetary fields into GBP
    scales = uk_execution_market["ticker"].map(gbp_scale)

    uk_execution_market[["Open", "Close", "Dividends"]] = uk_execution_market[["Open", "Close", "Dividends"]].mul(scales, axis=0)
    uk_execution_market = uk_execution_market.rename(columns={"Open": "open_gbp", "Close": "close_gbp", "Dividends": "dividend_gbp", "Volume": "reported_volume"})

    uk_execution_market["known_suspended"] = False
    for event in uk_known_suspensions.itertuples(index=False):
        affected = (uk_execution_market["ticker"].eq(event.ticker) & (uk_execution_market.index >= event.start) & (uk_execution_market.index < event.resume))
        uk_execution_market.loc[affected, "known_suspended"] = True

    uk_execution_market["open_reference_usable"] = (np.isfinite(uk_execution_market["open_gbp"]) & uk_execution_market["open_gbp"].gt(0) & ~uk_execution_market["known_suspended"])
    uk_execution_market["positive_reported_volume"] = (np.isfinite(uk_execution_market["reported_volume"]) & uk_execution_market["reported_volume"].gt(0))
    uk_execution_market = uk_execution_market.set_index("ticker", append=True).sort_index()

    return {"project": project, "spec": spec, "vintage": "original-development",
            "raw": uk_raw_panel, "clean": uk_clean_panel, "prices": uk_prices,
            "schedule": uk_schedule, "scale": gbp_scale, "market": uk_execution_market,
            "suspensions": uk_known_suspensions}


def prepare_validation(project, development):
    #rebuild original 2022-2023 overlap, share basis and suspended quotes

    project = Path(project)
    uk_frozen_spec = load_spec(project)
    holdout_cache_dir = project / "data/uk_holdout_2021_2023_v1"
    uk_holdout_raw_panel = pd.read_pickle(holdout_cache_dir / "raw_panel_with_2021_overlap.pkl")
    uk_full_schedule = pd.read_pickle(holdout_cache_dir / "lse_schedule_2015_2023.pkl")
    uk_raw_panel, uk_clean_panel = development["raw"], development["clean"]

    gbp_scale = development["scale"]
    cached_tickers = sorted(gbp_scale.index)
    old_raw = uk_raw_panel.set_index("ticker", append=True).sort_index()
    new_raw = uk_holdout_raw_panel.set_index("ticker", append=True).sort_index()
    common = old_raw.index.intersection(new_raw.index)

    adjustment_ratios = (old_raw.loc[common, "Adj Close"] / new_raw.loc[common, "Adj Close"]).dropna()
    adjustment_scales = adjustment_ratios.groupby(level="ticker").median()

    if not np.allclose(adjustment_ratios / adjustment_ratios.index.get_level_values("ticker").map(adjustment_scales), 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("Overlap differences require more than constant rebasing.")

    uk_future_panel = uk_holdout_raw_panel.loc[uk_holdout_raw_panel.index >= pd.Timestamp(uk_frozen_spec["periods"]["holdout_start"])].copy()
    uk_future_panel["Adj Close"] *= (uk_future_panel["ticker"].map(adjustment_scales).fillna(1.0))

    #keep FORG on the same synthetic share basis as the frozen history.
    forg_event = (uk_future_panel["ticker"].eq("FORG.L") & (uk_future_panel.index == pd.Timestamp("2023-12-21")))
    if forg_event.sum() != 1 or not np.isclose(
        uk_future_panel.loc[forg_event, "Stock Splits"].iloc[0], 0.1):
        raise ValueError("FORG's cached consolidation record has changed.")

    forg_after = (uk_future_panel["ticker"].eq("FORG.L") & (uk_future_panel.index >= pd.Timestamp("2023-12-21")))
    uk_future_panel["Volume"] = uk_future_panel["Volume"].astype(float)
    uk_future_panel.loc[forg_after, ["Open", "High", "Low", "Close", "Adj Close", "Dividends"]] *= 0.1
    uk_future_panel.loc[forg_after, "Volume"] /= 0.1
    uk_future_panel.loc[forg_event, "Stock Splits"] = 0.01

    uk_full_tickers = sorted(set(cached_tickers) | set(uk_future_panel["ticker"]))
    new_tickers = sorted(set(uk_full_tickers) - set(gbp_scale.index))

    metadata_path = holdout_cache_dir / "additional_quote_metadata.pkl"
    additional_metadata = pd.read_pickle(metadata_path)

    additional_metadata = additional_metadata.reindex(new_tickers)
    additional_scales = additional_metadata["currency"].map({"GBp": 0.01, "GBX": 0.01, "GBP": 1.0})
    if (additional_metadata["fetch_status"].ne("ok").any() or additional_scales.isna().any()):
        raise ValueError("Resolve missing quote currencies before running.")

    uk_full_gbp_scale = pd.concat([gbp_scale, additional_scales])
    uk_full_clean_panel = reversal_data.align_price_panel(pd.concat([uk_clean_panel[uk_holdout_raw_panel.columns],
        uk_future_panel]).sort_index(), uk_full_tickers, uk_full_schedule.index)
    
    uk_full_prices = reversal_data.add_adjusted_prices(uk_full_clean_panel)

    uk_full_keyed_prices = uk_full_prices.set_index("ticker", append=True).sort_index()
    uk_known_suspensions = development["suspensions"]
    uk_full_market = uk_full_keyed_prices[["Open", "Close", "Dividends", "Volume"]].rename(columns={
        "Open": "open_gbp",
        "Close": "close_gbp",
        "Dividends": "dividend_gbp",
        "Volume": "reported_volume"
    })
    money_columns = ["open_gbp", "close_gbp", "dividend_gbp"]
    uk_full_market[money_columns] = uk_full_market[money_columns].mul(uk_full_market.index.get_level_values("ticker").map(uk_full_gbp_scale), axis=0)

    uk_full_suspensions = uk_known_suspensions.copy()
    uk_full_suspensions.loc[len(uk_full_suspensions)] = ["REVB.L", pd.Timestamp("2022-09-01"), pd.Timestamp("2023-06-28")]

    dates = uk_full_market.index.get_level_values("Date")
    tickers = uk_full_market.index.get_level_values("ticker")
    uk_full_market["known_suspended"] = False
    for event in uk_full_suspensions.itertuples(index=False):
        affected = ((tickers == event.ticker) & (dates >= event.start) & (dates < event.resume))
        uk_full_market.loc[affected, "known_suspended"] = True

    uk_full_market["open_reference_usable"] = (np.isfinite(uk_full_market["open_gbp"]) & uk_full_market["open_gbp"].gt(0) & ~uk_full_market["known_suspended"])
    uk_full_market["positive_reported_volume"] = (np.isfinite(uk_full_market["reported_volume"]) & uk_full_market["reported_volume"].gt(0))

    pd.testing.assert_frame_equal(uk_full_market.loc[development["market"].index], development["market"])
    return {"project": project, "spec": uk_frozen_spec, "vintage": "original-validation",
            "prices": uk_full_prices, "schedule": uk_full_schedule,
            "scale": uk_full_gbp_scale, "market": uk_full_market}
