from pathlib import Path

import pandas as pd

from preparation import load_saved_daily_prices, audit_daily_prices
from eligibility import add_research_eligibility


def load_reversal_history(project_root, protocol):
    #read the saved snapshot. preserve gaps, reject errors.
    snapshot = (Path(project_root) / protocol["history_snapshot"]).resolve()
    prices, spec = load_saved_daily_prices(snapshot)
    expected = {
        "source": "coinbase_exchange",
        "quote_currency": protocol["quote_currency"],
        "start": protocol["history_start"],
        "end_exclusive": protocol["history_end_exclusive"]
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved history does not match the declared protocol.")
    start = pd.Timestamp(protocol["history_start"], tz="UTC")
    end = pd.Timestamp(protocol["history_end_exclusive"], tz="UTC")
    
    if not prices["timestamp"].between(start, end, inclusive="left").all():
        raise ValueError("A candle falls outside the declared history window.")
    
    audit, missing_days = audit_daily_prices(prices)
    invalid = ["duplicates", "off_midnight", "nonfinite_rows", "nonpositive_price_rows", "invalid_ohlc_rows", "negative_volume_rows"]
    if audit[invalid].ne(0).any().any():
        raise ValueError("Resolve invalid daily candles before building membership.")
    return prices, audit, missing_days


def build_daily_universe(prices, protocol):

    #return screened history, active members and every development decision
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    if start >= end:
        raise ValueError("The development window must have positive length.")
    
    maximum = protocol["universe"]["max_assets"]
    minimum = protocol["universe"]["min_assets"]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (minimum, maximum)):
        raise ValueError("Universe limits must be integers.")
    if not 1 <= minimum <= maximum:
        raise ValueError("Require 1 <= min_assets <= max_assets.")

    #retain warmup history. later-period candles cannot affect this stage
    panel = add_research_eligibility(prices.loc[prices["timestamp"].lt(end)].copy(), **protocol["eligibility"])

    panel["decision_at"] = panel["signal_available_at"]
    panel["meme_excluded"] = panel["product_id"].isin(protocol["excluded_products"])
    panel["universe_candidate"] = panel["research_candidate"] & ~panel["meme_excluded"]
    decisions = panel.loc[panel["decision_at"].ge(start) & panel["decision_at"].lt(end)]

    candidates = decisions.loc[decisions["universe_candidate"]].copy()
    candidates = candidates.sort_values(["decision_at", "median_dollar_volume", "product_id"], ascending=[True, False, True])
    candidates["liquidity_rank"] = candidates.groupby("decision_at").cumcount() + 1

    #include dates with no rows or insufficient breadth in audit
    calendar = pd.date_range(start, end, freq="D", inclusive="left", name="decision_at")
    coverage = pd.DataFrame(index=calendar)
    coverage["history_ready"] = decisions.groupby("decision_at")["history_ready"].sum()
    coverage["research_candidates"] = decisions.groupby("decision_at")["research_candidate"].sum()
    coverage["after_meme_exclusions"] = candidates.groupby("decision_at").size()
    coverage = coverage.fillna(0).astype("int64")
    coverage["capped_candidates"] = coverage["after_meme_exclusions"].clip(upper=maximum)
    coverage["active"] = coverage["capped_candidates"].ge(minimum)
    coverage["universe_size"] = coverage["capped_candidates"].where(coverage["active"], 0)

    active_dates = coverage.index[coverage["active"]]
    members = candidates.loc[candidates["liquidity_rank"].le(maximum) & candidates["decision_at"].isin(active_dates)].reset_index(drop=True)
    return panel, members, coverage
