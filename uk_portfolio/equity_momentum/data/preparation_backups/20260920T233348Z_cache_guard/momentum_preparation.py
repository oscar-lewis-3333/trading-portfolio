#offline preparation of the frozen UK momentum price dataset



from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

import momentum_data


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
ADJUSTED_COLUMNS = ["adj_open", "adj_high", "adj_low", "adj_close"]
BOOLEAN_COLUMNS = ["observed_row", "known_suspended", "close_suspended", "unverified_quote"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_price_panel(prices, schedule):
    #reject malformed keys/numbers
    if prices.attrs.get("quote_currency") != "GBP":
        raise ValueError("Prices must already be expressed in GBP.")
    if prices.index.names != ["Date", "ticker"] or not prices.index.is_unique:
        raise ValueError("Expected unique (Date, ticker) keys.")
        
    sessions = pd.DatetimeIndex(schedule.index)
    if (not sessions.is_unique or not sessions.is_monotonic_increasing or sessions.tz is not None or not sessions.equals(sessions.normalize())):
        raise ValueError("Expected unique, sorted, timezone-naive exchange sessions.")
    
    symbols = sorted(prices.index.get_level_values("ticker").unique())
    expected = pd.MultiIndex.from_product([sessions, symbols], names=["Date", "ticker"])
    if not prices.index.equals(expected):
        raise ValueError("Every downloaded ticker must retain the full exchange calendar.")
    
    values = prices[PRICE_COLUMNS + ADJUSTED_COLUMNS]
    if (values.notna() & (~np.isfinite(values) | values.le(0))).any().any():
        raise ValueError("Nonpositive or nonfinite prices require explicit review.")
    
    for column in ["Volume", "Dividends", "Stock Splits"]:
        values = prices[column]
        if (values.notna() & (~np.isfinite(values) | values.lt(0))).any():
            raise ValueError(f"Invalid {column} values.")
    if prices.loc[~prices.observed_row.eq(True), PRICE_COLUMNS].notna().any().any():
        raise ValueError("Prices were introduced on an unobserved session.")


def apply_data_review(prices, review):
    #apply dated blocks without replacing prices
    out = prices.copy()
    for column in BOOLEAN_COLUMNS:
        out[column] = out[column].eq(True) if column in out else False
    out["unit_quarantine"] = False
    out["unsupported_distribution"] = False
    dates = out.index.get_level_values("Date")
    tickers = out.index.get_level_values("ticker")
    flags, _ = momentum_data.audit_price_quality(out)

    def period(event, start_key="start", end_key="end_exclusive"):
        if event["ticker"] not in tickers:
            raise ValueError(f"Reviewed ticker absent: {event['ticker']}")
        
        mask = (tickers == event["ticker"]) & (dates >= pd.Timestamp(event[start_key]))
        if event.get(end_key):
            if pd.Timestamp(event[end_key]) <= pd.Timestamp(event[start_key]):
                raise ValueError("Review interval ends before it starts.")
            
            mask &= dates < pd.Timestamp(event[end_key])
        return mask

    for event in review["unit_quarantines"]:
        if pd.Timestamp(event["start"]) > dates.max():
            continue

        key = (pd.Timestamp(event["start"]), event["ticker"])
        if (key not in flags.index or not flags.loc[key, "suspected_unit_jump"] or out.loc[key, "Volume"] != 0 or out.loc[key, "Stock Splits"] != 0):
            raise ValueError(f"Reviewed unit-jump evidence changed: {key}")
        
        out.loc[period(event), "unit_quarantine"] = True

    for event in review["suspensions"]:
        out.loc[period(event, "start", "resume"), "close_suspended"] = True
        out.loc[period(event, "first_suspended_open", "resume"), "known_suspended"] = True

    for event in review["unsupported_distributions"]:
        out.loc[period(event), "unsupported_distribution"] = True

    out["unverified_quote"] |= out.unit_quarantine | out.unsupported_distribution
    blocked_open = out.known_suspended | out.unverified_quote
    blocked_close = out.close_suspended | out.unverified_quote
    changed = ((blocked_open & out.Open.notna()) | (blocked_close & out.Close.notna()))
    quarantined = prices.loc[changed, PRICE_COLUMNS + ["Volume", "Dividends", "Stock Splits"]].copy()

    for column in ["unit_quarantine", "known_suspended", "close_suspended", "unsupported_distribution"]:
        quarantined[column] = out.loc[changed, column]
    out.loc[blocked_open, ["Open", "adj_open"]] = np.nan
    out.loc[blocked_close, ["High", "Low", "Close", "Adj Close", "adj_high", "adj_low", "adj_close", "adjustment_factor"]] = np.nan

    #reported distributions and volume remain provenance
    out["vendor_volume_during_suspension"] = out.known_suspended & out.Volume.gt(0)
    out["open_reference_usable"] = out.observed_row & out.Open.gt(0) & ~blocked_open
    out["close_reference_usable"] = out.observed_row & out.Close.gt(0) & out.adj_close.gt(0) & ~blocked_close
    out["positive_reported_volume"] = out.observed_row & out.Volume.gt(0)
    out["traded_value_gbp"] = (out.Close * out.Volume).where(out.close_reference_usable)

    #this count uses the current and previous sessions only
    zero = (out.observed_row & out.Volume.eq(0)).unstack("ticker")
    positions = pd.DataFrame(np.broadcast_to(np.arange(len(zero))[:, None], zero.shape),
                             index=zero.index, columns=zero.columns)
    
    runs = positions - positions.where(~zero).ffill().fillna(-1)
    out["zero_volume_run"] = runs.stack().reindex(out.index).astype("int32")
    out["ohlc_inconsistent"] = flags.ohlc_inconsistent
    out["large_adj_move"] = flags.large_adj_move
    out.attrs = prices.attrs.copy()
    return out, quarantined


def build_coverage(prices, universe, failures):
    #report gaps seperately from pre-history and documented unavailable quotes
    rows = []
    for ticker, group in prices.groupby(level="ticker", sort=True):
        group = group.droplevel("ticker")
        observed = group.observed_row
        observed_dates = group.index[observed]
        valid_dates = group.index[group.close_reference_usable]

        first = observed_dates.min()
        last = observed_dates.max()
        internal = group.index.to_series().between(first, last)
        gaps = ~observed & internal
        runs = gaps.groupby(gaps.ne(gaps.shift()).cumsum()).sum()
        rows.append(dict(ticker=ticker, source=str(group.source.iloc[0]),
            observed_sessions=int(observed.sum()), usable_close_sessions=len(valid_dates),
            first_observed=first, last_observed=last,
            first_usable_close=valid_dates.min(), last_usable_close=valid_dates.max(),
            sessions_before_history=int((group.index < first).sum()),
            sessions_after_history=int((group.index > last).sum()),
            internal_missing_sessions=int(gaps.sum()), longest_internal_gap=int(runs.max()),
            suspended_sessions=int(group.known_suspended.sum()),
            unit_quarantine_sessions=int(group.unit_quarantine.sum()),
            unverified_quote_sessions=int(group.unverified_quote.sum()),
            zero_volume_pct=float(100 * (observed & group.Volume.eq(0)).sum() / max(observed.sum(), 1)),
            ohlc_inconsistent_sessions=int(group.ohlc_inconsistent.sum())
        ))

    summary = pd.DataFrame(rows).set_index("ticker")
    coverage = universe[["yf_symbol", "isin", "name"]].merge(summary, left_on="yf_symbol", right_index=True, how="left", validate="one_to_one")
    coverage["has_prices"] = coverage.observed_sessions.fillna(0).gt(0)
    coverage["source"] = coverage.source.fillna("unavailable")
    coverage["unavailable_reason"] = coverage.yf_symbol.map(failures).fillna("")
    if (coverage.loc[~coverage.has_prices, "unavailable_reason"] == "").any():
        raise ValueError("Unavailable candidates need a recorded download outcome.")
    
    count_columns = [c for c in coverage if c.endswith("sessions") or c in [
        "sessions_before_history", "sessions_after_history", "longest_internal_gap"]]
    coverage[count_columns] = coverage[count_columns].fillna(0).astype(int)
    return coverage


def build_market_inputs(prices):
    #unadjusted distrubutions references on sources split-adjusted basis

    columns = ["Open", "Close", "Dividends", "Volume", "known_suspended",
               "close_suspended", "unverified_quote", "open_reference_usable",
               "close_reference_usable", "positive_reported_volume", "traded_value_gbp"]
    market = prices[columns].rename(columns={
        "Open": "open_gbp", "Close": "close_gbp", "Dividends": "dividend_gbp",
        "Volume": "reported_volume"})
    market.attrs = prices.attrs.copy()
    return market


def _check_ready(prices, schedule):
    validate_price_panel(prices, schedule)
    for column, flag in [("Open", "known_suspended"), ("Close", "close_suspended")]:
        if prices.loc[prices[flag] | prices.unverified_quote, column].notna().any():
            raise ValueError("Blocked quotes remain usable.")
        
    flags, summary = momentum_data.audit_price_quality(prices)
    if flags.suspected_unit_jump.any():
        raise ValueError("Unreviewed 100-fold moves remain in prepared prices.")
    
    return summary


def load_prepared_momentum_data(project_dir=None):
    #load and verify local snapshot without downloading or reading reversal data
    project = Path(project_dir).resolve() if project_dir else Path(__file__).resolve().parents[1]
    folder = project / "data/prepared_2015_2025_v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    for relative, expected in manifest["local_sha256"].items():
        if sha256(project / relative) != expected:

            raise ValueError(f"Preparation inputs changed: {relative}; review before rebuilding.")
    if sha256(folder / "dataset.pkl") != manifest["dataset_sha256"]:
        raise ValueError("Prepared dataset fingerprint differs from its manifest.")
    
    result = pd.read_pickle(folder / "dataset.pkl")
    result["manifest"] = manifest
    result["market"] = build_market_inputs(result["prices"])
    return result


def prepare_momentum_data(project_dir=None, *, rebuild=False):
    #build once from trusted local caches
    project = Path(project_dir).resolve() if project_dir else Path(__file__).resolve().parents[1]
    folder = project / "data/prepared_2015_2025_v1"
    if (folder / "manifest.json").exists() and not rebuild:
        return load_prepared_momentum_data(project)

    universe_path = project / "data/candidate_universe_v1.csv"
    review_path = project / "data/data_review_v1.json"
    source_project = project.parent / "reversal_signals"
    source_path = source_project / "data/uk_extension_2023_2025_v1/prepared_2015_2025_v1.pkl"
    identity_path = source_project / "data/uk_2015_2021_v1/universe.csv"
    raw_dir = project / "data/raw_yahoo_2015_2025_v1"
    universe = pd.read_csv(universe_path, dtype="string", keep_default_na=False)
    for column in ["yf_symbol", "isin", "api_ticker"]:
        if universe[column].str.strip().eq("").any() or universe[column].duplicated().any():
            raise ValueError(f"Missing or duplicate candidate identifier: {column}")
        
    parent = pd.read_pickle(source_path)
    schedule = parent["schedule"].copy()
    source = parent["prices"]
    if source.attrs.get("quote_currency") != "GBP":
        raise ValueError("Inherited prices must already be in GBP.")
    
    old = source.loc[source.ticker.isin(universe.yf_symbol)].set_index("ticker", append=True).sort_index().copy()
    old["source"] = "inherited_reconciled"
    inherited_symbols = set(old.index.get_level_values("ticker"))
    old_identity = pd.read_csv(identity_path).set_index("yf_symbol")["isin"]
    for row in universe.itertuples():
        if row.yf_symbol in inherited_symbols and old_identity.get(row.yf_symbol) != row.isin:
            raise ValueError(f"Inherited ticker/ISIN mismatch: {row.yf_symbol}")
        
    missing = sorted(set(universe.yf_symbol) - inherited_symbols)
    failures, downloads = {}, []
    for ticker in missing:
        payload = pd.read_pickle(raw_dir / f"{ticker}.pkl")
        if (payload["symbol"] != ticker or payload["start"] != "2015-01-01" or payload["end"] != "2026-01-01"):
            raise ValueError(f"Unexpected cached request for {ticker}")
        
        if payload["status"] == "failed":
            failures[ticker] = payload["error"] or "Yahoo returned no usable history."
        elif payload["status"] == "downloaded":
            downloads.append(ticker)
        else:
            raise ValueError(f"Unknown cached download status: {ticker}")
        
    extra = momentum_data.prepare_downloaded_prices(downloads, raw_dir, schedule)
    extra["source"] = "additional_yahoo"
    prices = pd.concat([old, extra]).sort_index()
    prices.attrs = {"quote_currency": "GBP", "share_basis": "Per-ticker split-adjusted synthetic shares",
                    "adjusted_prices": "adj_* also include vendor distribution adjustments"}
    
    for column in BOOLEAN_COLUMNS:
        prices[column] = prices[column].eq(True) if column in prices else False
    validate_price_panel(prices, schedule)

    raw_flags, raw_audit = momentum_data.audit_price_quality(prices)
    review = json.loads(review_path.read_text())
    prepared, quarantined = apply_data_review(prices, review)
    cleaned_audit = _check_ready(prepared, schedule)
    coverage = build_coverage(prepared, universe, failures)

    diagnostics = prices.loc[raw_flags.large_adj_move | raw_flags.suspected_unit_jump,
                ["Open", "Close", "adj_close", "Volume", "Stock Splits"]].copy()
    diagnostics["suspected_unit_jump"] = raw_flags.suspected_unit_jump.reindex(diagnostics.index)
    diagnostics["quarantined"] = prepared.unverified_quote.reindex(diagnostics.index)
    for column in ["source", "data_vintage"]:
        prepared[column] = prepared[column].astype("category")

    limitations = ["Present-day candidate catalogue; historical broker membership and delisted-stock coverage are incomplete.",
        "593 price histories do not mean 593 eligible stocks at any date; history and trailing liquidity rules are still to be built.",
        "Recorded suspensions are not an exhaustive exchange register. False means no registered suspension, not verified trading access.",
        "Zero volume may reflect illiquidity or vendor omissions. OHLC inconsistencies remain diagnostic; do not simulate stops from high/low.",
        "No historical bid/ask, executable auction prices, or verified all-in transaction costs.",
        "Dated unit quarantines contain unresolved prices, not verified corrections. Held positions crossing gaps require explicit valuation/execution treatment.",
        "Source adjustments are retrospective vendor vintages, not a complete point-in-time corporate-action feed.",
        "SPDI's non-cash distribution is not valued; its affected period is blocked.",
        "No momentum features, strategy results, eligibility thresholds, or development/holdout split have been calculated here."]
    
    local_paths = [universe_path, review_path, project / "src/momentum_data.py", Path(__file__)]
    source_paths = [source_path, identity_path] + [raw_dir / f"{t}.pkl" for t in missing]
    manifest = {"built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "local_sha256": {str(p.relative_to(project)): sha256(p) for p in local_paths},
        "source_sha256": {str(p.resolve()): sha256(p) for p in source_paths},
        "summary": {"candidate_tickers": len(universe), "downloaded_tickers": len(inherited_symbols) + len(downloads),
            "inherited_tickers": len(inherited_symbols), "additional_tickers": len(downloads),
            "unavailable_tickers": len(failures), "sessions": len(schedule), "rows": len(prepared),
            "first_session": str(schedule.index.min().date()), "last_session": str(schedule.index.max().date()),
            "observed_rows": int(prepared.observed_row.sum()),
            "usable_close_rows": int(prepared.close_reference_usable.sum()),
            "suspended_rows": int(prepared.known_suspended.sum()),
            "newly_blocked_observations": len(quarantined),
            "unit_quarantine_tickers": len(review["unit_quarantines"]),
            "unit_quarantine_rows": int(prepared.unit_quarantine.sum()),
            "remaining_unit_jump_flags": int(cleaned_audit.suspected_unit_jump.sum()),
            "stage": "ready_for_features_with_documented_data_limits",
        }, "policy": review["policy"], "limitations": limitations}
    result = dict(prices=prepared, schedule=schedule, universe=universe, coverage=coverage,
                  raw_audit=raw_audit, cleaned_audit=cleaned_audit, diagnostics=diagnostics,
                  quarantined_observations=quarantined, review=review,
                  inherited_corrections=parent["corrections"],
                  inherited_suspensions=parent["suspensions"],
                  inherited_quarantined_quotes=parent["quarantined_quotes"])
    
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "dataset.pkl.tmp"
    pd.to_pickle(result, temporary)
    temporary.replace(folder / "dataset.pkl")
    manifest["dataset_sha256"] = sha256(folder / "dataset.pkl")
    for name in ["coverage", "raw_audit", "cleaned_audit", "diagnostics", "quarantined_observations"]:
        result[name].to_csv(folder / f"{name}.csv", index=name != "coverage")
    
    temporary = folder / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(folder / "manifest.json")
    result["manifest"] = manifest
    result["market"] = build_market_inputs(prepared)
    return result
