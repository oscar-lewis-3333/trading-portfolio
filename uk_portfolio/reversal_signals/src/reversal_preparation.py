
#preparing everything for the 2024-2025 period
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import reversal_data
from trading_portfolio.reversal_signals.tests.reversal_provenance import verify_frozen_source

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
MONEY_COLUMNS = PRICE_COLUMNS + ["Dividends"]
DATA_COLUMNS = MONEY_COLUMNS + ["Volume", "Stock Splits"]
UNIT_CORRECTIONS = ["CHL.L", "DISH.L", "ELEG.L", "GCG.L", "HSM.L", "PCH.L", "SAVE.L"]
FULL_HISTORY_REPAIRS = ["BMTO.L", "CMX.L", "CRTX.L", "HVTA.L", "LSC.L", "NAR.L",
                        "PNS.L", "PPHC.L", "SOLI.L", "SPSC.L", "HSM.L", "CHL.L",
                        "DISH.L", "ELEG.L", "GCG.L", "PCH.L", "SAVE.L", "TIN.L"]
DIAGNOSTIC_HISTORIES = ["TRB.L", "IES.L", "AURA.L", "RFG.L", "AST.L"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_schedule(start, end_exclusive):
    schedule = mcal.get_calendar("LSE").schedule(
        start_date=start, end_date=pd.Timestamp(end_exclusive) - pd.Timedelta(days=1))
    missing = pd.Timestamp("2022-05-30")
    if pd.Timestamp(start) <= missing < pd.Timestamp(end_exclusive) and missing not in schedule.index:
        row = pd.DataFrame({
            "market_open": [pd.Timestamp("2022-05-30 08:00", tz="Europe/London").tz_convert("UTC")],
            "market_close": [pd.Timestamp("2022-05-30 16:30", tz="Europe/London").tz_convert("UTC")],
        }, index=pd.DatetimeIndex([missing], name="Date"))
        schedule = pd.concat([schedule, row]).sort_index()
    schedule.index.name = "Date"
    return schedule


def _constant_ratio(numerator, denominator, context):
    valid = numerator.gt(0) & denominator.gt(0) & np.isfinite(numerator) & np.isfinite(denominator)
    ratio = (numerator / denominator).loc[valid]
    if ratio.empty:
        raise ValueError(f"No comparable positive prices: {context}")
    factor = float(ratio.median())
    if not np.allclose(ratio / factor, 1., rtol=1e-5, atol=1e-8):
        raise ValueError(f"Nonconstant price scale: {context}")
    return factor


def apply_suspensions(panel, events):
    """Retain valid opening quotes on days suspended only after the open.

    `known_suspended` means suspended at the opening auction, matching the
    frozen execution engine. Closing quotes are unavailable from the actual
    suspension date. Reported volume and cash dividends remain as provenance.
    """
    out = panel.copy()
    dates = out.index.get_level_values("Date")
    tickers = out.index.get_level_values("ticker")
    if "known_suspended" not in out:
        out["known_suspended"] = False
    out["close_suspended"] = out["known_suspended"].fillna(False).astype(bool)
    for event in events:
        start = pd.Timestamp(event["start"])
        first_open = pd.Timestamp(event.get("first_suspended_open", event["start"]))
        end = pd.Timestamp(event["resume"]) if event.get("resume") else None
        base = tickers == event["ticker"]
        if end is not None:
            base &= dates < end
        at_close = base & (dates >= start)
        at_open = base & (dates >= first_open)
        out.loc[at_open, "known_suspended"] = True
        out.loc[at_close, "close_suspended"] = True
        out.loc[at_close, ["High", "Low", "Close", "Adj Close"]] = np.nan
        out.loc[at_open, "Open"] = np.nan
    out["known_suspended"] = out["known_suspended"].fillna(False).astype(bool)
    out["vendor_volume_during_suspension"] = out["known_suspended"] & out["Volume"].gt(0)
    # Previously confirmed suspensions must also have unusable references.
    out.loc[out["known_suspended"], PRICE_COLUMNS] = np.nan
    return out


def _audit(panel, schedule, declared):
    values = panel[PRICE_COLUMNS]
    invalid = values.notna() & (~np.isfinite(values) | values.le(0))
    if invalid.any().any() or panel["Volume"].dropna().lt(0).any():
        raise ValueError("Nonpositive/nonfinite observed prices or negative volume")
    if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("Stock-session keys must be unique and sorted")
    if not values.loc[panel.known_suspended].isna().all().all():
        raise ValueError("A confirmed full-session suspension still has usable prices")
    dates = panel.index.get_level_values("Date")
    if not dates.isin(schedule.index).all():
        raise ValueError("Price date outside the LSE calendar")
    observed = panel.loc[panel.observed_row].reset_index("ticker")
    coverage = reversal_data.audit_price_coverage(observed, declared, schedule.index)
    complete = panel[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    outside = complete & ((panel[["Open", "Close"]].max(axis=1) > panel.High + 1e-8)
                          | (panel[["Open", "Close"]].min(axis=1) < panel.Low - 1e-8))
    price_audit = panel.loc[outside, ["Open", "High", "Low", "Close", "Volume"]].copy()
    # These are diagnostics, not retrospective exclusion rules or winsorisation.
    close = panel.Close.unstack("ticker")
    ratio = close / close.shift(1)
    extreme = ratio.where(ratio.gt(5) | ratio.lt(.2)).stack().rename("close_ratio").to_frame()
    return coverage, price_audit, extreme


def prepare_uk_extension(project_dir=None, *, rebuild=False):
    """Return a cached, audited 2015--2025 dataset ready for feature building.

    Cached inputs are required. No network calls, signal calculations or
    portfolio simulations occur. Changed inputs invalidate the prepared cache.
    """
    project = Path(project_dir) if project_dir is not None else Path(__file__).resolve().parents[1]
    extension = project / "data/uk_extension_2023_2025_v1"
    holdout = project / "data/uk_holdout_2021_2023_v1"
    spec = json.loads((extension / "extension_spec.json").read_text())
    frozen = spec["frozen_strategy"]
    if sha256(holdout / "frozen_spec.json") != spec["original_spec_sha256"]:
        raise ValueError("Original frozen specification changed")
    for name, digest in frozen["source_sha256"].items():
        verify_frozen_source(project / "src" / name, digest)
    inputs = [extension / "historical_input_gbp_v1.pkl", extension / "candidate_panel_gbp_v2.pkl",
              extension / "extension_spec.json", extension / "suspension_events_v1.json",
              extension / "raw_panel_with_2023_overlap.pkl",
              project / "data/uk_2015_2021_v1/raw_panel_v1.pkl",
              holdout / "raw_panel_with_2021_overlap.pkl", holdout / "lse_schedule_2015_2023.pkl"]
    inputs += [extension / "reconciliation_sources_v1" / f"{t}.pkl"
               for t in FULL_HISTORY_REPAIRS + DIAGNOSTIC_HISTORIES]
    fingerprints = {str(p.relative_to(project)): sha256(p) for p in inputs}
    fingerprints["preparation_source"] = sha256(Path(__file__))
    output = extension / "prepared_2015_2025_v1.pkl"
    if output.exists() and not rebuild:
        cached = pd.read_pickle(output)
        if cached["manifest"]["input_sha256"] != fingerprints:
            raise ValueError("Preparation inputs changed; review and call rebuild=True")
        return cached

    history = pd.read_pickle(inputs[0]).copy()
    candidate = pd.read_pickle(inputs[1]).copy()
    historical_dates = history.index.get_level_values("Date")
    historical_tickers = history.index.get_level_values("ticker")
    correction_log = []
    for ticker in UNIT_CORRECTIONS:
        mask = historical_tickers == ticker
        history.loc[mask, PRICE_COLUMNS] *= 100.
        correction_log.append(dict(ticker=ticker, change="historical price-unit correction",
                                   factor=100., note="Cash dividends and share volumes were already correctly scaled."))
    tin = historical_tickers == "TIN.L"
    history.loc[tin, MONEY_COLUMNS] *= 10.
    history.loc[tin, "Volume"] /= 10.
    correction_log.append(dict(ticker="TIN.L", change="ten old shares per new synthetic share",
                               factor=10., note="Price x10 and volume /10 preserve traded value."))
    # RNS, 30 November 2015: 1-for-20, while Yahoo records 1-for-2.
    ast_event = (pd.Timestamp("2015-12-01"), "AST.L")
    if not np.isclose(history.loc[ast_event, "Stock Splits"], .5):
        raise ValueError("Ascent's reviewed source action changed")
    ast_before = (historical_tickers == "AST.L") & (historical_dates < ast_event[0])
    history.loc[ast_before, MONEY_COLUMNS] *= 10.
    history.loc[ast_before, "Volume"] /= 10.
    history.loc[ast_event, "Stock Splits"] = .05
    correction_log.append(dict(ticker="AST.L", change="2015 consolidation: 1-for-20, not vendor 1-for-2",
                               factor=10., note="Pre-event price x10 and volume /10; cash value preserved."))

    # Norish/Roebuck's documented capital return is absent from Yahoo. Recognise
    # entitlement on the ex-date, as the frozen engine does for distributions.
    # The exact total-return adjustment avoids an artificial reversal signal
    # after this unusually large distribution. It is not a trading-price change.
    rfg_event = (pd.Timestamp("2021-11-23"), "RFG.L")
    if history.loc[rfg_event, "Dividends"] != 0.:
        raise ValueError("Roebuck distribution is no longer missing in the source")
    history.loc[rfg_event, "Dividends"] = 1.66
    rfg_factor = history.loc[rfg_event, "Close"] / (history.loc[rfg_event, "Close"] + 1.66)
    history.loc[(historical_tickers == "RFG.L") & (historical_dates < rfg_event[0]), "Adj Close"] *= rfg_factor
    correction_log.append(dict(ticker="RFG.L", change="missing GBP 1.66 capital return, ex-date 2021-11-23",
                               factor=float(rfg_factor), note="Cash entitlement and exact total-return adjustment; cash OHLC unchanged."))

    # Full-length dividend-adjustment histories avoid splicing incompatible
    # adjustment factors at the overlap. Existing cash OHLC is retained.
    vendor_comparisons = []
    for ticker in FULL_HISTORY_REPAIRS:
        payload = pd.read_pickle(extension / "reconciliation_sources_v1" / f"{ticker}.pkl")
        repaired = payload["repaired"]
        old_candidate = candidate.xs(ticker, level="ticker")
        overlap = old_candidate.index.intersection(repaired.index)
        scale = _constant_ratio(repaired.loc[overlap, "Close"], old_candidate.loc[overlap, "Close"], ticker)
        if not np.isclose(scale, 1., rtol=1e-5):
            raise ValueError(f"Full-history repair is not on the reviewed GBP basis: {ticker}")
        adjustment = repaired["Adj Close"] / repaired["Close"]
        if (adjustment.dropna().le(0) | ~np.isfinite(adjustment.dropna())).any():
            raise ValueError(f"Invalid adjustment factors: {ticker}")
        for frame in [history, candidate]:
            mask = frame.index.get_level_values("ticker") == ticker
            reference_dates = frame.index.get_level_values("Date")[mask]
            factor = adjustment.reindex(reference_dates).to_numpy()
            observed_close = frame.loc[mask, "Close"].notna().to_numpy()
            if np.any(observed_close & ~np.isfinite(factor)):
                raise ValueError(f"Missing adjustment for an existing close: {ticker}")
            frame.loc[mask, "Adj Close"] = frame.loc[mask, "Close"].to_numpy() * factor
            # Refresh the small GBP dividend conversion revisions from one
            # frozen vendor vintage for these two foreign-currency payers.
            if ticker in ["PPHC.L", "SPSC.L"]:
                dividends = repaired["Dividends"].reindex(reference_dates).to_numpy()
                supplied = np.isfinite(dividends)
                existing = frame.loc[mask, "Dividends"].to_numpy().copy()
                existing[supplied] = dividends[supplied]
                frame.loc[mask, "Dividends"] = existing
        vendor_comparisons.append(dict(ticker=ticker, comparable_rows=len(overlap), price_scale=scale,
                                       metadata_currency=payload["repaired_currency"], output_currency="GBP"))

    # Preserve fractional synthetic volume rather than vendor rounding of the
    # 1-for-10 conversion. Corporate-action metadata is not applied a second time.
    raw = pd.read_pickle(extension / "raw_panel_with_2023_overlap.pkl").set_index("ticker", append=True)
    tin_index = candidate.index[(candidate.index.get_level_values("ticker") == "TIN.L")
                                & (candidate.index.get_level_values("Date") < pd.Timestamp("2025-12-18"))]
    candidate.loc[tin_index, "Volume"] = raw.loc[tin_index, "Volume"] / 10.

    before = history.loc[historical_dates < pd.Timestamp("2023-01-01"), DATA_COLUMNS + ["known_suspended"]]
    after = candidate[DATA_COLUMNS + ["known_suspended"]]
    joined = pd.concat([before, after]).sort_index()
    joined["data_vintage"] = np.where(joined.index.get_level_values("Date") < pd.Timestamp("2023-01-01"),
                                      "historical_reconciled", "extension_reconciled")
    if not joined.index.is_unique:
        raise ValueError("Overlapping rows in the history join")

    schedule = build_schedule(frozen["periods"]["development_start"], spec["evaluation_end_exclusive"])
    pd.testing.assert_frame_equal(schedule.loc[pd.read_pickle(holdout / "lse_schedule_2015_2023.pkl").index],
                                  pd.read_pickle(holdout / "lse_schedule_2015_2023.pkl"))
    available = sorted(joined.index.get_level_values("ticker").unique())
    declared = frozen["tickers"]
    if not set(available).issubset(declared):
        raise ValueError("Undeclared ticker in prepared prices")
    index = pd.MultiIndex.from_product([schedule.index, available], names=["Date", "ticker"])
    joined = joined.reindex(index)
    joined["known_suspended"] = joined["known_suspended"].astype("boolean").fillna(False).astype(bool)
    source_indices = []
    for path in [project / "data/uk_2015_2021_v1/raw_panel_v1.pkl",
                 holdout / "raw_panel_with_2021_overlap.pkl", extension / "raw_panel_with_2023_overlap.pkl"]:
        source = pd.read_pickle(path).set_index("ticker", append=True)
        source_indices.append(source.index)
    observed = source_indices[0].union(source_indices[1]).union(source_indices[2])
    joined["observed_row"] = joined.index.isin(observed)
    events = json.loads((extension / "suspension_events_v1.json").read_text())["events"]
    joined = apply_suspensions(joined, events)
    # Quarantine reviewed, unresolved quotes; never invent replacement prices.
    # Tribal's zero-volume 118.8p plateau contradicts the issuer's annual price
    # ranges and repeats an old quote. Invinity has a zero-volume, flat-OHLC
    # consolidation-day spike that neither raw nor repaired Yahoo resolves.
    dates = joined.index.get_level_values("Date")
    tickers = joined.index.get_level_values("ticker")
    trb = ((tickers == "TRB.L") & (dates >= "2016-01-01") & (dates < "2017-03-01")
           & joined.Volume.eq(0) & np.isclose(joined.Close, 1.188, rtol=1e-5))
    ies = (tickers == "IES.L") & (dates == pd.Timestamp("2020-04-02"))
    joined["unverified_quote"] = trb | ies
    quarantined = joined.loc[joined.unverified_quote, DATA_COLUMNS].copy()
    joined.loc[joined.unverified_quote, PRICE_COLUMNS] = np.nan
    # Yahoo records this action before its AIM effective date. The repaired
    # prices already use the new share basis; move metadata only.
    old_event, new_event = (pd.Timestamp("2025-05-08"), "88E.L"), (pd.Timestamp("2025-05-13"), "88E.L")
    if np.isclose(joined.loc[old_event, "Stock Splits"], .04):
        joined.loc[old_event, "Stock Splits"] = 0.
        joined.loc[new_event, "Stock Splits"] = .04
    aura_old, aura_new = (pd.Timestamp("2021-03-23"), "AURA.L"), (pd.Timestamp("2021-03-26"), "AURA.L")
    if np.isclose(joined.loc[aura_old, "Stock Splits"], 1 / 13):
        joined.loc[aura_old, "Stock Splits"] = 0.
        joined.loc[aura_new, "Stock Splits"] = 1 / 13
    joined.attrs = {"quote_currency": "GBP", "share_basis": "consistent synthetic shares",
                    "evaluation_start": spec["evaluation_start"], "evaluation_end_exclusive": spec["evaluation_end_exclusive"]}
    coverage, ohlc_flags, extremes = _audit(joined, schedule, declared)
    prices = reversal_data.add_adjusted_prices(joined.reset_index("ticker"))
    market = joined[["Open", "Close", "Dividends", "Volume", "known_suspended"]].rename(columns={
        "Open": "open_gbp", "Close": "close_gbp", "Dividends": "dividend_gbp", "Volume": "reported_volume"})
    market["open_reference_usable"] = market.open_gbp.gt(0) & np.isfinite(market.open_gbp) & ~market.known_suspended
    market["positive_reported_volume"] = market.reported_volume.gt(0) & np.isfinite(market.reported_volume)
    summary = dict(declared_tickers=len(declared), available_tickers=len(available),
                   unavailable_tickers=len(set(declared) - set(available)), sessions=len(schedule),
                   evaluation_sessions=int(((schedule.index >= spec["evaluation_start"]) &
                                            (schedule.index < spec["evaluation_end_exclusive"])).sum()),
                   rows=len(joined), suspended_rows=int(joined.known_suspended.sum()),
                   quarantined_quotes=len(quarantined),
                   observed_rows=int(joined.observed_row.sum()), ohlc_consistency_flags=len(ohlc_flags),
                   extreme_close_ratios=len(extremes), stage="ready_for_features")
    manifest = dict(input_sha256=fingerprints, summary=summary, frozen_strategy=frozen,
                    built_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                    limitations=["Present-day catalogue, not historical point-in-time membership.",
                                 "Yahoo daily price/volume and dividend proxies; no historical bid/ask or auction fills.",
                                 "Suspension register includes documented events; zero volume alone is not a suspension.",
                                 "Original reported results are preserved and are not corrected-data performance results."])
    result = dict(panel=joined, prices=prices, market=market, schedule=schedule, coverage=coverage,
                  corrections=pd.DataFrame(correction_log), vendor_comparisons=pd.DataFrame(vendor_comparisons),
                  suspensions=pd.DataFrame(events), quarantined_quotes=quarantined,
                  ohlc_flags=ohlc_flags, extreme_moves=extremes,
                  manifest=manifest)
    temporary = output.with_suffix(".pkl.tmp")
    pd.to_pickle(result, temporary)
    temporary.replace(output)
    (extension / "prepared_manifest_v1.json").write_text(json.dumps(manifest, indent=2) + "\n")
    coverage.to_csv(extension / "prepared_coverage_v1.csv")
    extremes.to_csv(extension / "prepared_extreme_moves_v1.csv")
    return result
