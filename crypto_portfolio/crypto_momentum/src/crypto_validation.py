import pandas as pd
import math
import crypto_reporting
import hashlib
import json
from pathlib import Path


def build_annual_evaluation_windows(data_start, first_test_start, evaluation_end):

    #build expanding calibration windows and year long test windows
    dates = []
    for value in (data_start, first_test_start, evaluation_end):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError("Dates must be timezone-aware.")

        timestamp = timestamp.tz_convert("UTC")
        if (timestamp != timestamp.normalize() or timestamp.month != 1 or timestamp.day != 1):
            raise ValueError("Dates must be January 1 at midnight UTC.")
        dates.append(timestamp)

    data_start, first_test_start, evaluation_end = dates
    if not data_start < first_test_start < evaluation_end:
        raise ValueError("Require data_start < first_test_start < evaluation_end.")
    rows = []
    test_start = first_test_start

    while test_start < evaluation_end:
        test_end = test_start + pd.DateOffset(years=1)
        rows.append({
            "fold": len(rows) + 1,
            "calibration_start": data_start,
            "calibration_end": test_start,
            "test_start": test_start,
            "test_end": test_end
        })

        test_start = test_end

    return pd.DataFrame(rows).set_index("fold")


def calibrate_benchmark_allocations(momentum_ledger, windows):

    #estimate benchmark weights using only fold's earlier data
    ledger = momentum_ledger.sort_index()

    if ledger.empty or ledger.index.has_duplicates:
        raise ValueError("Provide a nonempty ledger with unique dates.")

    rows = []

    for fold, window in windows.iterrows():
        start = window["calibration_start"]
        cutoff = window["calibration_end"]
        if cutoff > window["test_start"]:
            raise ValueError("Calibration overlaps the test period.")

        history = ledger.loc[(ledger.index >= start) & (ledger.index < cutoff)].copy()
        if history.empty:
            raise ValueError(f"Fold {fold} has no calibration observations.")

        #candle's closing valuation becomes available the next day
        if (history["valuation_at"].isna().any() or (history["valuation_at"] > cutoff).any()):
            raise ValueError(f"Fold {fold} uses an unavailable valuation.")

        expected_dates = pd.date_range(start=max(start, ledger.index[0]), end=cutoff, freq="D", inclusive="left")
        if not history.index.equals(expected_dates):
            raise ValueError(f"Fold {fold} has incomplete daily history.")

        equity = history["equity_close"].astype(float)
        cash = history["cash_gbp"].astype(float)
        exposure = 1.0 - cash / equity

        if (not equity.gt(0).all() or not exposure.between(-1e-10, 1.0 + 1e-10).all()):
            raise ValueError(f"Fold {fold} has invalid exposure values.")

        crypto_weight = float(exposure.clip(0.0, 1.0).mean())
        rows.append({
            "fold": fold,
            "first_observation": history.index[0],
            "last_valuation_at": history["valuation_at"].max(),
            "calibration_days": len(history),
            "crypto_weight": crypto_weight,
            "BTC_weight": crypto_weight / 2,
            "ETH_weight": crypto_weight / 2,
            "cash_weight": 1.0 - crypto_weight
        })

    return pd.DataFrame(rows).set_index("fold")

def select_lookbacks_by_past_sharpe(sweep_runs, windows):

    #select each year's lookback using only preceding net returns (and sharpe)
    if not sweep_runs or windows.empty:
        raise ValueError("Provide candidate runs and evaluation windows.")
    if any(run["ledger"].empty for run in sweep_runs.values()):
        raise ValueError("Candidate ledgers must not be empty.")

    #all candidates need a preceding closing equity for period reporting
    common_start = max(run["ledger"].index.min() for run in sweep_runs.values()) + pd.Timedelta(days=1)

    selected_rows = []
    score_rows = []

    for fold, window in windows.iterrows():
        start = max(common_start, window["calibration_start"])
        cutoff = window["calibration_end"]

        if start >= cutoff or cutoff > window["test_start"]:
            raise ValueError(f"Fold {fold} has invalid training boundaries.")

        candidates = []
        for days, run in sweep_runs.items():
            ledger = run["ledger"]

            #retain preceding close, but exclude evaluation year rows
            history = ledger.loc[ledger.index < cutoff].copy()

            if (history["valuation_at"].isna().any() or (history["valuation_at"] > cutoff).any()):
                raise ValueError("Training contains unavailable valuations.")
            summary = crypto_reporting.summarise_backtest_period(ledger=history, start=start, end=cutoff)
            score = float(summary["sharpe_zero_cash_rate"])
            if not math.isfinite(score):
                raise ValueError(f"Undefined training Sharpe: fold {fold}, lookback {days}.")

            candidates.append({
                "fold": fold,
                "lookback_days": days,
                "training_start": start,
                "training_end": cutoff,
                "training_sharpe": score,
                "training_cagr": summary["cagr"],
                "training_turnover": summary["annualised_one_way_turnover"]
            })

        ranking = pd.DataFrame(candidates).sort_values(["training_sharpe", "lookback_days"], ascending=[False, False])
        best = ranking.iloc[0].to_dict()
        best["lookback_days"] = int(best["lookback_days"])
        best["test_start"] = window["test_start"]
        best["test_end"] = window["test_end"]

        selected_rows.append(best)
        score_rows.extend(candidates)

    selections = pd.DataFrame(selected_rows).set_index("fold")
    scores = pd.DataFrame(score_rows).set_index(["fold", "lookback_days"])

    return selections, scores

def freeze_research_protocol(protocol, path):
    #save dated protocol
    path = Path(path)
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))

        if (existing["protocol"] != protocol or existing["protocol_sha256"] != digest):
            raise ValueError("This protocol file already contains different settings.")
        return existing

    record = {
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol_sha256": digest,
        "protocol": protocol
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write("\n")

    return record