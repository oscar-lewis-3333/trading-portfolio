"""Offline replay helpers for the completed research notebook.

Strategy, accounting and primary inference stay in their original modules.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import crypto_inference
import crypto_reporting


def load_frozen_study(project_root):
    """Read the recorded snapshot without downloads or modifications.

    The original absolute snapshot path stays in the protocol. Resolve its
    directory name inside this checkout so a relocated project can replay it.
    Boundary rows are filtered per request; duplicates and gaps remain visible
    to the existing price audit.
    """
    root = Path(project_root)
    protocol_path = root / "research" / "holdout_protocol_v1.json"
    record = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol = record["protocol"]
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != record["protocol_sha256"]:
        raise ValueError("The frozen protocol hash does not match.")

    run_dir = root / "data" / "raw" / Path(protocol["data_snapshot"]).name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Frozen snapshot is required: {run_dir}")

    study = json.loads((run_dir / "study_spec.json").read_text(encoding="utf-8"))
    if study["symbols"] != protocol["symbols"]:
        raise ValueError("Snapshot and protocol universes differ.")

    records = []
    requests_seen = set()
    for path in sorted(run_dir.glob("coinbase_daily_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        symbol = raw["symbol"]
        url = urlparse(raw["request_url"])
        query = parse_qs(url.query)
        if (
            symbol not in study["symbols"]
            or raw["source"] != "coinbase_exchange"
            or url.netloc != "api.exchange.coinbase.com"
            or url.path != f"/products/{symbol}/candles"
            or query.get("granularity") != ["86400"]
        ):
            raise ValueError(f"Unexpected candle source: {path.name}")

        start = pd.Timestamp(query["start"][0])
        end = pd.Timestamp(query["end"][0])
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError(f"Invalid request boundaries: {path.name}")
        key = (symbol, start, end)
        if key in requests_seen:
            raise ValueError(f"Duplicate saved request: {path.name}")
        requests_seen.add(key)

        if not isinstance(raw["payload"], list):
            raise ValueError(f"Invalid candle payload: {path.name}")
        for row in raw["payload"]:
            if not isinstance(row, list) or len(row) != 6:
                raise ValueError(f"Invalid candle shape: {path.name}")
            if start.timestamp() <= row[0] < end.timestamp():
                item = dict(zip(["time", "low", "high", "open", "close", "volume"], row))
                item.update(symbol=symbol, source="coinbase_exchange")
                records.append(item)

    if not records:
        raise ValueError("The frozen snapshot contains no retained candles.")
    prices = pd.DataFrame(records)
    prices["timestamp"] = pd.to_datetime(prices["time"], unit="s", utc=True)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return {
        "prices": prices,
        "study_spec": study,
        "protocol_record": record,
        "protocol_path": protocol_path,
        "run_dir": run_dir,
        "saved_requests": len(requests_seen),
    }


def summarise_ledgers(ledgers, start=None, end=None):
    """Present existing reporting functions as a strategy-by-metric table."""
    if (start is None) != (end is None):
        raise ValueError("Provide both period boundaries or neither.")
    return pd.DataFrame({
        name: (
            crypto_reporting.summarise_backtest(ledger)
            if start is None
            else crypto_reporting.summarise_backtest_period(ledger, start, end)
        )
        for name, ledger in ledgers.items()
    }).T


def replay_exploratory_return_intervals(holdout_result):
    """Reproduce post-holdout return diagnostics; do not change the primary test.

    Both estimates are return differences in decimal units. Their displayed
    percentage values are percentage-point differences, not Sharpe ratios.
    """
    protocol = holdout_result["protocol"]
    settings = protocol["bootstrap"]
    names = [protocol["primary_strategy"]["name"], protocol["primary_benchmark"]["name"]]
    paired = holdout_result["returns"][names]
    values = paired.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Invalid paired return observations.")
    annualisation = protocol["sharpe_annualisation_days"]
    log_returns = np.log1p(values)
    excess = values[:, 0] - values[:, 1]
    cagrs = np.expm1(log_returns.mean(axis=0) * annualisation)
    estimates = [excess.mean() * annualisation, cagrs[0] - cagrs[1]]
    rows = []
    for block in [settings["block_days"], *settings["sensitivity_block_days"]]:
        rng = np.random.default_rng(settings["seed"])
        draws = np.empty((settings["repetitions"], 2))
        for repetition in range(settings["repetitions"]):
            indices = crypto_inference._circular_block_indices(len(values), block, rng)
            growth = np.expm1(log_returns[indices].mean(axis=0) * annualisation)
            draws[repetition] = [excess[indices].mean() * annualisation, growth[0] - growth[1]]
        alpha = 1 - settings["confidence_level"]
        lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
        for column, metric in enumerate(("annualised_mean_return_difference", "cagr_difference")):
            rows.append({"block_days": block, "metric": metric,
                         "estimate": estimates[column], "ci_lower": lower[column], "ci_upper": upper[column]})
    return pd.DataFrame(rows).set_index(["block_days", "metric"])
