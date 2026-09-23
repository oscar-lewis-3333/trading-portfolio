#quarterly development choices using only earlier, after-cost daily returns
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rebalance_schedule import build_development_grid


def development_walk_forward_rules(protocol):
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    holdout = pd.Timestamp(protocol["holdout_start"], tz="UTC")
    if (not start < start + pd.DateOffset(years=1) < end <= holdout or start.month != 1 or start.day != 1 or any(t != t.normalize() for t in (start, end, holdout))):
        raise ValueError("Need a January-start development study, a full initial year and later test dates.")
    
    return {
        "version": "development_walk_forward_v1",
        "research_protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "initial_training": "First complete calendar year of development",
        "training_window": "Expanding from development_start",
        "review_frequency": "Calendar quarters at 00:00 UTC",
        "selection_metric": "Highest mean daily strategy-minus-matched-benchmark return after costs",
        "tie_break": "Lexicographically lowest config_id on an exact score tie",
        "candidate_grid": "All 60 fixed development configurations",
        "fee_bps": 9.0,
        "other_cost_bps": 3.5,
        "training_availability": "A candle-start return is available at the following midnight; include only availability <= selection_at",
        "negative_training_score": "Still select the highest score; no additional entry filter",
        "cash": "Empty eligible universe or awaiting initial scheduled entry; otherwise carry holdings between rebalances",
        "application": "Use the new configuration at its first anchored decision on or after selection_at",
        "calendar_anchor": protocol["development_start"],
        "portfolio_switching": "Carry units/cash continuously; charge per-asset absolute net trades; never splice candidate returns",
        "benchmark_switching": "Carry the matched benchmark continuously and adopt the selected frequency",
        "initial_test_capital_gbp": 1000.0,
        "evaluation": "Development diagnostic; reviewed development outcomes informed this research process",
        "holdout": "Not accessed; freeze final strategy and inference separately before release"
    }


def freeze_development_walk_forward(project_root, protocol):
    #save a seperate policy, preserving the original research/data protocol
    expected = development_walk_forward_rules(protocol)
    path = Path(project_root) / "research" / "development_walk_forward_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Existing walk-forward policy differs; preserve it and use a new version.")
    return expected


def build_development_folds(protocol, rules):
    if rules != development_walk_forward_rules(protocol):
        raise ValueError("Walk-forward rules differ from the saved design.")
    
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    first = start + pd.DateOffset(years=1)
    boundaries = pd.date_range(first, end, freq="QS", inclusive="left")
    rows = []
    for fold, selection_at in enumerate(boundaries, 1):
        test_end = min(selection_at + pd.DateOffset(months=3), end)
        rows.append({
            "fold": fold,
            "training_start": start,
            "training_end_exclusive": selection_at,
            "last_training_candle": selection_at - pd.Timedelta(days=1),
            "last_training_valuation_at": selection_at,
            "training_days": (selection_at - start).days,
            "selection_at": selection_at,
            "test_start": selection_at,
            "test_end_exclusive": test_end,
            "test_days": (test_end - selection_at).days
        })
    return pd.DataFrame(rows).set_index("fold")


def select_development_configurations(strategy_returns, benchmark_returns, development_grid, protocol, rules):
    #return quarterly choices and fold-by configuration training scores
    folds = build_development_folds(protocol, rules)
    expected_grid = build_development_grid()
    columns = ["lookback_days", "top_fraction", "rebalance_days"]
    if (not development_grid.index.is_unique or set(development_grid.index) != set(expected_grid.index) or not set(columns).issubset(development_grid.columns)
        or not np.array_equal(development_grid.loc[expected_grid.index, columns].to_numpy(dtype=float), expected_grid[columns].to_numpy(dtype=float))):
        raise ValueError("Supply the complete, unchanged development grid.")
    
    dates = pd.date_range(protocol["development_start"], protocol["development_end_exclusive"], freq="D", tz="UTC", inclusive="left")
    for frame in (strategy_returns, benchmark_returns):
        if (not isinstance(frame, pd.DataFrame) or not frame.index.equals(dates) or not frame.columns.is_unique or set(frame.columns) != set(expected_grid.index)):
            raise ValueError("Return matrices must cover all development days and all 60 configurations.")
        
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= -1).any():
            raise ValueError("Daily returns must be finite and greater than -100%.")

    #explicit alignment protects pairing when input column orders differ
    excess = (strategy_returns.loc[:, expected_grid.index] - benchmark_returns.loc[:, expected_grid.index])
    choices, scores = [], []
    anchor = pd.Timestamp(protocol["development_start"], tz="UTC")
    for fold in folds.itertuples():
        training = excess.loc[(excess.index >= fold.training_start) & (excess.index < fold.training_end_exclusive)]
        available_at = training.index + pd.Timedelta(days=1)
        if (len(training) != fold.training_days or available_at.max() != fold.selection_at or not (available_at <= fold.selection_at).all()):
            raise ValueError("Training returns are missing or not yet available at selection.")
        
        means = training.mean().sort_index()
        config_id = means.idxmax()
        config = development_grid.loc[config_id]
        interval = int(config["rebalance_days"])
        days_since_anchor = (fold.selection_at - anchor).days
        first_decision = fold.selection_at + pd.Timedelta(days=(-days_since_anchor) % interval)

        choices.append({
            "fold": fold.Index, "config_id": config_id,
            "lookback_days": int(config["lookback_days"]),
            "top_fraction": float(config["top_fraction"]),
            "rebalance_days": interval,
            "training_mean_net_excess_bps": float(means.loc[config_id] * 10000),
            "positive_training_mean": bool(means.loc[config_id] > 0),
            "first_scheduled_decision_at": first_decision
        })
        scores.append((means * 10000).rename(fold.Index))
    training_scores = pd.DataFrame(scores)
    training_scores.index.name = "fold"
    return folds.join(pd.DataFrame(choices).set_index("fold"), validate="one_to_one"), training_scores
