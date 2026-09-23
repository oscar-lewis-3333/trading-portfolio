
import numpy as np
import pandas as pd

from rebalance_schedule import build_development_grid
from reversal_walk_forward import build_development_folds


def _weights(frame):
    dates = frame.index
    if (not isinstance(dates, pd.DatetimeIndex) or str(dates.tz) != "UTC"
            or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize()) or frame.columns.has_duplicates or "CASH" not in frame.columns):
        raise ValueError("Targets need unique ordered UTC dates, asset columns and CASH.")
    
    values = frame.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError("Targets must be non-negative and sum to one.")


def build_walk_forward_targets(choices, sweep_targets, protocol, rules):
    #return strategy targets, benchmark targets and one event audit
    folds = build_development_folds(protocol, rules)
    grid = build_development_grid()
    if (not choices.index.equals(folds.index) or not set(folds.columns).issubset(choices.columns)
            or not choices[folds.columns].equals(folds) or set(sweep_targets) != set(grid.index)):
        raise ValueError("Supply complete quarterly choices and the full target grid.")
    
    start = pd.Timestamp(protocol["development_start"], tz="UTC")
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    parts = {"strategy": [], "benchmark": []}
    events = []
    for fold in choices.itertuples():
        if fold.config_id not in grid.index:
            raise ValueError("Unknown selected configuration.")
        
        config = grid.loc[fold.config_id]

        for field in ("lookback_days", "top_fraction", "rebalance_days"):
            if getattr(fold, field) != config[field]:
                raise ValueError("Selected configuration ID and parameters disagree.")
            
        interval = int(config.rebalance_days)
        calendar = pd.date_range(start, end, freq=f"{interval}D", inclusive="left")
        selected_days = calendar[(calendar >= fold.selection_at) & (calendar < fold.test_end_exclusive)]
        if selected_days.empty or fold.first_scheduled_decision_at != selected_days[0]:
            raise ValueError("First selected decision disagrees with the anchored calendar.")
        
        for role in parts:
            source = sweep_targets[fold.config_id][role]
            _weights(source)
            if not source.index.equals(calendar):
                raise ValueError("Candidate targets omit or add scheduled dates.")
            
            parts[role].append(source.loc[selected_days].copy())
        for day in selected_days:
            events.append({"decision_at": day, "fold": fold.Index,
                           "selection_at": fold.selection_at, "config_id": fold.config_id,
                           "rebalance_days": interval, "event_kind": "rebalance"})

    assets = sorted({c for frames in parts.values() for frame in frames for c in frame.columns if c != "CASH"})
    matrices = {role: pd.concat(frames).reindex(columns=assets + ["CASH"], fill_value=0.).fillna(0.) for role, frames in parts.items()}
    events = pd.DataFrame(events).set_index("decision_at")
    first_day = folds.test_start.iloc[0]
    if events.index[0] > first_day:
        #a cash-only initial state keeps the full evaluation window observable.
        for role, matrix in matrices.items():
            initial = pd.DataFrame(0., index=pd.DatetimeIndex([first_day]), columns=matrix.columns)
            initial["CASH"] = 1.
            matrices[role] = pd.concat([initial, matrix])

        initial_event = events.iloc[[0]].copy()
        initial_event.index = pd.DatetimeIndex([first_day], name="decision_at")
        initial_event["event_kind"] = "initial_cash"
        events = pd.concat([initial_event, events])

    for role, matrix in matrices.items():
        matrix.index.name = "decision_at"
        _weights(matrix)
        if not matrix.index.equals(events.index):
            raise ValueError("Target and event calendars disagree.")
        
        events[f"{role}_assets"] = matrix.drop(columns="CASH").gt(0).sum(axis=1)
    events["configuration_changed"] = events.config_id.ne(events.config_id.shift()) & events.event_kind.eq("rebalance")

    #if we awaited an initial entry, that first real event starts the configuration.
    first_rebalance = events.index[events.event_kind.eq("rebalance")][0]
    events.loc[first_rebalance, "configuration_changed"] = True
    return matrices["strategy"], matrices["benchmark"], events


def audit_walk_forward_references(strategy_targets, benchmark_targets, references):
    #check current targets and preceding event targets across both portfolios

    for frame in (strategy_targets, benchmark_targets):
        _weights(frame)
    if strategy_targets.empty or not strategy_targets.index.equals(benchmark_targets.index):
        raise ValueError("Strategy and benchmark calendars must match.")
    
    required = {}
    for weights in (strategy_targets, benchmark_targets):
        previous = set()
        for day, row in weights.drop(columns="CASH").iterrows():
            current = set(row.index[row.gt(0)])
            for product in current | previous:
                flags = required.setdefault((day, product), [False, False])
                flags[0] |= product in current
                flags[1] |= product in previous
            previous = current
    columns = ["day", "product_id", "current_target", "previous_target", "exit_only"]
    plan = pd.DataFrame([{"day": day, "product_id": product, "current_target": flags[0], "previous_target": flags[1],
                          "exit_only": flags[1] and not flags[0]} for (day, product), flags in sorted(required.items())], columns=columns)
    plan["day"] = pd.to_datetime(plan["day"], utc=True)
    ref_columns = ["product_id", "day", "execution_at", "reference_bar_start", "assumed_available_at", "reference_price_gbp"]
    
    refs = references[ref_columns].copy()
    if refs[["product_id", "day"]].isna().any().any() or refs.duplicated(["product_id", "day"]).any():
        raise ValueError("Reference keys must be non-null and unique.")
    for column in ("day", "execution_at", "reference_bar_start", "assumed_available_at"):
        if not isinstance(refs[column].dtype, pd.DatetimeTZDtype) or str(refs[column].dt.tz) != "UTC":
            raise ValueError("Reference timestamps must use UTC.")
        
    coverage = plan.merge(refs, on=["day", "product_id"], how="left", validate="one_to_one", indicator=True)
    found = coverage["_merge"].eq("both")
    ages = (coverage.execution_at - coverage.reference_bar_start) / pd.Timedelta(minutes=1)
    minutes = (coverage.execution_at - coverage.day) / pd.Timedelta(minutes=1)

    valid = (found & np.isfinite(coverage.reference_price_gbp) & coverage.reference_price_gbp.gt(0)
             & ages.between(1, 5) & minutes.between(5, 60) & minutes.mod(1).eq(0)
             & coverage.reference_bar_start.ge(coverage.day) & coverage.reference_bar_start.eq(coverage.reference_bar_start.dt.floor("min"))
             & coverage.assumed_available_at.eq(coverage.reference_bar_start + pd.Timedelta(minutes=1)) & coverage.assumed_available_at.le(coverage.execution_at))
    
    common_counts = coverage.loc[found].groupby("day").execution_at.nunique()
    valid &= coverage.day.map(common_counts).eq(1)
    coverage["status"] = np.where(~found, "missing_reference", np.where(valid, "ready", "invalid_reference"))
    coverage = coverage.drop(columns="_merge")

    summary = pd.Series({
        "target_events": len(strategy_targets), "required_references": len(coverage),
        "exit_only_references": int(coverage.exit_only.sum()),
        "ready_references": int(coverage.status.eq("ready").sum()),
        "missing_references": int(coverage.status.eq("missing_reference").sum()),
        "invalid_references": int(coverage.status.eq("invalid_reference").sum()),
    })
    return coverage, summary
