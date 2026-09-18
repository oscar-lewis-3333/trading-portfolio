import numpy as np
import pandas as pd
import reversal_validation

def build_walk_forward_folds(net_returns, schedule, initial_train_sessions=504, test_sessions=63):
    #build expanding folds from daily returns known at close.
    if not isinstance(net_returns, pd.DataFrame) or net_returns.empty:
        raise ValueError("Provide a nonempty daily return DataFrame.")

    dates = net_returns.index
    if (not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing):
        raise ValueError("Return dates must be sorted and unique.")
    for name, value in [("initial_train_sessions", initial_train_sessions), ("test_sessions", test_sessions)]:
        if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be a positive integer.")

    positions = schedule.index.get_indexer(dates)
    if (positions < 0).any() or not np.all(np.diff(positions) == 1):
        raise ValueError("Returns must cover consecutive exchange sessions.")
    if len(dates) <= initial_train_sessions:
        raise ValueError("Insufficient observations for a test period.")
    
    rows = []
    for test_start in range(initial_train_sessions, len(dates), test_sessions):
        test_stop = min(test_start + test_sessions, len(dates))
        selection_date = dates[test_start - 1]

        rows.append({
            "fold": len(rows),
            "train_start": dates[0],
            "train_end": selection_date,
            "selection_date": selection_date,
            "selection_time": schedule.at[selection_date, "market_close"],
            "test_start": dates[test_start],
            "test_end": dates[test_stop - 1],
            "train_sessions": test_start,
            "test_sessions": test_stop - test_start
        })

    return pd.DataFrame(rows).set_index("fold")

def select_walk_forward_configurations(net_returns, folds, grid, rules):
    #choose config using only each folds training returns
    if (rules["selection_metric"] != "mean_daily_net_excess" or rules["tie_break"] != "lowest_configuration_id"):
        raise ValueError("Unsupported walk-forward selection rules.")

    if (not grid.index.is_unique or not net_returns.columns.is_unique or not net_returns.columns.equals(grid.index)):
        raise ValueError("Return columns must match the complete grid.")
    if folds.empty or not folds.index.is_unique:
        raise ValueError("Provide nonempty folds with unique identifiers.")

    rows = []
    for fold in folds.itertuples():
        training = net_returns.loc[fold.train_start:fold.train_end]

        if (training.empty or len(training) != fold.train_sessions or training.index[-1] != fold.selection_date or fold.selection_date >= fold.test_start or not np.isfinite(training.to_numpy()).all()):
            raise ValueError(f"Invalid training observations for fold {fold.Index}.")
        #lowest config ID wins an exact tie
        training_means = training.mean().sort_index()
        best_id = int(training_means.idxmax())
        best_mean = float(training_means.loc[best_id])

        allow_entries = best_mean > rules["minimum_training_mean"]

        rows.append({
            "fold": fold.Index,
            "configuration_id": best_id if allow_entries else pd.NA,
            "best_training_mean_bps": 10_000 * best_mean,
            "allow_new_entries": allow_entries
        })

    selection = pd.DataFrame(rows).set_index("fold").astype({"configuration_id": "Int64"})

    return folds.join(selection, validate="one_to_one").join(grid[["formation_sessions", "top_frac", "holding_sessions"]], on="configuration_id", validate="many_to_one")

def build_walk_forward_decisions(choices, feature_panels, decision_dates_by_holding, *, min_eligible=10):

    #build portfolio targets using each folds selected config
    if choices.empty or not choices.index.is_unique:
        raise ValueError("Provide nonempty choices with unique fold IDs.")

    selection_dates = choices["selection_date"]
    if (selection_dates.isna().any() or selection_dates.duplicated().any() or not selection_dates.is_monotonic_increasing):
        raise ValueError("Selection dates must be sorted and unique.")

    parts = []
    for fold in choices.itertuples():
        #existing positions still exit through execution engine
        if not fold.allow_new_entries:
            continue

        formation = int(fold.formation_sessions)
        holding = int(fold.holding_sessions)
        calendar = decision_dates_by_holding[holding]


        #preserve sweeps trading calender. signal at test_end belongs to next selection period
        dates = calendar[(calendar >= fold.selection_date) & (calendar < fold.test_end)]
        if dates.empty:
            continue

        features = feature_panels[formation]
        part = reversal_validation.select_reversal_candidates(
            features.loc[features.index.get_level_values("Date").isin(dates)],
            score_column="raw_reversal_score",
            top_frac=float(fold.top_frac),
            min_eligible=min_eligible
        )
        observed_dates = part.index.get_level_values("Date").unique()
        if not observed_dates.equals(dates):
            raise ValueError(f"Missing feature dates for fold {fold.Index}.")

        slots = part["target_slots"]
        part["target_weight"] = part["selected"].astype(float).div(slots.where(slots.gt(0))).fillna(0.0)

        part["holding_sessions"] = holding
        part["configuration_id"] = int(fold.configuration_id)
        part["wf_fold"] = fold.Index
        parts.append(part)

    #start valuation at first selection close (even if trading starts later)
    start = selection_dates.iloc[0]
    if (not parts or start not in parts[0].index.get_level_values("Date")):
        reference = feature_panels[min(feature_panels)]
        initial_cash = (reference.xs(start, level="Date", drop_level=False)[["signal_time", "eligible"]].copy().assign(
                eligible=False,
                selected=False,
                target_weight=0.0,
                holding_sessions=1,  #unused since no purchases
                configuration_id=pd.NA,
                wf_fold=choices.index[0]
            ))
        parts.insert(0, initial_cash)

    decisions = pd.concat(parts).sort_index()
    if not decisions.index.is_unique:
        raise ValueError("Walk-forward folds created duplicate decision rows.")

    decisions["configuration_id"] = decisions["configuration_id"].astype("Int64")
    return decisions