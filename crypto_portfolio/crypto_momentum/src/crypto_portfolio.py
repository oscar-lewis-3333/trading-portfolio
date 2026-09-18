import pandas as pd


def build_weekly_targets(features, symbols, execution_end, asset_weight=0.5):
    #build monday decisions with execution scheduled one day later
    symbols = tuple(symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Provide a nonempty, unique asset universe.")
    if set(features["symbol"]) != set(symbols):
        raise ValueError("Features must contain exactly the requested assets.")
    if not 0 < asset_weight <= 1 / len(symbols):
        raise ValueError("Asset weights must be positive and sum to at most 1.")

    execution_end = pd.Timestamp(execution_end)
    if execution_end.tzinfo is None:
        raise ValueError("execution_end must be timezone-aware.")

    execution_end = execution_end.tz_convert("UTC")
    monday_rows = features.loc[features["signal_available_at"].dt.dayofweek.eq(0)]
    signals = monday_rows.pivot(index="signal_available_at", columns="symbol", values="trend_positive").reindex(columns=list(symbols)).sort_index()

    ready = signals.notna().all(axis=1)
    if not ready.any():
        raise ValueError("No weekly decisions have complete signal history.")

    first_ready = ready.loc[ready].index[0]
    if not ready.loc[first_ready:].all():
        raise ValueError("Missing weekly signals after the initial warm-up.")

    targets = signals.loc[ready].astype(float) * asset_weight
    targets["GBP_cash"] = 1.0 - targets.sum(axis=1)
    targets = targets.rename_axis("decision_at").reset_index()
    targets.columns.name = None

    targets.insert(1, "execution_at", targets["decision_at"] + pd.Timedelta(days=1))
    return targets.loc[targets["execution_at"] < execution_end].reset_index(drop=True)

def build_benchmark_targets(weekly_targets, symbols, mode="buy_and_hold"):

    #create equal-weight allocation benchmarks
    symbols = tuple(symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Provide a nonempty, unique asset universe.")
    if weekly_targets.empty:
        raise ValueError("No execution schedule was supplied.")
    if mode not in {"buy_and_hold", "weekly"}:
        raise ValueError("mode must be 'buy_and_hold' or 'weekly'.")

    schedule = weekly_targets.sort_values("execution_at")
    if mode == "buy_and_hold":
        schedule = schedule.iloc[:1]

    targets = schedule[["decision_at", "execution_at"]].copy()
    for symbol in symbols:
        targets[symbol] = 1.0 / len(symbols)

    targets["GBP_cash"] = 0.0
    return targets.reset_index(drop=True)

def build_constant_mix_targets(weekly_targets, symbols, crypto_weight=0.5):
    #build weekly equal-weight crypto targets with a fixed cash allocation
    crypto_weight = float(crypto_weight)
    if not 0.0 <= crypto_weight <= 1.0:
        raise ValueError("crypto_weight must be between 0 and 1.")

    symbols = tuple(symbols)
    targets = build_benchmark_targets(weekly_targets=weekly_targets, symbols=symbols, mode="weekly")
    targets[list(symbols)] *= crypto_weight
    targets["GBP_cash"] = 1.0 - crypto_weight

    return targets

def build_calibrated_mix_targets(weekly_targets, symbols, windows, allocations):
    
    #apply annual calibrated weights to a continuous weekly schedule
    symbols = tuple(symbols)

    if windows.empty or not windows.index.is_unique:
        raise ValueError("Provide nonempty windows with unique fold IDs.")
    if not allocations.index.is_unique:
        raise ValueError("Allocation fold IDs must be unique.")

    targets = build_constant_mix_targets(weekly_targets=weekly_targets, symbols=symbols, crypto_weight=0.5)
    targets["calibration_fold"] = 0

    for fold, window in windows.sort_values("test_start").iterrows():
        allocation = allocations.loc[fold]
        available_at = allocation["last_valuation_at"]

        if (pd.isna(available_at) or available_at > window["calibration_end"] or window["calibration_end"] > window["test_start"]):
            raise ValueError(f"Fold {fold} uses unavailable information.")

        weight = float(allocation["crypto_weight"])
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Fold {fold} has an invalid crypto weight.")

        #select by decision time
        selected = ((targets["decision_at"] >= window["test_start"]) & (targets["decision_at"] < window["test_end"]))

        if not selected.any():
            raise ValueError(f"Fold {fold} has no weekly decisions.")

        if targets.loc[selected, "calibration_fold"].ne(0).any():
            raise ValueError("Evaluation windows overlap.")

        targets.loc[selected, list(symbols)] = weight / len(symbols)
        targets.loc[selected, "GBP_cash"] = 1.0 - weight
        targets.loc[selected, "calibration_fold"] = fold

    #only initialisation period may retain default allocation
    needs_calibration = targets["decision_at"] >= windows["test_start"].min()
    if targets.loc[needs_calibration, "calibration_fold"].eq(0).any():
        raise ValueError("Some later decisions have no calibration window.")

    return targets

def build_selected_momentum_targets(weekly_targets, symbols, sweep_runs, selections):
    #build one continuous schedule from annual lookback selections
    
    symbols = tuple(symbols)
    if selections.empty or not selections.index.is_unique:
        raise ValueError("Provide selections with unique fold IDs.")

    targets = build_constant_mix_targets(weekly_targets=weekly_targets, symbols=symbols, crypto_weight=0.5)
    targets["selection_fold"] = 0
    targets["selected_lookback_days"] = pd.Series(pd.NA, index=targets.index, dtype="Int64")

    weight_columns = list(symbols) + ["GBP_cash"]

    for fold, selection in selections.sort_values("test_start").iterrows():
        start = selection["test_start"]
        end = selection["test_end"]
        available_at = selection["training_end"]
        days = int(selection["lookback_days"])
        if (pd.isna(available_at) or available_at > start or start >= end):
            raise ValueError(f"Fold {fold} has invalid timing.")

        chosen = ((targets["decision_at"] >= start) & (targets["decision_at"] < end))
        if not chosen.any():
            raise ValueError(f"Fold {fold} has no weekly decisions.")
        if targets.loc[chosen, "selection_fold"].ne(0).any():
            raise ValueError("Selection periods overlap.")

        candidate = sweep_runs[days]["targets"]
        if candidate["decision_at"].duplicated().any():
            raise ValueError("Candidate decisions must be unique.")

        aligned = candidate.set_index("decision_at").reindex(targets.loc[chosen, "decision_at"])
        if aligned[["execution_at"] + weight_columns].isna().any().any():
            raise ValueError("Candidate is missing required targets.")
        if not pd.DatetimeIndex(aligned["execution_at"]).equals(pd.DatetimeIndex(targets.loc[chosen, "execution_at"])):
            raise ValueError("Candidate execution dates do not match.")

        targets.loc[chosen, weight_columns] = aligned[weight_columns].to_numpy(dtype=float)
        targets.loc[chosen, "selection_fold"] = fold
        targets.loc[chosen, "selected_lookback_days"] = days

    needs_selection = (targets["decision_at"] >= selections["test_start"].min())
    if targets.loc[needs_selection, "selection_fold"].eq(0).any():
        raise ValueError("Some later decisions have no selected rule.")

    return targets