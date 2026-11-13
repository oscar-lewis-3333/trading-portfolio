import numpy as np
import pandas as pd


def build_annual_walk_forward_folds(net_returns, schedule, training_years=3):
    #build rolling training windows followed by annual test periods
    if not isinstance(net_returns, pd.DataFrame) or net_returns.empty:
        raise ValueError("Provide a non-empty daily return DataFrame.")

    dates = net_returns.index
    if (not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing):
        raise ValueError("Return dates must be sorted and unique.")
    if (isinstance(training_years, bool) or not isinstance(training_years, (int, np.integer)) or training_years < 1):
        raise ValueError("training_years must be a positive integer.")

    calendar = schedule.index
    positions = calendar.get_indexer(dates)
    if (positions < 0).any() or not np.all(np.diff(positions) == 1):
        raise ValueError("Returns must cover consecutive exchange sessions.")

    rows = []
    first_test_year = dates[0].year + training_years

    for year in range(first_test_year, dates[-1].year + 1):
        training_start = pd.Timestamp(year=year - training_years, month=1, day=1)
        test_boundary = pd.Timestamp(year=year, month=1, day=1)
        test_stop = pd.Timestamp(year=year + 1, month=1, day=1)

        train_dates = calendar[(calendar >= training_start) & (calendar < test_boundary)]
        test_dates = dates[(dates >= test_boundary) & (dates < test_stop)]

        if test_dates.empty:
            continue
        if train_dates.empty or not train_dates.isin(dates).all():
            raise ValueError(f"Incomplete training history for {year}.")

        rows.append({
            "fold": len(rows),
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "selection_date": train_dates[-1],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "train_sessions": len(train_dates),
            "test_sessions": len(test_dates)
        })

    if not rows:
        raise ValueError("Insufficient history for a walk-forward fold.")

    return pd.DataFrame(rows).set_index("fold")


def select_walk_forward_configurations(net_returns, folds, parameter_grid, rules):
    #choose parameters using only each folds training returns
    if (rules["selection_metric"] != "sharpe_zero_rf" or rules["refit_frequency"] != "annual" or rules["tie_break"] != "lowest_configuration_id"):
        raise ValueError("Unsupported walk-forward selection rules.")

    if (parameter_grid.empty or not parameter_grid.index.is_unique
        or not pd.api.types.is_integer_dtype(parameter_grid.index.dtype) or not net_returns.columns.is_unique or not net_returns.columns.equals(parameter_grid.index)):
        raise ValueError("Return columns must match the complete integer-ID grid.")
    if (not isinstance(net_returns.index, pd.DatetimeIndex) or net_returns.index.hasnans or not net_returns.index.is_unique
        or not net_returns.index.is_monotonic_increasing or folds.empty or not folds.index.is_unique):
        raise ValueError("Invalid return dates or fold identifiers.")

    records = []
    for fold in folds.itertuples():
        training = net_returns.loc[fold.train_start:fold.train_end]

        if (len(training) < 2 or len(training) != fold.train_sessions or training.index[0] != fold.train_start
            or training.index[-1] != fold.train_end or fold.train_end != fold.selection_date
            or fold.selection_date >= fold.test_start or not np.isfinite(training.to_numpy()).all()):
            raise ValueError(f"Invalid training observations for fold {fold.Index}.")

        mean_returns = training.mean()
        volatility = training.std(ddof=1)

        sharpe = (np.sqrt(252) * mean_returns / volatility.where(volatility.gt(0)))
        ranked = sharpe.loc[np.isfinite(sharpe)].sort_index()
        if ranked.empty:
            raise ValueError(f"No rankable configurations for fold {fold.Index}.")

        #idxmax chooses the first maximum
        best_id = int(ranked.idxmax())

        records.append({
            "fold": fold.Index,
            "configuration_id": best_id,
            "training_sharpe": float(ranked.loc[best_id]),
            "training_mean_net_bps": (10_000 * float(mean_returns.loc[best_id])),
            "training_volatility_pct": (100 * np.sqrt(252) * float(volatility.loc[best_id])),
        })

    selections = pd.DataFrame(records).set_index("fold")

    return folds.join(selections, validate="one_to_one").join(parameter_grid, on="configuration_id", validate="many_to_one")

def build_walk_forward_targets(choices, sweep_results):
    #combine selected configs monthly target weights
    if choices.empty or not choices.index.is_unique:
        raise ValueError("Provide non-empty choices with unique fold IDs.")

    ordered = choices.sort_values("test_start")
    if (ordered["test_start"].iloc[1:].to_numpy() <= ordered["test_end"].iloc[:-1].to_numpy()).any():
        raise ValueError("Test periods must not overlap.")

    reference = sweep_results["benchmark"]["targets"].loc[ordered["test_start"].min():ordered["test_end"].max()]
    available_dates = sweep_results["benchmark"]["daily"].index
    target_parts = []
    decision_parts = []

    for choice in ordered.itertuples():
        if (not choice.selection_date < choice.test_start <= choice.test_end or choice.test_start not in available_dates or choice.test_end not in available_dates):
            raise ValueError(f"Invalid test period for fold {choice.Index}.")

        result = sweep_results["backtests"][choice.configuration_id]
        part = result["targets"].loc[choice.test_start:choice.test_end].copy()
        expected = reference.loc[choice.test_start:choice.test_end]
        if (part.empty or not part.index.equals(expected.index) or not part.columns.equals(reference.columns)
            or not np.isfinite(part.to_numpy()).all() or part.lt(0).any().any() or part.sum(axis=1).gt(1 + 1e-12).any()):
            raise ValueError(f"Invalid target weights for fold {choice.Index}.")

        decisions = result["selection"].loc[part.index].copy()

        #apply the new config at the first test-period rebalance
        if (part.index[0] != choice.test_start or decisions["signal_date"].iloc[0] != choice.selection_date):
            raise ValueError(f"Selection and first rebalance do not align for fold {choice.Index}.")

        decisions["fold"] = choice.Index
        decisions["configuration_id"] = choice.configuration_id

        target_parts.append(part)
        decision_parts.append(decisions)

    targets = pd.concat(target_parts).sort_index()
    decisions = pd.concat(decision_parts).sort_index()

    if (not targets.index.is_unique or not targets.index.equals(reference.index) or not decisions.index.equals(targets.index)):
        raise ValueError("Walk-forward targets contain gaps or duplicate dates.")

    return targets, decisions