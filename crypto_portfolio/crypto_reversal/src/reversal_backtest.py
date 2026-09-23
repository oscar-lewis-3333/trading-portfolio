import numpy as np
import pandas as pd
from portfolio import rebalance_gbp


def _utc_times(values, label, midnight=False):
    times = pd.DatetimeIndex(values)
    if str(times.tz) != "UTC" or times.hasnans:
        raise ValueError(f"{label} must contain non-null UTC timestamps.")
    if midnight and not (times == times.normalize()).all():
        raise ValueError(f"{label} must contain UTC midnights.")
    return times


def run_development_ledger(gbp_prices, decision_targets, execution_schedule, execution_references, protocol, rebalance_days, initial_cash=1000.0, fee_bps=9.0, other_cost_bps=3.5):
    #keep the fixed-configuration calendar contract used by the original sweep.
    if (isinstance(rebalance_days, (bool, np.bool_)) or not np.isfinite(rebalance_days) or rebalance_days < 1 or int(rebalance_days) != rebalance_days):
        raise ValueError("rebalance_days must be a positive integer.")
    expected = pd.date_range(protocol["development_start"], protocol["development_end_exclusive"], freq=f"{int(rebalance_days)}D", tz="UTC", inclusive="left")
    days = _utc_times(decision_targets.index, "Decisions", midnight=True)
    if not days.equals(expected):
        raise ValueError("Targets must cover the full anchored development rebalance calendar.")
    return _run_ledger(gbp_prices, decision_targets, execution_schedule, execution_references, protocol, pd.Timestamp(protocol["development_start"], tz="UTC"), initial_cash, fee_bps, other_cost_bps)


def run_walk_forward_ledger(gbp_prices, decision_targets, execution_schedule, execution_references, protocol, rules):
    #start afresh only at the beginning of evaluation, never at later quarters.
    from reversal_walk_forward import build_development_folds

    folds = build_development_folds(protocol, rules)
    start = folds.test_start.iloc[0]
    days = _utc_times(decision_targets.index, "Walk-forward decisions", midnight=True)
    if days.empty or days[0] != start:
        raise ValueError("Include the first evaluation day, using a cash target if awaiting entry.")
    return _run_ledger(gbp_prices, decision_targets, execution_schedule, execution_references, protocol, start, rules["initial_test_capital_gbp"], rules["fee_bps"], rules["other_cost_bps"])


def _run_ledger(gbp_prices, decision_targets, execution_schedule, execution_references, protocol, evaluation_start, initial_cash, fee_bps, other_cost_bps):
    #carry units/cash and rebalance only on supplied target dates.
    development_start = pd.Timestamp(protocol["development_start"], tz="UTC")
    start = pd.Timestamp(evaluation_start)
    end = pd.Timestamp(protocol["development_end_exclusive"], tz="UTC")
    if not development_start <= start < end <= pd.Timestamp(protocol["holdout_start"], tz="UTC"):
        raise ValueError("Invalid development/holdout boundaries.")

    targets = decision_targets.copy()
    schedule = execution_schedule.copy()
    refs = execution_references.copy()
    prices = gbp_prices.copy()

    if targets.empty or not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Provide targets and positive initial cash.")
    if targets.columns.has_duplicates or "CASH" not in targets:
        raise ValueError("Expected unique target columns including CASH.")
    
    days = _utc_times(targets.index, "Decisions", midnight=True)
    if (days.has_duplicates or not days.is_monotonic_increasing or ((days < start) | (days >= end)).any()):
        raise ValueError("Targets must be unique ordered dates inside the evaluation window.")
    
    values = targets.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Invalid target weights.")
    if (not np.isfinite([fee_bps, other_cost_bps]).all() or min(fee_bps, other_cost_bps) < 0 or fee_bps + other_cost_bps >= 10000):
        raise ValueError("Invalid cost assumptions.")

    candle_times = _utc_times(prices["timestamp"], "Daily candles", midnight=True)
    if prices.empty or prices["product_id"].isna().any() or prices.duplicated(["product_id", "timestamp"]).any():
        raise ValueError("Missing or duplicate daily candles.")
    if ((candle_times < development_start) | (candle_times >= end)).any():
        raise ValueError("Daily valuation candles must be restricted to development.")
    
    prices = prices.loc[prices["timestamp"].ge(start)].copy()

    #saved execution data cover the grid
    for frame, column in ((schedule, "day"), (refs, "day")):
        times = _utc_times(frame[column], "Shared reference/schedule dates", midnight=True)
        if ((times < development_start) | (times >= end)).any():
            raise ValueError("Execution data extend outside development.")
    if schedule["day"].duplicated().any() or refs.duplicated(["product_id", "day"]).any():
        raise ValueError("Duplicate shared execution data.")
    
    schedule = schedule.loc[schedule["day"].isin(days)].copy()
    refs = refs.loc[refs["day"].isin(days)].copy()

    schedule_days = _utc_times(schedule["day"], "Schedule days", midnight=True)
    if schedule_days.has_duplicates or not schedule_days.sort_values().equals(days):
        raise ValueError("The schedule must cover every target date exactly once.")
    if not schedule["status"].isin(["ready", "cash_only"]).all():
        raise ValueError("Resolve the execution schedule before backtesting.")
    
    execution_times = _utc_times(schedule["execution_at"], "Execution times")
    minutes = (execution_times - schedule_days).total_seconds() / 60
    if ((minutes < 5) | (minutes > 60) | (minutes % 1 != 0)).any():
        raise ValueError("Execution must occur at a minute boundary from 00:05 to 01:00.")
    
    schedule = schedule.set_index("day").sort_index()

    _utc_times(refs["day"], "Reference days", midnight=True)
    _utc_times(refs["execution_at"], "Reference execution times")
    _utc_times(refs["reference_bar_start"], "Reference bars")
    _utc_times(refs["assumed_available_at"], "Reference availability")
    if (refs["product_id"].isna().any() or refs.duplicated(["product_id", "day"]).any() or not refs["day"].isin(days).all()):
        raise ValueError("Missing IDs, duplicate references or references outside the schedule.")
    
    if not refs["execution_at"].eq(refs["day"].map(schedule["execution_at"])).all():
        raise ValueError("Reference times disagree with the common schedule.")
    
    bars = refs["reference_bar_start"]
    availability = bars + pd.Timedelta(minutes=1)
    ages = (refs["execution_at"] - bars).dt.total_seconds() / 60
    if (not bars.eq(bars.dt.floor("min")).all() or not bars.ge(refs["day"]).all() or not refs["assumed_available_at"].eq(availability).all() or not availability.le(refs["execution_at"]).all()or not ages.between(1, 5).all()):
        raise ValueError("Future, stale or inconsistent execution references.")
    
    reference_values = refs["reference_price_gbp"].to_numpy(dtype=float)
    if not np.isfinite(reference_values).all() or (reference_values <= 0).any():
        raise ValueError("Execution references must be finite and positive.")
    
    counts = refs.groupby("day").size().reindex(days, fill_value=0)
    if not counts.eq(schedule["required_assets"]).all():
        raise ValueError("Execution reference coverage differs from the audited schedule.")

    assets = targets.columns.drop("CASH").tolist()
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    panels = {field: prices.pivot(index="timestamp", columns="product_id", values=field).reindex(index=dates, columns=assets) for field in ("open_gbp", "close_gbp")}

    execution_prices = {day: group.set_index("product_id")["reference_price_gbp"] for day, group in refs.groupby("day")}
    units = pd.Series(0.0, index=assets)
    cash = previous_equity = float(initial_cash)
    ledger_rows, trade_parts, position_rows = [], [], []

    def require_prices(row, needed, date, label):
        selected = row.reindex(needed)
        bad = ~np.isfinite(selected.to_numpy(dtype=float)) | selected.le(0).to_numpy()
        if bad.any():
            raise ValueError(f"{date}: invalid {label} prices for {selected.index[bad].tolist()}")
        
        return selected

    def mark(row, date, label):
        held = units.loc[units.gt(0)]
        p = require_prices(row, held.index, date, label)
        equity = float(cash + (held * p).sum())
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError(f"{date}: invalid equity.")
        
        return equity

    for date in dates:
        opening = panels["open_gbp"].loc[date]
        closing = panels["close_gbp"].loc[date]
        equity_open = mark(opening, date, "opening")
        gap_pnl = equity_open - previous_equity

        scheduled = date in schedule.index
        fee = other = notional = error = pre_execution_pnl = 0.0
        before = after = np.nan
        execution_at = pd.NaT

        if scheduled:
            execution_at = schedule.loc[date, "execution_at"]
            reference = execution_prices.get(date, pd.Series(dtype=float))
            target = targets.loc[date]

            wanted = target.drop("CASH").loc[lambda x: x.gt(0)].index
            needed = units.loc[units.gt(0)].index.union(wanted)
            require_prices(reference, needed, date, "execution")
            before = mark(reference, date, "pre-rebalance")

            pre_execution_pnl = before - equity_open
            units, cash, trades, audit = rebalance_gbp(units, cash, reference, target, fee_bps=fee_bps, other_cost_bps=other_cost_bps)

            fee = float(trades["fee_gbp"].sum())
            other = float(trades["other_cost_gbp"].sum())
            notional = float(audit["traded_notional_gbp"])
            after = mark(reference, date, "post-rebalance")

            error = after + fee + other - before
            if abs(error) > 1e-10 * max(1.0, before):
                raise RuntimeError(f"{date}: rebalance accounting does not reconcile.")
            
            actual = trades.loc[trades["signed_notional_gbp"].ne(0)].reset_index()
            if not actual.empty:
                actual["decision_at"] = date
                actual["execution_at"] = execution_at
                trade_parts.append(actual)

        equity_close = mark(closing, date, "closing")
        post_execution_pnl = equity_close - (after if scheduled else equity_open)
        pnl_error = (equity_close - previous_equity) - (gap_pnl + pre_execution_pnl + post_execution_pnl - fee - other)
        if abs(pnl_error) > 1e-10 * max(1.0, previous_equity):
            raise RuntimeError(f"{date}: daily profit/loss does not reconcile.")
        
        ledger_rows.append({
            "candle_start": date,
            "valuation_at": date + pd.Timedelta(days=1),
            "scheduled_rebalance": scheduled,
            "execution_at": execution_at,
            "equity_open_gbp": equity_open,
            "equity_before_rebalance_gbp": before,
            "equity_after_rebalance_gbp": after,
            "equity_close_gbp": equity_close,
            "cash_gbp": cash,
            "fee_gbp": fee,
            "other_cost_gbp": other,
            "traded_notional_gbp": notional,
            "net_return": equity_close / previous_equity - 1.0,
            "crypto_weight_close": 1.0 - cash / equity_close,
            "previous_close_to_open_pnl_gbp": gap_pnl,
            "pre_execution_pnl_gbp": pre_execution_pnl,
            "post_execution_or_nonrebalance_pnl_gbp": post_execution_pnl,
            "accounting_error_gbp": error,
            "daily_pnl_error_gbp": pnl_error
        })
        position_rows.append(units.reindex(assets).copy())
        previous_equity = equity_close

    ledger = pd.DataFrame(ledger_rows).set_index("candle_start")
    ledger["execution_at"] = pd.to_datetime(ledger["execution_at"], utc=True)
    positions = pd.DataFrame(position_rows, index=ledger.index).reindex(columns=assets)

    trade_columns = ["product_id", "reference_price_gbp", "units_before", "units_after", "signed_notional_gbp", "fee_gbp", "other_cost_gbp", "decision_at", "execution_at"]
    trade_log = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(columns=trade_columns)
    return ledger, trade_log, positions