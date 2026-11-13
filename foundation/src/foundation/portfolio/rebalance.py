
#causal targets, paired net rebalance days comparisions
from numbers import Integral

import numpy as np
import pandas as pd

from foundation.data import utc


_SCHEDULE = "foundation_tranche"
_INDEX_NAME = "tranche_execution_at"


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _clock(values):
    clock = pd.DatetimeIndex(utc(values))
    if clock.empty or clock.has_duplicates or not clock.is_monotonic_increasing:
        raise ValueError("Nonempty sorted unique timezone-aware clocks required")
    return clock


def tranche(targets, k, calendar):

    #expand signal day weights into execution time target weights
    k = _integer(k, "k", 1)
    calendar = _clock(calendar)
    if not isinstance(targets, pd.DataFrame):
        raise ValueError("targets must be a signal-indexed DataFrame including CASH")
    
    signals = _clock(targets.index)
    if (targets.columns.has_duplicates or "CASH" not in targets or not np.isfinite(targets.to_numpy(float)).all() or (targets < 0).any().any() or not np.allclose(targets.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("Targets must be finite, nonnegative and sum to one including CASH")
    first = calendar.searchsorted(signals, side="right")
    if (first >= len(calendar)).any() or len(set(first)) != len(first):
        raise ValueError("Every signal needs a distinct future executable observation")
    
    starts = dict(zip(first, range(len(signals))))
    values = targets.to_numpy(float)
    current = np.zeros(len(targets.columns))
    current[targets.columns.get_loc("CASH")] = 1.
    rows = []
    schedule = []
    step = k
    signal_row = None
    interrupted = 0

    for i, at in enumerate(calendar):
        if i in starts:
            interrupted += int(step < k)
            signal_row = starts[i]
            start = current.copy()
            destination = values[signal_row]
            step = 0
        if step < k:
            step += 1
            fraction = step / k
            current = (destination.copy() if step == k else
                       (1 - fraction) * start + fraction * destination)
            #enforce precise bounds for the roundoff at equal endpoints
            current = np.clip(current, np.minimum(start, destination), np.maximum(start, destination))
            schedule.append({"execution_at": at.isoformat(), "signal_at": signals[signal_row].isoformat()})
        rows.append(current.copy())
    result = pd.DataFrame(rows, index=calendar.rename(_INDEX_NAME), columns=targets.columns)
    result.attrs[_SCHEDULE] = {"version": 1, "k": k, "schedule": schedule,
                               "interrupted_ramps": interrupted,
                               "pending_steps": k - step}
    return result


def _execution_plan(targets, calendar):
    #validate the preserved schedule at execution boundary

    metadata = targets.attrs.get(_SCHEDULE)
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError("Missing or unsupported tranche schedule; preserve DataFrame attrs")
    _integer(metadata.get("k"), "k", 1)
    if not targets.index.equals(calendar):
        raise ValueError("Tranche calendar must exactly match executable price observations")
    records = metadata.get("schedule")
    if not isinstance(records, list) or not records:
        raise ValueError("Missing tranche execution schedule")
    
    execution_at = _clock([r["execution_at"] for r in records])
    signal_at = pd.DatetimeIndex(utc([r["signal_at"] for r in records]))
    execution = calendar.get_indexer(execution_at)
    if (execution < 0).any():
        raise ValueError("Tranche execution absent from calendar")
    if (execution_at <= signal_at).any():
        raise ValueError("A tranche cannot execute at or before its signal date")
    if not signal_at.is_monotonic_increasing:
        raise ValueError("Tranche signal dates must be ordered")
    
    for signal in signal_at.unique():
        actual = execution[signal_at == signal]
        first = calendar.searchsorted(signal, side="right")
        if not np.array_equal(actual, np.arange(first, first + len(actual))):
            raise ValueError("Tranches must start at the next observation and execute consecutively")
        if len(actual) > metadata["k"]:
            raise ValueError("Too many steps for a tranche signal")
        
    return execution, signal_at


def monthly_signal_dates(calendar, offset=0):
    #select monthly closes, 0 last session, n=n sessions earlier

    offset = _integer(offset, "offset", 0)
    calendar = _clock(calendar)
    months = calendar.tz_localize(None).to_period("M")
    return pd.DatetimeIndex([group[max(0, len(group) - 1 - offset)]
                             for month in months.unique()
                             for group in [calendar[months == month]]])


def rebalance_sensitivity(strategy_fn, offsets=range(21), *, periods_per_year=252):
    #call strategy_fn(offset) and summarise net returns on identical dates

    from foundation.reporting import metrics

    offsets = [_integer(x, "offset", 0) for x in offsets]
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("Nonempty unique offsets required")
    
    rows = []
    common_dates = None
    for offset in offsets:
        result = strategy_fn(offset)
        returns = result["ledger"]["net_return"] if isinstance(result, dict) else result
        if not isinstance(returns, pd.Series):
            raise ValueError("strategy_fn must return net dated returns or a backtest result")
        
        dates = _clock(returns.index)
        if common_dates is None:
            common_dates = dates
        elif not dates.equals(common_dates):
            raise ValueError("Sensitivity runs require exact common dates")
        
        performance = metrics(returns, periods_per_year)
        row = {"offset": offset, **performance}
        if isinstance(result, dict):
            ledger = result["ledger"]
            row.update(total_cost=float(ledger.cost.sum()),
                       traded_notional=float(ledger.traded_notional.sum()),
                       trades=len(result["trades"]))
        rows.append(row)
    spread = {}
    for key in ("cagr", "sharpe", "max_drawdown"):
        values = [r[key] for r in rows]
        spread[key] = None if any(v is None for v in values) else max(values) - min(values)
    return {"offsets": offsets, "periods_per_year": periods_per_year,
            "start": common_dates[0].isoformat(), "end": common_dates[-1].isoformat(),
            "observations": len(common_dates), "per_offset": rows, "spread": spread}
