import numpy as np
import pandas as pd

from .contracts import validate_panel, validate_schedule


def pivot_closes(panel):
    #wide close matrix. any gaps are left as NaN
    
    validate_panel(panel)
    closes = panel.pivot(index="timestamp", columns="product_id", values="close")
    closes = closes.sort_index().sort_index(axis=1)
    closes.columns.name = None

    return closes


def pivot_eligible(panel):
    #wide eligibility matrix. any missing observations are not eligible
    validate_panel(panel)
    eligible = panel.pivot(index="timestamp", columns="product_id", values="eligible")
    eligible = eligible.sort_index().sort_index(axis=1).astype("boolean").fillna(False).astype(bool)
    eligible.columns.name = None

    return eligible


def daily_returns(closes):
    #close-to-close daily returns. gap yields NaN, not bridged return
    if not isinstance(closes, pd.DataFrame):
        raise ValueError("Closes must be a DataFrame.")
    if not closes.index.is_monotonic_increasing or closes.index.has_duplicates:
        raise ValueError("Close timestamps must be unique and sorted.")

    return closes.pct_change(fill_method=None)


def as_of(panel, decision_at):
    #each signal must only see complete observations strictly before the decision time

    validate_panel(panel)
    stamp = pd.Timestamp(decision_at)
    if stamp.tzinfo is None:
        raise ValueError("The decision stamp must be timezone-aware.")

    return panel.loc[panel["timestamp"] < stamp.tz_convert("UTC")]


def trailing_return(closes, decision_at, lookback_days):
    #close-to-close returns over the past lookback_days (only valid observations used)
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days < 1:
        raise ValueError("lookback_days must be a positive integer.")

    history = closes.loc[closes.index < decision_at]
    if len(history) <= lookback_days:
        return pd.Series(np.nan, index=closes.columns, dtype=float)

    latest = history.iloc[-1]
    earlier = history.iloc[-(lookback_days + 1)]
    result = (latest / earlier) - 1.0

    #a gap at either endpoint means no observable return
    return result.where(np.isfinite(result.to_numpy(dtype=float)))


def realised_volatility(returns, decision_at, window_days, periods_per_year=365):
    #volatility calculated over the past window_days, using only valid observations
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 2:
        raise ValueError("window_days must be an integer of at least two.")

    history = returns.loc[returns.index < decision_at].tail(window_days)
    if len(history) < window_days:
        return pd.Series(np.nan, index=returns.columns, dtype=float)

    return history.std(ddof=1) * np.sqrt(periods_per_year)


def eligible_at(eligible, decision_at):
    #membershup uses the most recent valid observation before the decision time
    history = eligible.loc[eligible.index < decision_at]
    if history.empty:
        return pd.Series(False, index=eligible.columns, dtype=bool)

    return history.iloc[-1].astype(bool)


def weekly_schedule(panel, weekday=0):
    #shared schedule each signal follows, with one monday midnight decision time

    validate_panel(panel)
    if isinstance(weekday, bool) or not isinstance(weekday, int) or not 0 <= weekday <= 6:
        raise ValueError("weekday must be an integer from 0 to 6.")

    stamps = panel["timestamp"]
    freq = f"W-{['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][weekday]}"
    schedule = pd.date_range(stamps.min(), stamps.max(), freq=freq, tz="UTC", name="decision_at")

    return validate_schedule(schedule)
