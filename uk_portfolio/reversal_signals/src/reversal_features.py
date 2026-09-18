import numpy as np
import pandas as pd

def build_forward_outcomes(panel, schedule, horizon=5):

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")

    required = {"ticker", "Open", "Dividends"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if panel.empty or panel["ticker"].isna().any():
        raise ValueError("Provide a nonempty panel with valid ticker labels.")
    if (schedule.empty or not schedule.index.is_unique or not schedule.index.is_monotonic_increasing):
        raise ValueError("Schedule must be nonempty, unique and sorted.")
    if "Capital Gains" in panel:
        if panel["Capital Gains"].fillna(0).ne(0).any():
            raise ValueError("Capital-gain distributions need separate handling.")

    frames = []

    for ticker, df in panel.groupby("ticker", sort=False):
        df = df.sort_index()
        if not df.index.equals(schedule.index):
            raise ValueError(f"{ticker} is not aligned to the full schedule.")

        opens = df["Open"].astype(float)
        dividends = df["Dividends"].astype(float)
        if (opens.le(0) | np.isinf(opens)).any():
            raise ValueError(f"{ticker} contains invalid opening prices.")
        if (dividends.lt(0) | np.isinf(dividends)).any():
            raise ValueError(f"{ticker} contains invalid dividend amounts.")

        out = pd.DataFrame(index=df.index)
        out["ticker"] = ticker
        out["horizon_sessions"] = horizon
        out["signal_time"] = schedule["market_close"]
        out["entry_time"] = schedule["market_open"].shift(-1)
        out["exit_time"] = schedule["market_open"].shift(-(horizon + 1))
        out["entry_open"] = opens.shift(-1)
        out["exit_open"] = opens.shift(-(horizon + 1))

        #exclude entry day divedends, but include exit day. series additions preservs NaN

        out["dividends_earned"] = sum(dividends.shift(-offset)for offset in range(2, horizon + 2))
        out["forward_return"] = (out["exit_open"] + out["dividends_earned"]) / out["entry_open"] - 1
        out["outcome_observed"] = np.isfinite(out["forward_return"])
        out["forward_return"] = out["forward_return"].where(out["outcome_observed"])

        frames.append(out)
    return pd.concat(frames).sort_index()


def build_raw_reversal_features(panel, schedule, lookback=5):

    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("Lookback must be a positive integer.")

    required = {"ticker", "adj_close"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if panel.empty or panel["ticker"].isna().any():
        raise ValueError("Provide a nonempty panel with valid ticker labels.")

    if (schedule.empty or not schedule.index.is_unique or not schedule.index.is_monotonic_increasing):
        raise ValueError("Schedule must be nonempty, unique and sorted.")

    frames = []

    for ticker, df in panel.groupby("ticker", sort=False):
        df = df.sort_index()
        if not df.index.equals(schedule.index):
            raise ValueError(f"{ticker} is not aligned to the full schedule.")

        close = df["adj_close"].astype(float)
        if (close.le(0) | np.isinf(close)).any():
            raise ValueError(f"{ticker} contains invalid adjusted closes.")

        out = pd.DataFrame(index=df.index)
        out["ticker"] = ticker
        out["signal_time"] = schedule["market_close"]
        out["formation_sessions"] = lookback

        #compare today adjusted close to the close from 5 days ago, then the reversal score is the negative of that return, as we seek reversal signals (to rank tickers by their reversal potential).
        out["raw_return"] = close / close.shift(lookback) - 1
        out["raw_reversal_score"] = -out["raw_return"]

        frames.append(out)

    return pd.concat(frames).sort_index()

def build_liquidity_features(activity, schedule, lookback=60):
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("Lookback must be a positive integer.")

    required = {"ticker", "traded_value_proxy_gbp", "reported_zero_volume"}
    missing = required.difference(activity.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if activity.empty or activity["ticker"].isna().any():
        raise ValueError("Provide nonempty activity data with valid tickers.")

    frames = []

    for ticker, df in activity.groupby("ticker", sort=False):
        df = df.sort_index()
        if not df.index.equals(schedule.index):
            raise ValueError(f"{ticker} is not aligned to the schedule.")

        value = df["traded_value_proxy_gbp"].astype(float)
        if (value.lt(0) | np.isinf(value)).any():
            raise ValueError(f"{ticker} contains invalid traded values.")

        #todays observation can't enter todays background estimate
        value_window = value.shift(1).rolling(lookback, min_periods=lookback)
        zero_window = df["reported_zero_volume"].astype(float).shift(1).rolling(lookback, min_periods=lookback)

        out = pd.DataFrame(index=df.index)
        out["ticker"] = ticker
        out["lookback_sessions"] = lookback
        out["mean_traded_value_gbp"] = value_window.mean()
        out["median_traded_value_gbp"] = value_window.median()
        out["zero_volume_fraction"] = zero_window.mean()

        out["history_ready"] = out[[
                "mean_traded_value_gbp",
                "median_traded_value_gbp",
                "zero_volume_fraction"
            ]].notna().all(axis=1)

        frames.append(out)

    return pd.concat(frames).sort_index()