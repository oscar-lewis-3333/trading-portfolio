import pandas as pd


def add_momentum_features(prices, lookback_days=90):
    #add momentum to already audited daily price panel
    if (isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days < 1):
        raise ValueError("lookback_days must be a positive integer.")

    panel = prices.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if panel.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Duplicate asset/date observations.")

    date_gaps = panel.groupby("symbol")["timestamp"].diff()
    if not date_gaps.dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Each asset must have consecutive daily prices.")

    lagged_close = panel.groupby("symbol")["close"].shift(lookback_days)
    panel["momentum_return"] = (panel["close"] / lagged_close - 1)
    panel["signal_available_at"] = panel["timestamp"] + pd.Timedelta(days=1)
    panel["trend_positive"] = panel["momentum_return"].gt(0).astype("boolean").mask(panel["momentum_return"].isna())

    return panel