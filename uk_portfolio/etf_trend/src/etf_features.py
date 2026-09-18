import numpy as np
import pandas as pd


def build_trend_features(monthly_prices, lookback_months=12):
    
    #return trailing returns and month-end long/cash signals
    if monthly_prices.empty:
        raise ValueError("Provide nonempty monthly prices.")
    if (isinstance(lookback_months, bool) or not isinstance(lookback_months, int) or lookback_months < 1):
        raise ValueError("lookback_months must be a positive integer.")

    months = monthly_prices.index.to_period("M")
    expected = pd.period_range(months.min(), months.max(), freq="M")

    if not months.equals(expected):
        raise ValueError("Prices must contain consecutive, ordered months.")

    prices = monthly_prices.astype(float)
    invalid = prices.notna() & ((prices <= 0) | ~np.isfinite(prices))
    if invalid.any().any():
        raise ValueError("Observed prices must be positive and finite.")
    
    trailing_return = prices / prices.shift(lookback_months) - 1

    #12 monthly returns require 13 consecutive price observations
    complete_window = prices.notna().rolling(lookback_months + 1, min_periods=lookback_months + 1).sum().eq(lookback_months + 1)

    trailing_return = trailing_return.where(complete_window)
    signal = trailing_return.gt(0).astype(float).where(trailing_return.notna())

    return trailing_return, signal