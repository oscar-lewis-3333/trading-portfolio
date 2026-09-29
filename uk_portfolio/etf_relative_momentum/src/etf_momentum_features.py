import numpy as np
import pandas as pd


def build_momentum_scores(monthly_prices, lookback_months=12):
    if monthly_prices.empty:
        raise ValueError("Provide nonempty monthly prices.")
    if (isinstance(lookback_months, bool) or not isinstance(lookback_months, int) or lookback_months < 1):
        raise ValueError("lookback_months must be a positive integer.")

    months = monthly_prices.index.to_period("M")
    expected = pd.period_range(months.min(), months.max(), freq="M")
    if not months.equals(expected):
        raise ValueError("Prices must contain consecutive, ordered months.")

    prices = monthly_prices.astype(float)
    invalid = prices.notna() & (prices.le(0) | ~np.isfinite(prices))
    if invalid.any().any():
        raise ValueError("Observed prices must be positive and finite.")

    complete_window = prices.notna().rolling(lookback_months + 1, min_periods=lookback_months + 1).sum().eq(lookback_months + 1)
    momentum = prices / prices.shift(lookback_months) - 1
    return momentum.where(complete_window)