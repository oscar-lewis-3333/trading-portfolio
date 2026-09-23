import numpy as np
import pandas as pd


def build_liquidity_features(prices, schedule, lookback=60):
    #measure trailing liquidity using information available after each close
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer.")

    prices = prices.sort_index()
    if (prices.index.names != ["Date", "ticker"] or not prices.index.is_unique):
        raise ValueError("Expected unique (Date, ticker) rows.")

    tickers = sorted(prices.index.get_level_values("ticker").unique())
    expected_index = pd.MultiIndex.from_product([schedule.index, tickers], names=["Date", "ticker"])
    if not prices.index.equals(expected_index):
        raise ValueError("Prices must retain every scheduled session per ticker.")

    volume = prices["Volume"]
    traded_value = prices["traded_value_gbp"]
    valid = (prices["observed_row"].eq(True) & prices["close_reference_usable"].eq(True)& np.isfinite(volume)
            & volume.ge(0) & np.isfinite(traded_value) & traded_value.ge(0))
    #zero-volume obs count as zero
    activity = volume.gt(0).astype(float).where(valid)
    traded_value = traded_value.where(valid)

    features = pd.DataFrame(index=prices.index)
    features["median_traded_value_gbp"] = (traded_value.groupby(level="ticker").transform(
            lambda series: series.rolling(lookback, min_periods=lookback).median()))
    features["positive_volume_fraction"] = activity.groupby(level="ticker").transform(
            lambda series: series.rolling(lookback, min_periods=lookback).mean())
    features["liquidity_history_complete"] = features.notna().all(axis=1)

    return features

def build_raw_momentum_features(prices, schedule, formation_sessions=252, skip_sessions=21):
    #calculate trailing momentum while excluding the most recent sessions
    for value in [formation_sessions, skip_sessions]:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Session counts must be integers.")

    if not 0 <= skip_sessions < formation_sessions:
        raise ValueError("Require 0 <= skip_sessions < formation_sessions.")

    prices = prices.sort_index()

    if (prices.index.names != ["Date", "ticker"] or not prices.index.is_unique):
        raise ValueError("Expected unique (Date, ticker) rows.")

    tickers = sorted(prices.index.get_level_values("ticker").unique())
    expected_index = pd.MultiIndex.from_product([schedule.index, tickers], names=["Date", "ticker"])
    if not prices.index.equals(expected_index):
        raise ValueError("Prices must retain every scheduled session per ticker.")

    close = prices["adj_close"].where(prices["close_reference_usable"].eq(True) & np.isfinite(prices["adj_close"]) & prices["adj_close"].gt(0))
    by_ticker = close.groupby(level="ticker")
    start_price = by_ticker.shift(formation_sessions)
    end_price = by_ticker.shift(skip_sessions)


    #require every closing observation between two endpoints (231-session return interval needs 232 closing prices)
    required_prices = formation_sessions - skip_sessions + 1
    valid_count = close.notna().groupby(level="ticker").transform(lambda series: (
            series.rolling(required_prices, min_periods=required_prices).sum().shift(skip_sessions)))

    complete = valid_count.eq(required_prices)
    return pd.DataFrame({"formation_start_price": start_price,
            "formation_end_price": end_price,
            "formation_history_complete": complete,
            "raw_momentum_score": (end_price / start_price - 1).where(complete)}, index=prices.index)

def build_momentum_sweep_features(prices, schedule, liquidity, parameter_grid):
    #build momentum variants with a shared eligibility rule
    if not liquidity.index.equals(prices.index):
        raise ValueError("Liquidity and prices must have matching indexes.")

    feature_pairs = parameter_grid[["formation_sessions", "skip_sessions"]].drop_duplicates()
    if feature_pairs.empty:
        raise ValueError("The parameter grid is empty.")

    common_eligible = liquidity["liquidity_eligible"].eq(True).copy()
    feature_panels = {}
    for formation, skip in feature_pairs.itertuples(index=False, name=None):
        momentum = build_raw_momentum_features(prices, schedule, formation_sessions=formation, skip_sessions=skip)
        signal_available = (momentum["formation_history_complete"].eq(True) & np.isfinite(momentum["raw_momentum_score"]))

        common_eligible &= signal_available
        feature_panels[(formation, skip)] = momentum[["raw_momentum_score"]].copy()

    #every variant ranks the same eligible stocks on each date
    for panel in feature_panels.values():
        panel["eligible"] = common_eligible

    return feature_panels, common_eligible.rename("eligible")