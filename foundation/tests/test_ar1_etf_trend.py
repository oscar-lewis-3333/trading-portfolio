from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from foundation.signals import etf_trend as port
def synthetic_inputs(start="2010-01-01", end="2016-12-30", *, rising=True):
    #non-constant volatility with reproducible, strictly positive price paths
    dates = pd.bdate_range(start, end, name="Date")
    phase = np.arange(len(dates), dtype=float)
    tickers = ["EQ1", "EQ2", "BND", "REAL"]
    daily = np.column_stack([
        0.0020 + 0.0009 * np.sin(phase / 7),
        0.0018 + 0.0010 * np.cos(phase / 9),
        0.0013 + 0.0005 * np.sin(phase / 11),
        (0.0016 if rising else -0.0016) + 0.0007 * np.cos(phase / 13),
    ])
    close = pd.DataFrame(100 * np.cumprod(1 + daily, axis=0), index=dates, columns=tickers)
    opened = close * 0.9998
    cash = pd.DataFrame({
        "cash_reference_open": np.exp(phase * 0.00001),
        "cash_reference_close": np.exp(phase * 0.00001),
        "cash_source": "Bank Rate",
    }, index=dates)
    return {"adjusted_close": close,
        "execution_open": opened.copy(),
        "valuation_open": opened.copy(),
        "valuation_close": close.copy(),
        "cash": cash,
        "cash_execution_allowed": pd.Series(True, index=dates),
        "bank_rate_hurdle": cash["cash_reference_close"].copy(),
        "universe": pd.DataFrame({"asset_class": ["equities", "equities", "bonds", "real_assets"]}, index=tickers)}


def test_legacy_lookback_preserves_missing_months_and_zero_returns():
    dates = pd.date_range('2020-01-31', periods=20, freq='ME')
    prices = pd.DataFrame({'up':np.arange(20.)+100, 'flat':100.}, index=dates)
    prices.iloc[7, 0] = np.nan
    returns, signal = port.build_trend_features(prices, 3)
    assert signal.iloc[7:11, 0].isna().all()
    assert signal.iloc[3:, 1].eq(0).all()
    assert returns.iloc[:3].isna().all().all()
    with pytest.raises(ValueError):
        port.build_trend_features(prices.drop(dates[5]), 3)
    with pytest.raises(ValueError):
        port.build_trend_features(prices, True)

def test_phase_a_signal_prefix_and_execution_clock():
    inputs = synthetic_inputs(end='2013-12-31')
    first = port.build_allocation_targets(inputs)
    altered = deepcopy(inputs)
    cutoff = pd.Timestamp('2012-12-31')
    altered['adjusted_close'].loc[cutoff+pd.Timedelta(days=1):] *= .01
    second = port.build_allocation_targets(altered)
    for key in first['signals']:
        pd.testing.assert_frame_equal(first['signals'][key].loc[:cutoff], second['signals'][key].loc[:cutoff], check_exact=True)
    schedule = first['decision_schedule']
    assert (schedule.execution_date > schedule.index).all()
    for weights in first['targets'].values():
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-14)
        assert weights.ge(0).all().all()
