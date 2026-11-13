import numpy as np
import pandas as pd
import pytest
from foundation.signals.weekly_beta import build_rolling_beta, build_weekly_return_panels, build_lse_schedule

def test_beta_uses_matched_market_denominator_and_full_window_age():
    dates = pd.date_range('2020-01-03', periods=8, freq='W-FRI')
    market = pd.Series([.01, .02, -.01, .04, .03, -.02, .05, -.03], index=dates)
    stock = pd.DataFrame({'A': 2 * market}, index=dates)

    stock.iloc[2] = np.nan
    beta, count = build_rolling_beta(stock, market, lookback=4, min_observations=3)
    assert beta.iloc[:3].isna().all().all()
    assert count.iloc[3, 0] == 3

    np.testing.assert_allclose(beta.iloc[3:, 0], 2, rtol=1e-12)
    pairs = pd.concat([stock.A, market], axis=1).iloc[:4].dropna()
    assert beta.iloc[3, 0] == pairs.iloc[:, 0].cov(pairs.iloc[:, 1]) / pairs.iloc[:, 1].var()

def test_zero_variance_gaps_and_future_mutations():
    dates = pd.date_range('2020-01-03', periods=10, freq='W-FRI')
    market = pd.Series(np.arange(10.) / 100, index=dates)
    stock = pd.DataFrame({'A': market}, index=dates)
    first = build_rolling_beta(stock, market, 4, 3)[0]

    changed = stock.copy(); changed.iloc[7:] *= -100
    pd.testing.assert_frame_equal(first.iloc[:7], build_rolling_beta(changed, market, 4, 3)[0].iloc[:7], check_exact=True)
    assert build_rolling_beta(stock, market * 0, 4, 3)[0].isna().all().all()

    with pytest.raises(ValueError):
        build_rolling_beta(stock, market.iloc[1:], 4, 3)
    with pytest.raises(ValueError):
        build_rolling_beta(stock, market, 4, 5)

def test_incomplete_week_and_missing_close_are_not_filled():
    schedule = build_lse_schedule('2024-01-02', '2024-01-25')
    index = pd.MultiIndex.from_product([schedule.index, ['A']], names=['Date','ticker'])
    prices = pd.DataFrame({'adj_close': np.arange(len(index))+100., 'close_reference_usable': True, 'positive_reported_volume': True}, index=index)
    proxy = pd.DataFrame({'Adj Close': np.arange(len(schedule))+100., 'Volume': 10.}, index=schedule.index)
    
    prices.loc[(pd.Timestamp('2024-01-12'), 'A'), 'adj_close'] = np.nan
    stocks, market = build_weekly_return_panels(prices, proxy, schedule)
    assert stocks.index.max() == pd.Timestamp('2024-01-19')
    assert stocks.A.isna().all()
    assert market.iloc[1:].notna().all()
