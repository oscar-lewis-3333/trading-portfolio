import numpy as np
import pandas as pd
import pytest
from foundation.signals import crypto_trend as port
from foundation._research.synthetic import make_panel
from foundation._research import weekly_schedule

@pytest.mark.parametrize('builder', [port.trend_agreement, port.volatility_scaled_trend, port.trend_agreement_vol])
def test_prefix_invariance_cash_slots_and_bounds(builder):
    panel = make_panel(assets=3, days=280, seed=19, gap_probability=.02)
    schedule = weekly_schedule(panel)
    full = builder(panel, schedule)
    cutoff = schedule[len(schedule)//2]
    prefix = builder(panel.loc[panel.timestamp < cutoff], schedule[schedule <= cutoff])
    pd.testing.assert_frame_equal(full.loc[:cutoff], prefix, check_exact=True)
    np.testing.assert_allclose(full.sum(axis=1), 1, rtol=0, atol=1e-15)
    assert full.ge(0).all().all() and full.le(1).all().all()

def test_bad_configuration_rejected():
    panel = make_panel(assets=2, days=220)
    schedule = weekly_schedule(panel)
    for kwargs in ({'lookbacks': ()}, {'target_volatility': 0}, {'no_trade_band': 1}):
        with pytest.raises(ValueError): port.trend_agreement_vol(panel, schedule, **kwargs)
