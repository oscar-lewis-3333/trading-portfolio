from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from foundation.costs import *


def trade(**kw):
    return Trade(**(dict(instrument_id='A',venue='trading212',side='buy',quantity=100,price=10, currency='GBP',at=pd.Timestamp('2026-10-07T12:00Z'),sdrt='liable')|kw))


@pytest.mark.parametrize('side,stamp',[('buy',5),('sell',0)])
def test_stamp_and_spread(side,stamp):
    c=CostModel(10).quote(trade(side=side))
    assert c.sdrt == stamp and c.spread_slippage == 1


def test_exempt_and_unknown():
    assert CostModel().quote(trade(sdrt='exempt')).sdrt==0
    with pytest.raises(ValueError):
        CostModel().quote(trade(sdrt='unknown'))
    assert CostModel().quote(trade(asset_type='etf',sdrt='unknown')).sdrt==0


@pytest.mark.parametrize('side',['buy','sell'])
def test_fx_both_sides(side):
    assert CostModel().quote(trade(market='US',currency='USD',side=side,fx_conversion=True)).fx==1.5


@pytest.mark.parametrize('liquidity,fee',[('maker',0),('taker',.9)])
def test_revolut(liquidity,fee):
    assert CostModel(0).quote(trade(venue='revolut_x',asset_type='crypto',liquidity=liquidity)).commission==fee


def test_ibkr_minimum_cap_fx_and_contract():
    m = CostModel(0,futures_per_contract={'MES':.25})
    assert m.quote(trade(venue='ibkr',market='US',currency='USD')).commission==1
    assert m.quote(trade(venue='ibkr',market='US',currency='USD',quantity=10,price=.2)).commission==.02
    c = m.quote(trade(venue='ibkr',sdrt='exempt',fx_conversion=True,usd_per_currency=1.25))
    assert c.commission==1 and c.fx==1.6
    assert m.quote(trade(venue='ibkr',instrument_id='MES',asset_type='future',quantity=2,multiplier=5)).commission==.5
    with pytest.raises(ValueError):
        m.quote(trade(venue='ibkr',asset_type='future'))


def test_measured_replaces_only_spread_and_blocks_future():
    f=pd.DataFrame([dict(instrument_id='A',venue='trading212',side='both',currency='GBP',per_side_bps=2,
                        max_notional=2000,available_at='2026-10-01T00:00Z',valid_from='2026-10-01T00:00Z',valid_to='2026-11-01T00:00Z')])
    model=CostModel(30,measured=MeasuredTable(f))

    assert model.quote(trade()).total == 5.2
    with pytest.raises(ValueError):
        model.quote(trade(at=pd.Timestamp('2026-09-01T00:00Z')))
    with pytest.raises(ValueError):
        model.quote(trade(quantity=300))


def test_crossings_label_extrapolation():
    assert break_even_cost([0,10,20,30],[4,3,2,1])['bps_per_side']==40
    assert 'extrapolation' in break_even_cost([0,10,20,30],[4,3,2,1])['method']
    assert break_even_cost([0,10,20,30],[2,1,-1,-2])['bps_per_side']==15

    assert replay_break_even(lambda bp:1-bp/23)['bps_per_side']==pytest.approx(23)
    assert break_even_cost([0,10],[-1,-2])['bps_per_side'] is None


def test_quote_fit_and_attribution():
    q = pd.DataFrame([dict(instrument_id='A',venue='trading212',currency='GBP',bid=99,ask=101,received_at='2026-10-01T00:00Z')])
    table=fit_quote_table(q,available_at='2026-10-02T00:00Z',valid_from='2026-10-02T00:00Z',valid_to='2027-01-01T00:00Z',max_notional=2000)
    assert table.rate(trade())==.01
    result = CostModel(10).attribute([trade(),trade(side='sell')])
    assert result.total.sum()==7 and result.sdrt.sum()==5
