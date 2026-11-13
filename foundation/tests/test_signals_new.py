import numpy as np
import pandas as pd
import pytest
from foundation.signals import *
from foundation.costs import CostModel


def test_registry_filters_future_revisions_and_lag():
    r=Registry() 
    r.register(SignalSpec('a','test mechanism',('v',),'1D'),lambda f, at:f.groupby('instrument_id').v.last())
    panel=pd.DataFrame({'instrument_id':['A','A'],'v':[1,99],'event_at':['2020-01-01T00:00Z']*2,'available_at':['2020-01-01T00:00Z','2020-01-03T00:00Z']})
    assert r.build('a',panel,'2020-01-03T00:00Z').signal.item()==1


def test_normalisation_missing_and_neutral_residual_orthogonality():
    f=pd.DataFrame({'decision_at':['2020']*20,'signal':np.arange(20.)**2,'sector':['a','b']*10,'size':np.arange(20)+1,'beta':np.arange(20.)/20})
    f.loc[0,'signal']=np.nan
    assert normalise(f,['signal']).signal_missing.sum()==1
    g=neutralise(f); ok=~g.neutral_missing

    assert abs(g.loc[ok,'neutral_signal'].sum())<1e-8
    assert abs(np.dot(g.loc[ok,'neutral_signal'],g.loc[ok,'beta']))<1e-8


def test_exact_cost_conservation_and_next_bar_fill():
    dates=pd.date_range('2020-01-01',periods=4,tz='UTC')
    prices=pd.DataFrame({'A':[10.,10,11,11]},index=dates)
    targets=pd.DataFrame({'A':[1.,0.],'CASH':[0.,1.]},index=dates[[0,2]])

    result=backtest_targets(targets,prices,CostModel(10),instruments={'A':dict(venue='simulation',currency='GBP',sdrt='exempt')})
    assert result['trades'].at[0,'at']==dates[1]
    expected=10000/1.001*1.1*.999

    assert result['ledger'].equity.iloc[-1]==pytest.approx(expected)
    assert result['ledger'].cash.iloc[-1]==pytest.approx(expected)
    assert result['ledger'].cost.sum()==pytest.approx(result['trades'].total.sum())


def test_event_delayed_entry_exit_and_cash():
    ix=pd.date_range('2020-01-01',periods=6,tz='UTC')
    q=pd.DataFrame({'instrument_id':['A']*6,'at':ix,'price':[10.,10,10,12,12,12],'tradable':[True,False,True,False,True,True]})
    ev=pd.DataFrame({'event_id':['e'],'instrument_id':['A'],'available_at':[ix[0]]})
    out=run_events(ev,q,1,CostModel(0),instruments={'A':dict(venue='simulation',currency='GBP',sdrt='exempt')},slots=1)

    assert out['trades']['at'].tolist()==[ix[2],ix[4]]
    assert out['ledger'].equity.iloc[-1]==12000 and not out['open_positions']
    assert out['ledger'].cash.iloc[0]==10000


def test_event_end_censoring_not_fake_sale():
    ix=pd.date_range('2020-01-01',periods=2,tz='UTC')
    q=pd.DataFrame({'instrument_id':['A']*2,'at':ix,'price':[10.,11.],'tradable':True})
    ev=pd.DataFrame({'event_id':['e'],'instrument_id':['A'],'available_at':[ix[0]]})
    
    out=run_events(ev,q,2,CostModel(0),instruments={'A':dict(venue='simulation',currency='GBP',sdrt='exempt')},slots=1)
    assert len(out['trades'])==1 and out['outcomes'].status.item()=='censored_open'
