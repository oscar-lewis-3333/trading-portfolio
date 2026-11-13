import numpy as np
import pandas as pd
import pytest
from foundation.vol_regime import *
from foundation.models import Protocol


def test_har_exact_windows_and_prefix():
    ix=pd.date_range('2020-01-01',periods=40,tz='UTC'); rv=pd.Series(np.arange(40)+1.,index=ix)
    f=har_features(rv)

    assert f.iloc[21].monthly==11.5 and f.iloc[21].weekly==20
    pd.testing.assert_frame_equal(f.iloc[:30],har_features(rv.iloc[:30]))


def test_qlike_perfect_and_bad():
    np.testing.assert_allclose(qlike([1,2],[1,2]),[0,0])
    assert qlike([1],[2])[0]>0
    with pytest.raises(ValueError):
        qlike([1],[0])


def test_har_forecasts_all_models_causal():
    ix=pd.date_range('2020-01-01',periods=180,tz='UTC'); rv=pd.Series(np.exp(np.random.default_rng(2).normal(-8,.3,180)),index=ix)
    r=forecast(rv,protocol=Protocol(min_train=30,retrain_every=30))
    assert set(r['mean_qlike'])=={'har','loghar','xgboost'}

    assert np.isfinite(list(r['mean_qlike'].values())).all()
    assert all(pd.Timestamp(f['last_label_end'])<pd.Timestamp(f['cutoff']) for f in r['fits'])
    assert 'lower_95' in compare_qlike(r['losses'],block_size=5,resamples=100)


def test_gates_availability_and_gamma_unknown():
    f=pd.DataFrame({'available_at':['2020-01-01T00:00Z','2020-01-03T00:00Z'],'trend_ratio':[.1,-.1]})
    m=pd.DataFrame({'available_at':['2020-01-01T00:00Z'],'event_at':['2020-01-02T00:10Z']})

    g=gamma_proxy(pd.DataFrame(),at='2020-01-02T00:00Z')
    result=gates('2020-01-02T00:00Z',features=f,macro_calendar=m,gamma=g)
    assert result=={'trend':True,'outside_macro_window':False,'positive_gamma_proxy':None}


def test_gamma_units():
    f=pd.DataFrame({'gamma':[.01],'open_interest':[10.],'spot':[100.],'multiplier':[100.],'available_at':['2020-01-01T00:00Z']})
    g=gamma_proxy(f,at='2020-01-02T00:00Z',inventory_sign=[-1],assumption='explicit synthetic short inventory')
    assert g.value==-1000
