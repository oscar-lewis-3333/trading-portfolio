import numpy as np
import pandas as pd
import pytest
from foundation.models import *
from foundation.costs import Trade, CostModel
from foundation.validate import TrialLog


def data():
    ix = pd.date_range('2020-01-01',periods=120,tz='UTC'); rng=np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(120,3)),index=ix,columns=list('abc')); y=(X.a*.01).rename('y')
    return X,y


@pytest.mark.parametrize('family',['ridge','xgboost'])
def test_walkforward_never_uses_unobserved_labels(family):
    X,y=data()
    end=X.index+pd.Timedelta('3D')
    r = fit_predict(X,y,end,available_at=X.index,protocol=Protocol(min_train=20,retrain_every=20,half_life=30),family=family)
    assert r['prediction'].notna().sum()>0
    assert all(pd.Timestamp(f['last_label_end'])<pd.Timestamp(f['cutoff']) for f in r['fits'])

    mutated=y.copy(); mutated.iloc[-5:]=100
    second=fit_predict(X,mutated,end,available_at=X.index,protocol=Protocol(min_train=20,retrain_every=20,half_life=30),family=family)
    pd.testing.assert_series_equal(r['prediction'],second['prediction'])


def test_availability_leak_and_decay():
    X,y=data() 
    with pytest.raises(ValueError):
        feature_audit(X,X.index+pd.Timedelta('1D'),X.index)
    w = decay_weights(pd.date_range('2020-01-01',periods=3,tz='UTC'),1)
    assert w[-1]/w[0]==4


def test_barriers_ambiguous_and_label_clock():
    ix = pd.date_range('2020-01-01',periods=5,tz='UTC')
    b = pd.DataFrame({'open':100.,'high':[100,103,100,100,100],'low':[100,100,97,100,100],'close':100.},index=ix)
    e = pd.DataFrame({'entry_at':[ix[0]],'side':[1]})
    r = triple_barrier(b,e,max_holding=3)
    assert r.label.iloc[0]==1 and r.label_end.iloc[0]==ix[2]
    b.loc[ix[1],'low']=97
    assert triple_barrier(b,e,max_holding=3).status.iloc[0]=='ambiguous_both'


def test_calibration_and_late_predictions():
    X,y = data(); classes=(y>0).astype(int)
    model = CalibratedMetaModel('ridge').fit(X.iloc[:100],classes.iloc[:100],X.index[:100]+pd.Timedelta('1D'),available_at=X.index[:100],calibration_start=X.index[60],ready_at=X.index[100])
    p = model.predict_probability(X.iloc[100:],available_at=X.index[100:]); assert p.between(0,1).all()
    with pytest.raises(ValueError):
        model.predict_probability(X.iloc[:5],available_at=X.index[:5])


def test_cost_gate_uses_two_sides_and_net_no_double_count():
    t=Trade('A','simulation','buy',10,100,'GBP',pd.Timestamp('2020-01-01T00:00Z'),sdrt='exempt')
    assert cost_gated_size(.8,.001,t,CostModel(10))['size']==0
    assert cost_gated_size(.8,.003,t,CostModel(10))['size']==pytest.approx(.6)
    assert cost_gated_size(.8,.001,t,CostModel(10),forecast_is_net=True)['size']==pytest.approx(.6)


def test_ladder_identical_rungs_cannot_promote(tmp_path):
    _,s=data()
    frames={name:s for name in ['rank_average','ridge','xgboost']}
    r = compare_ladder(frames,s,project='test',protocol={'v':1},trial_log=TrialLog(tmp_path/'trials.jsonl'),resamples=100,block_size=5)
    assert not any(v['passes_research_gate'] for v in r.values())


def test_bar_availability_cannot_leak_ohlc_label():
    ix = pd.date_range('2020-01-01',periods=2,tz='UTC')
    bars = pd.DataFrame({'open':[100.,100.],'high':[103.,103.],'low':[100.,100.],'close':[102.,102.],'available_at':ix},index=ix)
    with pytest.raises(ValueError,match='bar open'):
        triple_barrier(bars,pd.DataFrame({'entry_at':[ix[0]],'side':[1]}),max_holding=1)
    bars['available_at']=ix+pd.Timedelta('1D')
    result=triple_barrier(bars,pd.DataFrame({'entry_at':[ix[0]],'side':[1]}),max_holding=1)
    assert result.label.iloc[0]==1 and result.label_end.iloc[0]==ix[1]
