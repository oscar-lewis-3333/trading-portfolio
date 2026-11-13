import numpy as np
import pandas as pd
import pytest
from foundation.portfolio import *


def test_inverse_and_hrp_diagonal():
    w=inverse_vol({'a':.1,'b':.2})
    assert w.a==pytest.approx(2/3)
    cov=pd.DataFrame([[1.,0],[0,4.]],index=['a','b'],columns=['a','b'])
    
    h=hrp(cov); assert h.a==pytest.approx(.8) and h.sum()==pytest.approx(1)
    pd.testing.assert_series_equal(h.sort_index(),hrp(cov.iloc[::-1,::-1]).sort_index())


def test_covariance_positive_and_vol_target_lag():
    rng=np.random.default_rng(2)
    r=pd.DataFrame(rng.normal(size=(100,3)))
    c=ledoit_wolf(r)

    assert np.linalg.eigvalsh(c).min()>0
    one=vol_target(r[0],window=20)
    r.iloc[-1,0]=1000
    pd.testing.assert_frame_equal(one,vol_target(r[0],window=20))


def test_kelly_capped_and_not_probability():
    assert fractional_kelly(.2,.2)==.25
    assert fractional_kelly(-.2,.2)==0

    with pytest.raises(ValueError):
        fractional_kelly(.2,.2,fraction=.5)


def test_core_sweeps_unused_budgets_and_internal_cash():
    w=core_sweep({'rev':{'A':.4,'CASH':.6},'q':{'B':1}}, {'rev':.3,'q':.2},{'CORE':1})

    assert w.to_dict()==pytest.approx({'A':.12,'B':.2,'CORE':.68})
    assert w.sum()==pytest.approx(1)


def test_bands_preserve_account_and_sweep_small_cash():
    current={'A':.2,'CORE':.8}
    w=core_sweep({'rev':{'A':1}},{'rev':.201},{'CORE':1},current=current,band=.01)
    assert w.to_dict()==current

    w=core_sweep({'rev':{'A':1}},{'rev':.2},{'CORE':1},current={'A':.2,'CORE':.799,'CASH':.001},band=.01)
    assert w.get('CASH',0)==0 and w['CORE']==.8


def test_bad_weights_fail():
    with pytest.raises(ValueError):
        core_sweep({'a':{'X':1}},{'a':1.1},{'C':1})

    with pytest.raises(ValueError):
        core_sweep({'a':{'X':1}},{'a':.1},{'CASH':1})
