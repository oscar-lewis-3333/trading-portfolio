import itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pytest
from foundation import validate as v
from foundation.store import EvidenceLog, atomic_json


def test_purged_embargoed_grouped_splits():
    times = pd.date_range('2020-01-01',periods=40,tz='UTC').repeat(2)
    folds=list(v.walk_forward(times,times+pd.Timedelta('2D'),min_train=6,test_size=5,purge='1D',embargo='2D',train_window=8))
    assert folds
    for f in folds:
        assert (times[f.train]+pd.Timedelta('2D') < f.cutoff).all()
        assert times[f.test].min()-f.cutoff==pd.Timedelta('3D')
        assert 6<=len(times[f.train].unique())<=8
        assert len(f.test)%2==0


def test_paired_bootstrap_identical_has_zero_bounds():
    a = np.random.default_rng(1).normal(0,.01,100)
    result = v.paired_bootstrap(a,a,block_size=9,resamples=101)
    assert result['mean_lower']==result['mean_upper']==result['sharpe_lower']==0
    idx = v.circular_indices(5,3,20)
    assert ((idx[:,1]-idx[:,0])%5==1).all()


def test_alignment_and_missingness_fail_closed():
    with pytest.raises(ValueError):
        v.paired_bootstrap([1,2,np.nan],[1,2,3])
    with pytest.raises(ValueError):
        v.paired_bootstrap(pd.Series([1,2],index=[1,2]),pd.Series([1,2],index=[2,3]))


def test_holm_known_example():
    adjusted, rejected=v.holm([.01,.04,.03])
    np.testing.assert_allclose(adjusted,[.03,.06,.06]); assert rejected.tolist()==[True,False,False]


def test_global_trials_count_failures_and_chain(tmp_path):
    log=v.TrialLog(tmp_path/'trials.jsonl')
    for i,sr in enumerate([1.,2.,None]):
        log.record(project='test',trial_id=str(i),protocol={'v':1},sharpe_value=sr,status='failed' if sr is None else 'evaluated')
    dsr=v.deflated_sharpe(np.random.default_rng(1).normal(.001,.01,100),log)

    assert dsr['trials']==3 and dsr['trials_with_metrics']==2
    assert dsr['deflated_sharpe_estimate'] < dsr['observed_sharpe']
    assert 0 <= dsr['probability'] <= 1

    log.path.write_text(log.path.read_text().replace('"v":1','"v":2'))
    with pytest.raises(ValueError):
        log.read()


def test_concurrent_appends_and_immutable_manifest(tmp_path):
    log=EvidenceLog(tmp_path/'e.jsonl')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i:log.append({'i':i}),range(20)))

    assert len(log.read())==20
    path=tmp_path/'manifest.json'; atomic_json(path,{'v':1}); atomic_json(path,{'v':1})
    with pytest.raises(FileExistsError): 
        atomic_json(path,{'v':2})


def test_pbo_and_candidate_permutation():
    a = np.random.default_rng(10).normal(0,1,(100,4))
    one = v.pbo(a,blocks=4); two=v.pbo(a[:,::-1],blocks=4)
    assert one['splits']==6 and one['pbo']==two['pbo']


def test_screen_every_insertion_order():
    rng=np.random.default_rng(3); a=rng.normal(size=100)
    streams={'a':a,'b':a+.01*rng.normal(size=100),'c':rng.normal(size=100)}
    for order in itertools.permutations(streams):
        assert v.fixed_cluster_screen({k:streams[k] for k in order},{'a':1,'b':0,'c':0})['selected']==['b','c']


def test_controls_causal_and_exposure_distribution():
    ix = pd.date_range('2020-01-01',periods=100,tz='UTC')
    base = pd.Series(np.random.default_rng(7).normal(0,.01,100),index=ix)
    e = pd.Series(np.arange(100)%2,index=ix)

    controls = v.exposure_controls(base,e,repetitions=10)
    np.testing.assert_array_equal((controls['random_skip']!=0).sum(),50)
    other=base.copy()
    other.iloc[-1]=2
    pd.testing.assert_series_equal(controls['vol_matched'].iloc[:-1],v.exposure_controls(other,e,repetitions=10)['vol_matched'].iloc[:-1])


def test_year_publication_capacity():
    s=pd.Series([.01,-.005,.02,.03],index=pd.date_range('2020-12-30',periods=4,tz='UTC'))
    assert len(v.year_spread(s))==2
    assert len(v.publication_split(s,'2021-01-01T00:00Z')['post'])==2
    c=v.turnover_capacity([100,-200],nav=10000,adv=[10000,10000],observations=252)
    assert c['capital_capacity']==5000 and not c['within_limit']
