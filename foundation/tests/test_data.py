import pandas as pd
import pytest
from foundation.data import *
from foundation.data import SOURCES


def fixture_file(tmp_path):
    p=tmp_path/'data.csv'
    pd.DataFrame(dict(id=['A','A','B'],event=['2020-01-01T00:00Z']*3,
                      known=['2020-01-02T00:00Z','2020-02-01T00:00Z','2020-01-02T00:00Z'],value=[1,99,None])).to_csv(p,index=False)
    return p


@pytest.mark.parametrize('source',SOURCES)
def test_every_source_keeps_only_available_revision(source,tmp_path):
    p=fixture_file(tmp_path)
    f=load(source,p,as_of='2020-01-15T00:00Z',mapping=dict(instrument_id='id',event_at='event',available_at='known'),keys=['instrument_id','event_at'])

    assert f.loc[f.instrument_id=='A','value'].item()==1
    assert pd.isna(f.loc[f.instrument_id=='B','value'].item())
    assert f.available_at.notna().all() and f.source.eq(source).all()


def test_clocks_and_database_fail_closed(tmp_path):
    p=fixture_file(tmp_path)
    with pytest.raises(ValueError): 
        load('market_data_lab',p,as_of='2020-01-15T00:00Z',mapping=dict(instrument_id='id',event_at='event'),keys=['id'])
    db=tmp_path/'market_data_lab.sqlite3'; db.write_text('never read')
    with pytest.raises(ValueError,match='prohibited'):
        load('market_data_lab',db,as_of='2020-01-15T00:00Z',mapping={},keys=['id'])


def test_future_membership_and_revision():
    f=pd.DataFrame(dict(event_id=['1','2','3'],instrument_id=['A','A','B'],effective_at=['2020-01-01T00:00Z','2020-02-01T00:00Z','2020-01-01T00:00Z'],available_at=['2019-12-01T00:00Z','2020-01-20T00:00Z','2019-12-01T00:00Z'],action=['add','remove','add'],status=['known','known','unknown']))
    r=universe_at(f,'2020-01-25T00:00Z')

    assert r['members'].instrument_id.tolist()==['A'] and r['unknown_or_excluded']==1
    assert universe_at(f,'2020-02-02T00:00Z')['members'].empty


def test_cache_detects_damage(tmp_path):
    c=Cache(tmp_path/'cache'); f=pd.DataFrame({'v':[1,2]})
    p={'input_sha256':'a','code_sha256':'b','protocol_sha256':'c'}

    path=c.put(f,p); pd.testing.assert_frame_equal(c.get(p),f); assert c.size_bytes()>0
    path.write_bytes(b'bad')
    with pytest.raises(ValueError): c.get(p)


def test_streams_do_not_bridge_missing_equity(tmp_path):
    p = tmp_path/'stream.csv'
    pd.DataFrame({'Date':['2020-01-01','2020-01-02','2020-01-03'],'value':[101,None,103]}).to_csv(p,index=False)
    r = load_streams({'a':StreamSpec(p,'value','equity',100)})
    assert r.iloc[0,0]==pytest.approx(.01) and r.iloc[1:,0].isna().all()
