from dataclasses import asdict, replace
import pandas as pd
import pytest
from foundation.events import Event, EventStore, EventOrchestrator, Decision, default_registry
from foundation.contracts import TargetPositions
from foundation.store import fingerprint


def rns(published, received=None):
    return Event('uk_rns','r1','results',('A',),published,received or published,{'headline':'Results'},published_at=published)

@pytest.mark.parametrize('published,deadline', [('2026-10-08T06:00Z','2026-10-08T07:00Z'),
    ('2026-10-08T07:00Z','2026-10-09T07:00Z'),
    ('2026-10-23T15:30Z','2026-10-26T08:00Z'),
    ('2026-12-24T12:30Z','2026-12-29T08:00Z')])

def test_next_xlon_open_boundaries(published,deadline):
    registry=default_registry();spec=registry.get('uk_results_release');event=rns(published)
    assert spec.deadline(event)==pd.Timestamp(deadline)
    assert registry.fire(event,deadline)

    assert not registry.fire(event,pd.Timestamp(deadline)+pd.Timedelta('1ns'))
    assert not registry.fire(rns(published,str(pd.Timestamp(deadline)+pd.Timedelta('1s'))),pd.Timestamp(deadline)+pd.Timedelta('1s'))


def test_rns_requires_publication_and_versioned_calendar_policy():
    r=default_registry()
    s=r.get('uk_regulatory_announcement')
    assert s.version==2 and s.latency_policy=='next_xlon_open' and s.latency_budget_seconds is None

    with pytest.raises(ValueError,match='publication'):
        s.deadline(rns(None,'2026-10-08T08:00Z'))
    with pytest.raises(ValueError):
        replace(s,latency_budget_seconds=300)
    old=r.get('insider_purchase');legacy=asdict(old);legacy.pop('latency_policy')
    assert old.sha256==fingerprint(legacy)


def setup(tmp_path,timer):
    store=EventStore(tmp_path/'events.db');store.append(rns('2026-10-08T06:00Z'),recorded_at='2026-10-08T06:00Z')
    return EventOrchestrator(store,default_registry(),tmp_path/'out',system='paper',currency='GBP',protocol={'rns':2},timer=timer)


def test_rns_processing_expiry(tmp_path):
    ticks=iter([0.,2.])
    o=setup(tmp_path,lambda:next(ticks))
    assert o.run(lambda c:Decision({'A':1}),at='2026-10-08T06:59:59Z')[0]['status']=='expired_after_decision'
    assert not (tmp_path/'out/current_target.json').exists()


def test_rns_outbox_recovery_expiry(tmp_path,monkeypatch):
    o=setup(tmp_path,lambda:0.)
    publish=TargetPositions.publish
    monkeypatch.setattr(TargetPositions,'publish',lambda *a,**kw:(_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        o.run(lambda c:Decision({'A':1},300),at='2026-10-08T06:59:59Z')
    monkeypatch.setattr(TargetPositions,'publish',publish)
    assert o.run(lambda c:pytest.fail('must not rerun'),at='2026-10-08T07:00:01Z')[0]['status']=='expired_before_publish'
    assert not (tmp_path/'out/current_target.json').exists()
