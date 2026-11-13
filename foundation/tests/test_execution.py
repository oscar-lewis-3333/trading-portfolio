from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import pandas as pd
import pytest
from foundation.execution import *
from foundation.execution import transition
from foundation.contracts import TargetPositions
from foundation.risk_monitor import Limits
from foundation.risk_monitor import DrawdownStop, drift
from foundation.costs import CostModel

LIMITS=Limits(5000,10000,10000,10000,20000)
T0='2020-01-01T10:00Z'


def setup(tmp_path,adapter=Trading212Paper):
    broker=adapter(tmp_path/'paper.sqlite3',system='isa')
    risk=DrawdownStop(tmp_path/'risk.json');risk.update(10000,T0)
    return broker,risk,PaperService(broker,LIMITS,risk_monitor=risk)


def target(**kw):
    return TargetPositions(**(dict(system='isa',target_id='t1',created_at=T0,effective_at=T0,expires_at='2020-01-02T00:00Z',currency='GBP',quantities={'A':10.})|kw))


@pytest.mark.parametrize('adapter',[Trading212Paper,RevolutXPaper,IBKRPaper,AlpacaPaper])
def test_adapter_persistent_partial_fill_and_restart(tmp_path,adapter):
    broker, risk, service = setup(tmp_path,adapter)
    t=target()
    order_id = service.reconcile(t,{'A':100},at=T0)[0]

    broker.fill(order_id,fill_id='fill1',quantity=4,price=100,fee=1,at=T0,limits=LIMITS,prices={'A':100})
    restarted=adapter(tmp_path/'paper.sqlite3',system='isa'); service2=PaperService(restarted,LIMITS,risk_monitor=risk)

    assert service2.reconcile(t,{'A':101},at=T0)==[order_id]
    restarted.fill(order_id,fill_id='fill1',quantity=4,price=100,fee=1,at=T0,limits=LIMITS,prices={'A':100})
    assert restarted.positions()=={'A':4} and restarted.cash()==9599
    restarted.fill(order_id,fill_id='fill2',quantity=6,price=100,fee=1,at=T0,limits=LIMITS,prices={'A':100})
    assert restarted.positions()=={'A':10} and restarted.cash()==8998
    assert restarted.orders()[0]['state']=='FILLED' and len(restarted.fills()) == 2


def test_collision_expiry_system_and_overfill(tmp_path):
    b,r,s=setup(tmp_path); t=target(); key=s.reconcile(t,{'A':100},at=T0)[0]
    with pytest.raises(ValueError):
        s.reconcile(target(quantities={'A':11}),{'A':100},at=T0)
    with pytest.raises(ValueError):
        s.reconcile(target(system='crypto'),{'A':100},at=T0)
    with pytest.raises(ValueError):
        s.reconcile(t,{'A':100},at='2020-01-03T00:00Z')
    with pytest.raises(ValueError):
        b.fill(key,fill_id='bad',quantity=11,price=100,fee=0,at=T0,limits=LIMITS,prices={'A':100})
    assert not b.positions() and b.cash()==10000


def test_new_target_cancels_only_remaining_and_sells_actual(tmp_path):
    b,r,s = setup(tmp_path); key=s.reconcile(target(),{'A':100},at=T0)[0]
    b.fill(key,fill_id='f1',quantity=4,price=100,fee=0,at=T0,limits=LIMITS,prices={'A':100})
    t = target(target_id='t2',effective_at='2020-01-01T11:00Z',quantities={})
    ids = s.reconcile(t,{'A':100},at='2020-01-01T11:00Z')
    assert len(ids) == 1
    orders = b.orders()
    assert orders[0]['state']=='CANCELLED' and orders[1]['payload']['quantity']==-4


def test_aggregate_pretrade_limits_and_transaction_rollback(tmp_path):
    b,r,s = setup(tmp_path)
    strict = PaperService(b,Limits(5000,5000,6000,6000,10000),risk_monitor=r)
    with pytest.raises(PermissionError):
        strict.reconcile(target(quantities={'A':40,'B':40}),{'A':100,'B':100},at=T0)
    assert not b.orders()


def test_cash_including_fees_at_fill(tmp_path):
    b,r,s = setup(tmp_path); key=s.reconcile(target(),{'A':100},at=T0)[0]
    with pytest.raises(PermissionError):
        b.fill(key,fill_id='f',quantity=10,price=100,fee=10000,at=T0,limits=LIMITS,prices={'A':100})
    assert not b.fills() and not b.positions()


def test_drawdown_latches_across_restart(tmp_path):
    b,r,s = setup(tmp_path); r.update(8000,'2020-01-01T11:00Z')
    again = DrawdownStop(tmp_path/'risk.json'); assert again.update(11000,'2020-01-01T12:00Z')['stopped']
    with pytest.raises(PermissionError):
        s.reconcile(target(),{'A':100},at='2020-01-01T12:00Z')


def test_no_live_paths_and_state_machine():
    assert not live_gate(LIVE=True)['enabled']
    with pytest.raises(ValueError):
        transition('FILLED','SUBMITTED')


def test_cross_process_style_rate_reservations(tmp_path):
    p=tmp_path/'rate.json'
    with ThreadPoolExecutor(max_workers=4) as pool:
        slots=list(pool.map(lambda _:SharedRateLimiter(p).reserve(now=100),range(10)))
    assert sorted(slots)==list(range(100,110))
    with pytest.raises(PermissionError):
        SharedRateLimiter(p).reserve(now=100,real_requests=True)
    with pytest.raises(ValueError):
        SharedRateLimiter(p).reserve(now=99)


def test_file_target_run_once_idempotence(tmp_path):
    b,r,s=setup(tmp_path); p=tmp_path/'target.json'; target().publish(p)
    kwargs=dict(at=T0,cost_model=CostModel(10),instruments={'A':{'sdrt':'exempt','venue':'trading212','currency':'GBP'}})
    with pytest.raises(ValueError,match='venue/currency'):
        s.run_once(p,{'A':100},**(kwargs|{'instruments':{'A':{'venue':'ibkr','currency':'GBP','sdrt':'exempt'}}}))

    assert not b.orders()
    first=s.run_once(p,{'A':100},**kwargs); second=s.run_once(p,{'A':100},**kwargs)

    assert first==second and first['positions']=={'A':10} and first['cash']==8999


def test_drift_alerts():
    r=drift(slippage_bps=[20]*30,backtest_slippage_bps=2,hits=[0]*30,expected_hit_rate=.5,
            features={'a':list(range(100,130))},reference_features={'a':list(range(30))})
    assert set(r['alerts'])=={'slippage','hit_rate','feature:a'}


def test_partial_ibkr_commission_is_per_order(tmp_path):
    b,r,s=setup(tmp_path,IBKRPaper)
    p=tmp_path/'target.json';target().publish(p)
    kwargs=dict(at=T0,cost_model=CostModel(0),instruments={'A':{'sdrt':'exempt'}},fill_fraction=.5)

    s.run_once(p,{'A':100},**kwargs)
    kwargs['fill_fraction']=1.
    s.run_once(p,{'A':100},**kwargs)
    assert sum(f['commission'] for f in b.fills())==1 and b.cash()==8999


def test_target_quantities_cannot_mutate():
    t=target()
    with pytest.raises(TypeError):
        t.quantities['A']=20
