#adversarial PIT, retry, source-clock, network-boundary and research tests. Primarily for IBKR news and SEC filings.

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import fcntl
import io
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import pytest
from foundation.events import *
from foundation.events.core import stamp
from foundation.events.ibkr_news import NewsWirePolicy
from foundation.contracts import TargetPositions
from foundation.costs import CostModel
from foundation.store import file_hash
from foundation.validate import TrialLog

T='2026-10-08T10:00:00Z'
def event(**kwargs):

    return replace(Event('test','e1','news',('A',),T,T,{'value':1},T),**kwargs)
def registry(budget=120):
    r=TriggerRegistry()
    r.register(TriggerSpec('test', 'test','Test hypothesis','max(published_at, received_at)',budget,('news',)),lambda e,at: e.payload['value']>0)
    return r


def test_event_frozen_nested_payload_and_dedup_hash():
    payload={'nested':[{'value':1}]};e=event(payload=payload);payload['nested'][0]['value']=2

    assert e.payload['nested'][0]['value']==1
    with pytest.raises(TypeError):
        e.payload['value']=4

    assert replace(e,received_at='2026-10-08T10:01:00Z').content_hash==e.content_hash

@pytest.mark.parametrize('field,value',[('received_at','2026-10-08'),('payload',{'x':float('nan')}),('payload',[]),('instrument_ids',('A','A')),('source','')])
def test_invalid_event(field,value):
    with pytest.raises((ValueError,TypeError)): event(**{field:value})


def test_store_revisions_receipt_ingestion_and_precision(tmp_path):
    s = EventStore(tmp_path/'events.db');first=event();s.append(first, recorded_at=T)
    duplicate, inserted = s.append(replace(first, received_at='2026-10-08T10:00:02Z'), recorded_at='2026-10-08T10:00:02Z')

    assert not inserted and duplicate.received_at==first.received_at
    correction=event(payload={'value':2}, received_at='2026-10-08T10:00:00.500000Z')

    s.append(correction, recorded_at='2026-10-08T10:00:03Z')
    assert s.as_of(T)[0].payload['value'] == 1
    assert s.as_of('2026-10-08T10:00:00.5Z')[0].payload['value'] == 2
    assert s.as_of('2026-10-08T10:00:02Z', known_at='2026-10-08T10:00:02Z')[0].payload['value'] == 1
    assert len(s.timeline('2026-10-08T10:00:05Z')) == 2

    with pytest.raises(ValueError, match='strictly later'): 
        s.append(event(payload={'value':3}), recorded_at=T)
    with pytest.raises(ValueError, match='future'):
        s.append(event(event_id='future'), recorded_at='2026-10-08T09:59:00Z')
    with pytest.raises(sqlite3.IntegrityError, match='Immutable'):
        with s.transaction() as db: db.execute('DELETE FROM event_versions')


def test_registry_availability_latency_and_specs():
    r = registry();e=event(received_at='2026-10-08T10:00:10Z')
    assert not r.fire(e,T)
    assert len(r.fire(e,'2026-10-08T10:02:00Z'))==1
    assert not r.fire(e,'2026-10-08T10:02:00.000001Z')
    assert r.describe()[0]['signal_kind']=='activated'
    with pytest.raises(ValueError):
        r.register(r.get('test'),lambda e,t:True)
    with pytest.raises(ValueError):
        replace(r.get('test'), latency_budget_seconds=float('nan'))


def orchestrator(tmp_path, budget=120, **kwargs):
    s=EventStore(tmp_path/'events.db');s.append(event(), recorded_at=T)
    return EventOrchestrator(s,registry(budget),tmp_path/'out', system='paper-account', currency='GBP', protocol={'v':1}, **kwargs)


def test_decision_batches_targets_and_restart_dedup(tmp_path):
    o = orchestrator(tmp_path);o.store.append(event(event_id='e2'), recorded_at=T);calls=[]
    def handler(ctx):
        calls.append(ctx); 
        return Decision({'A':2,'B':4})
    assert o.run(handler,at=T)[0]['status'] == 'published'
    target=TargetPositions.load(tmp_path/'out/current_target.json')
    assert target.quantities=={'A':2,'B':4} and len(calls[0].firings) == 2
    assert o.run(handler,at=T)==[] and len(calls) == 1
    with pytest.raises(ValueError,match='bound'):
        EventOrchestrator(o.store,registry(),tmp_path/'another',system='paper-account',currency='GBP',protocol={'v':1})


def test_ready_target_recovery_does_not_repeat_handler(tmp_path,monkeypatch):
    o = orchestrator(tmp_path);publish=TargetPositions.publish;calls=[]
    monkeypatch.setattr(TargetPositions,'publish',lambda *a,**k: (_ for _ in ()).throw(OSError('disk error')))

    with pytest.raises(OSError):
        o.run(lambda ctx:(calls.append(ctx) or Decision({'A':1})),at=T)
    monkeypatch.setattr(TargetPositions,'publish',publish)
    assert o.run(lambda ctx:pytest.fail('Repeated prepared decision'),at=T)[0]['status'] == 'published'
    assert len(calls) == 1


def test_processing_latency_budget_and_paper_gate(tmp_path):
    ticks=iter([0.,121.]);o=orchestrator(tmp_path, timer=lambda:next(ticks))
    assert o.run(lambda ctx:Decision({'A':1}),at=T)[0]['status'] == 'expired_after_decision'
    assert not (tmp_path/'out/current_target.json').exists()
    with pytest.raises(ValueError,match='paper'):
        orchestrator(tmp_path/'live',paper=False)


def test_no_action_and_failed_decisions_are_durable(tmp_path):
    o = orchestrator(tmp_path)
    assert o.run(lambda c:None,at=T)[0]['status'] == 'no_action'
    assert o.run(lambda c:pytest.fail('Repeated no-action'),at=T) == []
    o = orchestrator(tmp_path/'bad')
    with pytest.raises(ValueError):
        o.run(lambda c:Decision({'A':float('nan')}),at=T)
    with o.store.transaction() as db:
        assert db.execute('SELECT status FROM decision_runs').fetchone()[0] == 'failed'
    assert o.run(lambda c:pytest.fail('Failed handler must need explicit review'),at=T) == []


def test_poll_finishes_after_real_receipts(tmp_path):
    o = orchestrator(tmp_path);later=event(event_id='later',received_at='2026-10-08T10:00:05Z')
    class Watcher:
        def poll(self,at):
            return [later]
    assert o.poll(Watcher(),lambda c:Decision({'A':1}),at=T)[0]['status'] == 'published'


ATOM=b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>accession</id><title>4 - Owner</title><updated>2026-10-08T09:59:50Z</updated><category term="4"/><link href="https://www.sec.gov/Archives/example"/></entry></feed>'''


def test_edgar_form4_metadata_is_not_an_insider_buy():
    rows = parse_edgar_atom(ATOM,received_at=T)
    assert rows[0].instrument_ids == () and not default_registry().fire(rows[0],T)
    enriched = replace(rows[0],instrument_ids=('US-ISSUER',),payload={**dict(rows[0].payload),'reported_purchase':True})
    assert default_registry().fire(enriched,T)[0].spec.name == 'insider_purchase'
    fixture = EDGARWatcher(None,form='4',fixture=ATOM,issuer_ids={'accession':('A',)})
    assert fixture.poll(T)[0].instrument_ids==('A',)


def test_edgar_form4_transactions_and_unsafe_xml():
    raw = b'<ownershipDocument><issuer><issuerCik>42</issuerCik></issuer><nonDerivativeTransaction><transactionCode>P</transactionCode><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></nonDerivativeTransaction></ownershipDocument>'
    assert parse_form4_document(raw)['reported_purchase']
    assert not parse_form4_document(raw.replace(b'<transactionCode>P',b'<transactionCode>S'))['reported_purchase']
    with pytest.raises(ValueError, match='entity'):
        parse_form4_document(b'<!DOCTYPE x []>'+raw)


def test_sec_submissions_clocks_and_alignment():
    value={'cik':42,'filings':{'recent':{'accessionNumber':['a'],'form':['8-K'],'acceptanceDateTime':[T],'primaryDocument':['form.htm']}}}
    e = parse_edgar_submissions(value,received_at=T,issuer_ids={'0000000042':('A',)})[0]
    assert e.kind=='8k' and e.instrument_ids==('A',)
    value['filings']['recent']['form']=[]
    with pytest.raises(ValueError,match='Unaligned'):
        parse_edgar_submissions(value,received_at=T,issuer_ids={})


def test_sec_default_off_rate_header_and_cooldown(tmp_path):
    clock=[1000.]; sent=[]
    def sleep(seconds):
        clock[0]+=seconds
    def opener(request,timeout):
        sent.append((clock[0],dict(request.header_items())))
        return io.BytesIO(ATOM)
    opts = dict(user_agent='Research contact@example.org',state_path=tmp_path/'rate.json',clock=lambda:clock[0],sleeper=sleep,opener=opener)
    off=SECClient(**opts)
    with pytest.raises(PermissionError):
        off.get('https://www.sec.gov/test')
    on=SECClient(**opts,network_enabled=True,shared_budget_confirmed=True)
    assert on.get('https://www.sec.gov/test')==ATOM;on.get('https://data.sec.gov/test')
    assert sent[1][0]-sent[0][0]>=1 and 'contact@example.org' in sent[0][1]['User-agent']
    with pytest.raises(ValueError):
        on.get('https://www.sec.gov.evil.test')
    def throttled(*a,**k):
        raise HTTPError('https://www.sec.gov/test',429,'rate',{'Retry-After':'1000'},None)
    on.opener=throttled
    with pytest.raises(HTTPError):
        on.get('https://www.sec.gov/test')
    with pytest.raises(PermissionError,match='cooldown'):
        on.get('https://www.sec.gov/test')
    with pytest.raises(ValueError):
        SECClient(user_agent='anonymous')


def test_nsm_rns_export_preserves_late_receipt_and_hash(tmp_path):
    path=tmp_path/'licensed.csv';path.write_text('id,published_at,received_at,headline,instrument_id,kind\na,2026-10-08T09:00:00Z,2026-10-08T10:00:00Z,Results,A,results\n')
    reader=NSMRNSReader(path,source='uk_rns',mapping={'id':'event_id'},expected_sha256=file_hash(path))
    assert reader.poll('2026-10-08T09:30:00Z')==[]
    e=reader.poll(T)[0];assert e.kind=='results' and e.available_at==pd.Timestamp(T)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='hash'):
        reader.poll(T)


def test_price_monitor_is_past_only_and_rejects_unknown_gap_reference():
    times=pd.date_range(T,periods=6,freq='min')
    f=pd.DataFrame({'instrument_id':'A','event_at':times,'available_at':times,'price':[100,101,99,100,102,120]})
    base=PriceMonitor(f,window=3).poll(times[4]);changed=f.copy();changed.loc[5,'price']=10000
    assert [e.to_dict() for e in base]==[e.to_dict() for e in PriceMonitor(changed,window=3).poll(times[4])]
    kinds={e.kind for e in PriceMonitor(f,window=3).poll(times[-1])};assert {'volatility_shock','range_breakout'}<=kinds
    gap=f.iloc[-1:].assign(reference_price=100,reference_available_at=times[-1]+pd.Timedelta('1s'))
    with pytest.raises(ValueError,match='known'):
        PriceMonitor(gap).poll(times[-1])
    gap['reference_available_at']=times[0]
    assert PriceMonitor(gap).poll(times[-1])[0].kind=='price_gap'


def test_macro_revision_cancels_old_scheduled_trigger(tmp_path):
    old={'release_id':'cpi','scheduled_at':'2026-10-08T12:00:00Z','known_at':'2026-10-01T00:00:00Z',
         'received_at':'2026-10-01T00:00:00Z','name':'CPI','country':'GB','instrument_ids':['A']}
    new={**old,'scheduled_at':'2026-10-08T13:00:00Z','received_at':T,'known_at':T}
    store=EventStore(tmp_path/'events.db')
    for e in MacroCalendar([old,new]).poll(T):
        store.append(e,recorded_at=T)
    r=default_registry()
    assert firing_frame(store,r,as_of='2026-10-08T12:00:00Z',names=['macro_release']).empty
    assert len(firing_frame(store,r,as_of='2026-10-08T13:00:00Z',names=['macro_release']))==1


def test_crypto_observations_period_and_past_volatility():
    row={'event_at':T,'available_at':T,'instrument_id':'BTC-GBP','venue':'existing-export','funding_rate':.003,
         'funding_period_hours':8,'realised_vol':.8,'reference_vol':.2,'reference_available_at':T}
    assert {e.kind for e in CryptoMetricsMonitor([row]).poll(T)}=={'funding_extreme','crypto_volatility'}
    with pytest.raises(ValueError):
        CryptoMetricsMonitor([{**row,'funding_period_hours':0}]).poll(T)
    assert len(catalogue())==12 and all(row['paper_only'] for row in catalogue())


def guard(tmp_path,clock):
    root = tmp_path/'uk';(root/'research').mkdir(parents=True)
    uk = root/'config.json';uk.write_text(json.dumps({'runtime_root':str(root),'protocol_file':'protocol.json','readonly':True,'client_id':27101}))
    (root/'research/protocol.json').write_text(json.dumps({'schedule':{'slots':['08:05','16:20']}}))
    options = tmp_path/'options.toml';options.write_text(f'provider="ibkr"\nuk_config="{uk}"\n[slots.morning]\nnew_york_time="10:30"\nwindow_minutes=25\n[slots.close]\nnew_york_time="15:45"\nminutes_before_close=15\n')
    locks = [tmp_path/'uk.lock',tmp_path/'options.lock']
    for p in locks:
        p.touch()
    return CaptureGuard(uk_config=uk,options_config=options,uk_lock=locks[0],options_lock=locks[1],clock=clock)

@pytest.mark.parametrize('at,blocked',[('2026-10-08T22:00:00Z',False),('2026-10-08T07:00:00Z',True),
 ('2026-10-08T19:45:00Z',True),('2026-10-28T20:15:00Z',True),('2026-11-04T21:00:00Z',True),
 ('2026-10-10T12:00:00Z',False),('2026-10-08T06:39:45Z',True)])
def test_capture_blackouts_dst_and_prestart_buffer(tmp_path,at,blocked):
    g = guard(tmp_path,lambda:pd.Timestamp(at))
    if blocked:
        with pytest.raises(PermissionError,match='blackout'):
            g.check()
    else:
        g.check()


def test_capture_guard_drift_busy_and_missing_lock(tmp_path):
    g=guard(tmp_path,lambda:pd.Timestamp('2026-10-08T22:00:00Z'))
    with g.locks[0].open('rb') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(PermissionError,match='busy'):
            g.check()
        fcntl.flock(handle,fcntl.LOCK_UN)
    g.uk_activation.parent.mkdir(parents=True)
    g.uk_activation.write_text('{}')
    with pytest.raises(PermissionError,match='Activated UK'):
        g.check()
    g.uk_activation.unlink()
    g.locks[0].unlink()
    with pytest.raises(PermissionError,match='absent'):
        g.check()
    g.uk_config.write_text(g.uk_config.read_text()+' ')
    with pytest.raises(PermissionError,match='changed'):
        g.check()


def test_ibkr_wire_policy_rejects_order_account_and_unscoped_requests():
    class Guard:
        def check(self):
            pass
    p = NewsWirePolicy(Guard());p.check(71)
    for msg in [1,2,3,4,5,8,85,86]:
        with pytest.raises(PermissionError):p.check(msg)
    with p.request(1):
        p.check(1)
        with pytest.raises(PermissionError):
            p.check(3)
    with pytest.raises(PermissionError):
        p.check(1)


def test_ibkr_fixture_clock_and_network_default_off():
    row={'provider':'BRF','article_id':'one','timestamp_ms':int(pd.Timestamp(T).timestamp()*1000),
         'received_at':'2026-10-08T10:00:01Z','headline':'Headline'}
    watcher = IBKRNewsWatcher(provider='BRF',fixture=[row])
    assert watcher.poll(T)==[] and len(watcher.poll('2026-10-08T10:00:02Z'))==1
    with pytest.raises(PermissionError):
        IBKRNewsWatcher(provider='BRF').poll(T)
    with pytest.raises(ValueError):
        IBKRNewsWatcher(provider='BRF',client_id=27101)


def test_replay_enters_after_receipt_and_includes_idle_capital(tmp_path):
    s=EventStore(tmp_path/'events.db')
    s.append(event(),recorded_at=T)
    dates=pd.date_range(T,periods=5,freq='D')
    q=pd.DataFrame([{'instrument_id':i,'at':at,'price':100+n,'tradable':True} for i in ['A','B'] for n,at in enumerate(dates)])
    instruments={i:{'venue':'simulation','currency':'GBP','sdrt':'exempt'} for i in ['A','B']}
    run=replay(s,registry(),q,names=['test'],holding=1,cost_model=CostModel(10),instruments=instruments,
               volatility={'A':.2,'B':.4},allocation_known_at=T)
    assert run['trades'].iloc[0]['at']==dates[1]
    assert run['ledger'].iloc[0].cash==pytest.approx(10000)
    assert run['weights']['A']==pytest.approx(2/3)
    assert run['ledger'].iloc[1].cash>=10000/3-1e-8
    assert run['trades'].total.sum()==pytest.approx(run['ledger'].cost.sum())
    with pytest.raises(ValueError,match='after replay'):
        replay(s,registry(),q,names=['test'],holding=1,cost_model=CostModel(10),instruments=instruments,
               volatility={'A':.2,'B':.4},allocation_known_at=dates[1])


def test_full_event_study_validation_costs_and_global_log_interface(tmp_path):
    dates=pd.date_range('2025-01-01T10:00:00Z',periods=100,freq='D')
    rng=np.random.default_rng(19)
    prices=100*np.cumprod(1+rng.normal(.001,.01,len(dates)))
    q = pd.DataFrame({'at':dates,'price':prices,'instrument_id':'A','tradable':True})
    s = EventStore(tmp_path/'events.db')
    r=default_registry()
    for i in range(5,90,8):
        e=Event('price_monitor',str(i),'price_gap' if i%3 else 'range_breakout',('A',),stamp(dates[i]),stamp(dates[i]),{},stamp(dates[i]))
        s.append(e,recorded_at=dates[i])
    variants={'gap':['opening_gap'],'breakout':['range_breakout']}
    p={'project':'synthetic_event_test','variants':variants,'holding':3,'initial_capital':10000,
       'periods_per_year':365,'block_size':5,'resamples':100,'seed':7,'control_window':5,
       'min_train':30,'test_size':15,'purge':'1D','embargo':'1D','publication_date':str(dates[50])}
    
    from foundation.signals import backtest_targets
    price_frame=pd.DataFrame({'A':prices},index=dates)
    instruments={'A':{'venue':'simulation','currency':'GBP','sdrt':'exempt'}}
    def benchmarks(bp):
        w=pd.DataFrame({'A':1.,'CASH':0.}, index=dates[:1]-pd.Timedelta('1s'))
        returns=backtest_targets(w,price_frame, CostModel(bp), instruments=instruments)['ledger'].net_return
        return returns,returns
    log=TrialLog(tmp_path/'synthetic_trials.jsonl')
    result=study(s,r,q,variants=variants,primary='gap',protocol=p,benchmarks=benchmarks,instruments=instruments,
                 volatility={'A':.2},allocation_known_at=dates[0],provenance={'synthetic_seed':19},trial_log=log,
                 adv_at_trade=lambda trades:np.repeat(1e6,len(trades)))
    report=result['report'];assert not report['automatic_promotion'] and len(log.read())==2
    assert len(report['validation']['cost_scenarios']) == 4
    assert report['validation']['walk_forward'] and report['deflated_sharpe']
    assert set(report['paired_inference'])=={'natural_benchmark','buy_and_hold'}
    assert report['publication_split'] and len(report['validation']['holm'])==2


def test_official_ibapi_encodes_news_and_refuses_orders_on_both_wires():
    pytest.importorskip('ibapi')  
    from foundation.events.ibkr_news import official_client
    from ibapi.common import PROTOBUF_MSG_ID
    from ibapi.contract import Contract
    from ibapi.server_versions import MAX_CLIENT_VER
    from ibapi.order import Order

    class Guard:
        def check(self):
            pass
    sent=[]
    received=[]
    class Wire:
        def sendMsg(self,msg):
            sent.append(msg)
    client=official_client(lambda *args:received.append(args),Guard())
    client.isConnected = lambda:True;client.conn=Wire()
    contract=Contract();contract.symbol='BRF:BRF_ALL';contract.secType='NEWS';contract.exchange='BRF'
    for version in [157,MAX_CLIENT_VER]:
        client.serverVersion_=version
        with client.policy.request(1):client.reqMktData(9001,contract,'mdoff,292',False,False,[])
        with client.policy.request(2):client.cancelMktData(9001)
    assert len(sent) == 4
    for method,msg in [(client.sendMsg,3),(client.sendMsgProtoBuf,3+PROTOBUF_MSG_ID)]:
        with pytest.raises(PermissionError):
            method(msg,b'forbidden')
    with pytest.raises(PermissionError):
        client.placeOrder(1,contract,Order())
    client.nextValidId(1);assert client.ready.is_set()
    client.tickNews(9001,1000,'BRF','id','headline','')
    assert len(received)==1 and len(sent)==4


def test_ibkr_rejects_seconds_mistaken_for_milliseconds():
    watcher=IBKRNewsWatcher(provider='BRF',fixture=[{'provider':'BRF','article_id':'one',
        'timestamp_ms':int(pd.Timestamp(T).timestamp()),'received_at':T,'headline':'headline'}])
    with pytest.raises(ValueError,match='milliseconds'):
        watcher.poll(T)


def test_network_news_lifecycle_with_fake_broker_and_no_capture_changes(tmp_path):
    pytest.importorskip('ibapi')
    import threading
    class Guard:
        reserved={27101}
        def check(self,**kwargs):
            pass
    g=Guard();trace=[]
    class Client:
        def __init__(self,callback,guard):
            self.ready=threading.Event()
            self.policy=NewsWirePolicy(guard)
            self.failures=[]
            self.callback=callback
            self.connected=False

        def connect(self,host,port,client_id):
            trace.append(('connect',host,port,client_id));self.connected=True;self.ready.set()

        def run(self):
            pass
        def reqMktData(self,req,contract,ticks,snapshot,paid,opts):
            self.policy.check(1);assert contract.secType=='NEWS' and not snapshot and not paid
            trace.append(('news',ticks))
            self.callback(req,int(pd.Timestamp.now(tz='UTC').timestamp()*1000),'BRF','id','headline','')

        def cancelMktData(self,req):
            self.policy.check(2);trace.append(('cancel_news',req))
        def isConnected(self):
            return self.connected
        def disconnect(self):
            self.connected=False;trace.append(('disconnect',))
    
    watcher=IBKRNewsWatcher(provider='BRF',guard=g,seconds=.01,client_factory=Client,
                            network_enabled=True,readonly_confirmed=True,entitlement_confirmed=True,
                            session_lock=tmp_path/'news.lock')
    rows=watcher.poll(T)
    assert len(rows)==1 and trace[0][-1]==27401 and trace[-1]==('disconnect',)
    assert [t[0] for t in trace]==['connect','news','cancel_news','disconnect']


def test_no_event_trial_is_counted_and_inference_refused(tmp_path):
    dates=pd.date_range('2025-01-01T10:00:00Z',periods=40,freq='D')
    q=pd.DataFrame({'instrument_id':'A','at':dates,'price':100.,'tradable':True})
    s=EventStore(tmp_path/'events.db');log=TrialLog(tmp_path/'trials.jsonl')
    p={'project':'no_firing_example','variants':{'gap':['opening_gap']},'holding':3, 'initial_capital':10000,'periods_per_year':365}

    with pytest.raises(ValueError,match='trial recorded as rejected'):

        study(s,default_registry(),q,variants=p['variants'],primary='gap',protocol=p,
              benchmarks=lambda bp:pytest.fail('No unsupported inference'),
              instruments={'A':{'venue':'simulation','currency':'GBP','sdrt':'exempt'}},
              volatility={'A':.2},allocation_known_at=dates[0],provenance={'synthetic':True},trial_log=log)
        
    assert len(log.read())==1 and log.read()[0]['record']['status']=='rejected'
