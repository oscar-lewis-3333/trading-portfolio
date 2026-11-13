import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from foundation.store import EvidenceLog
from foundation.validate import screen_candidates,TrialLog
from foundation.data.news import html_to_text,to_version
from foundation.data.prices import spread_bps
from foundation.models.structured import validate
from foundation.models.news_reading import grounding
from foundation.reporting import serialise


def test_real_subprocess_log_writers(tmp_path):
    p=tmp_path/'log.jsonl'
    script='from foundation.store import EvidenceLog; import sys; EvidenceLog(sys.argv[1]).append({"worker":sys.argv[2]})'
    children=[subprocess.Popen([sys.executable,'-c',script,str(p),str(i)]) for i in range(4)]
    
    assert all(c.wait(timeout=20)==0 for c in children)
    assert len(EvidenceLog(p).read())==4


def test_full_screen_invariant_and_logs_all(tmp_path):
    rng=np.random.default_rng(4);a=pd.Series(rng.normal(.0005,.01,100))
    streams={'a':a,'b':a+.000001}; priorities={'a':0,'b':1}; costs={n:{'gross_active_return':.2,'cost_fraction_total':.01} for n in streams}
    protocol={'screen_priorities':priorities}
    log=TrialLog(tmp_path/'log.jsonl')

    one=screen_candidates(streams,priorities,costs,project='test',protocol=protocol,trial_log=log,resamples=100)
    two=screen_candidates(dict(reversed(list(streams.items()))),priorities,costs,project='test',protocol=protocol,trial_log=log,resamples=100)
    assert one==two and len(log.read())==4
    assert not one['results']['b']['accepted']


def test_offline_market_tools_ports():
    assert html_to_text('<p>A &amp; B</p>')=='A & B'
    item = SimpleNamespace(id=1,headline='News',summary='a',content='b',symbols=['A'],source='fixture',url='https://example.org',created_at='2020-01-01T00:00Z',updated_at=None)
    r = to_version(item,'2020-01-02T00:00Z');assert r['received_at'].startswith('2020-01-02')
    assert spread_bps(99,101)==pytest.approx(200)

    assert not validate({'v':2},{'type':'object','required':['v'],'properties':{'v':{'type':'integer','minimum':1}}})
    assert grounding({'facts':[{'quote':'A & B'}]},'A & B')['passed']
    assert serialise({'a':False,'b':True})=={'a':False,'b':True}
    assert type(serialise(False)) is bool


def test_structured_validation_rejects_extra_nonfinite_and_bool_numbers():
    schema={'type':'object','additionalProperties':False,'required':['x'],'properties':{'x':{'type':'number'}}}
    assert validate({'x':float('nan')},schema)

    assert validate({'x':True},schema)
    assert validate({'x':1,'hidden':'action'},schema)


def test_separate_paper_process_exits_logs_and_restarts_without_duplicate(tmp_path):
    from datetime import datetime,timedelta,timezone
    from foundation.contracts import TargetPositions
    from foundation.execution import Trading212Paper
    now=datetime.now(timezone.utc)
    TargetPositions('isa-paper','smoke-1',now.isoformat(),now.isoformat(),
                    (now+timedelta(hours=1)).isoformat(),'GBP',{'EXAMPLE':5.}).publish(tmp_path/'target.json')
    (tmp_path/'prices.json').write_text(json.dumps({'available_at':now.isoformat(),'prices':{'EXAMPLE':100.}}))

    config={'LIVE':False,'venue':'trading212','system':'isa-paper','currency':'GBP',
            'initial_cash':10000,'state_dir':str(tmp_path/'state'),
            'targets_file':str(tmp_path/'target.json'),'prices_file':str(tmp_path/'prices.json'),
            'flat_bps':10,'poll_seconds':5,'max_price_age_seconds':60,'maximum_drawdown':.15,
            'limits':{'max_order_notional':1000,'max_position_notional':3000,'max_gross_notional':10000,
                      'max_net_notional':10000,'max_daily_turnover':5000,'allow_short':False},
            'instruments':{'EXAMPLE':{'venue':'trading212','currency':'GBP','market':'UK','sdrt':'liable'}}}
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    for _ in range(2):
        subprocess.run([sys.executable,'-m','foundation.execution','--config',str(path),'--once'],
                       check=True,timeout=20,capture_output=True,text=True)
    broker=Trading212Paper(tmp_path/'state/paper.sqlite3',system='isa-paper',currency='GBP',initial_cash=10000)

    assert broker.positions()=={'EXAMPLE':5.} and broker.cash()==9497.
    assert len(broker.fills())==1
    assert len(EvidenceLog(tmp_path/'state/service.jsonl').read())==2
