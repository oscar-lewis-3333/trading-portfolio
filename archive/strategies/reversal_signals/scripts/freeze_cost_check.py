"""Freeze selected 10bp decisions and inputs before changing source code."""
from pathlib import Path
import hashlib, json, shutil, sys
import pandas as pd
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P.parents[3]))
sys.path.insert(0,str(P.parents[2]/'original_portfolio/risk_management/src'))
sys.path.insert(0,str(P/'src'))
import reversal_research as research
from reversal_ml_research import ResearchCache, sha256
from reversal_ml_walk_forward import matched_benchmark
F=P/'data/cost_check_v1'
if (F/'manifest.json').exists():
    raise SystemExit('Freeze already exists; refusing to overwrite.')
F.mkdir(parents=True,exist_ok=True)
c=ResearchCache(P)
wpath=c.directory/'walk_forward.pkl'
wmeta=json.loads(wpath.with_suffix('.json').read_text())
assert wmeta=={'fingerprint':c.fingerprint,'sha256':sha256(wpath)}
w=pd.read_pickle(wpath)
i=research.prepare_extension(P)
e=research.run_holdout(i,'2024-01-01',use_cache=True)
r=research.run_risk_comparison(i,e,use_cache=True)
d=research.selected_decisions(i,e['choices'],start='2024-01-01')
fixed={'Original':d,'Capped':d.assign(target_weight=d.target_weight.clip(upper=.05))}
cases={}
for name,targets in fixed.items():
    cases['Fixed_'+name]={'decisions':targets,'benchmark':research.matched_benchmark(targets),'expected_daily':r['equity'][name]}
for name,targets in w['selected_decisions'].items():
    cases[name]={'decisions':targets,'benchmark':matched_benchmark(targets,targets.eligible),'expected_daily':w['daily'][(name,'Strategy')]['equity_gbp']}
for name,case in cases.items():
    case['lengths']=case['decisions'].holding_sessions.groupby(level='Date').first().astype('int64')
    case['expected_benchmark']=r['equity'][name.removeprefix('Fixed_')+' benchmark'] if name.startswith('Fixed_') else w['daily'][(name,'Benchmark')]['equity_gbp']
pd.to_pickle(cases,F/'cases.pkl')
pd.to_pickle({'ml_choices':w['choices'],'ml_choice_folds':w['choice_folds'],'fixed_choices':e['choices'],'spec':i['spec']},F/'choices_and_spec.pkl')
for rel in ['data/uk_extension_2023_2025_v1/prepared_2015_2025_v1.pkl','data/uk_extension_2023_2025_v1/prepared_manifest_v1.json','data/uk_holdout_2021_2023_v1/frozen_spec.json','results/ml_extension/continuous_cost_daily_returns.csv','README.md','notebooks/reversal_signals_uk.ipynb','notebooks/reversal_signals_ml.ipynb']:
    out=F/'original'/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/rel,out)
shutil.copytree(P/'src',F/'original/src',ignore=shutil.ignore_patterns('__pycache__'))
# Historical source is now /Volumes/data_storing/cold/trading_portfolio/isa_portfolio/uk_portfolio/equity_momentum/data/candidate_universe_v1.csv
shutil.copy2(P.parent/'equity_momentum/data/candidate_universe_v1.csv',F/'candidate_universe_v1.csv')
manifest={'version':1,'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'cache_fingerprint':c.fingerprint,'cache_sha256':sha256(wpath),'cache_provenance':c.provenance,'selection_cost_per_side_bps':10,'files':{str(p.relative_to(F)):sha256(p) for p in sorted(F.rglob('*')) if p.is_file()},'expected_equities':{k:float(v['expected_daily'].iloc[-1]) for k,v in cases.items()},'deferred':['Part (a) step 4 manual Trading 212 quotes','All part (b) real trades']}
(F/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert ResearchCache(P).fingerprint==c.fingerprint
print(json.dumps({'freeze':str(F),'files':len(manifest['files']),'expected_equities':manifest['expected_equities'],'source_unchanged':True},indent=2))
