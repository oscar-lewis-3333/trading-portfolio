
#one unmuteable results schema, HTML tear sheet for each project

from datetime import date,datetime
from decimal import Decimal
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
from foundation.store import atomic_json, fingerprint, lock
from foundation.validate import paired_bootstrap, year_spread, publication_split, deflated_sharpe


def serialise(value):
    if isinstance(value,dict): 
        return {str(k):serialise(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray,pd.Index)):
        return [serialise(v) for v in value]
    if isinstance(value,pd.DataFrame):
        return serialise(value.reset_index().to_dict(orient='records'))
    if isinstance(value,pd.Series): 
        return serialise(value.rename('value').reset_index().to_dict(orient='records'))
    if isinstance(value,(date,datetime,pd.Timestamp)):
        return value.isoformat()
    if isinstance(value,Decimal):
        return str(value)
    if isinstance(value,(float,np.floating)): 
        return float(value) if np.isfinite(value) else None
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    if isinstance(value,(int,np.integer)):
        return int(value)
    if value is pd.NA or value is pd.NaT:
        return None
    
    return value


def metrics(returns,periods_per_year):

    a=np.asarray(returns,float)

    if len(a)<2 or not np.isfinite(a).all() or (a<=-1).any() or periods_per_year<=0: 
        raise ValueError('Complete finite dated returns above -100% required')
    wealth=np.r_[1.,np.cumprod(1+a)]
    
    return {'observations':len(a),'terminal_wealth':float(wealth[-1]),'total_return':float(wealth[-1]-1),
            'cagr':float(wealth[-1]**(periods_per_year/len(a))-1),
            'sharpe':float(a.mean()/a.std(ddof=1)*np.sqrt(periods_per_year)) if a.std(ddof=1)>0 else None,
            'max_drawdown':float((wealth/np.maximum.accumulate(wealth)-1).min())}


def build_results(*,project,protocol,strategy,natural_benchmark,buy_and_hold,trades,validation, provenance,limitations,periods_per_year=252,publication_date=None,trial_log=None,
                  block_size=20,resamples=10000,rebalance_sensitivity=None):
    if not project or not protocol or not provenance: 
        raise ValueError('Project, frozen protocol and provenance required')
    if not isinstance(strategy.index,pd.DatetimeIndex) or strategy.index.tz is None or strategy.index.has_duplicates or not strategy.index.is_monotonic_increasing:
        raise ValueError('Sorted unique UTC-aware result dates required')
    streams={'strategy':strategy,'natural_benchmark':natural_benchmark,'buy_and_hold':buy_and_hold}

    for s in streams.values():
        if not s.index.equals(strategy.index):
            raise ValueError('Results require exact common dates; report missingness before comparison')
    for column in ['commission','spread_slippage','sdrt','fx','external','total']:
        if column not in trades:
            raise ValueError('Per-trade cost components required, even for an empty trade frame')
    costs=trades[['commission','spread_slippage','sdrt','fx','external','total']]
    if not np.isfinite(costs).all().all() or (costs<0).any().any() or not np.allclose(costs.drop(columns='total').sum(axis=1),costs.total,rtol=0,atol=1e-8):
        raise ValueError('Cost attribution does not reconcile')
    inference={name:paired_bootstrap(strategy,streams[name],block_size=block_size,resamples=resamples,periods_per_year=periods_per_year) for name in ['natural_benchmark','buy_and_hold']}
    publication=None
    if publication_date is not None:
        publication={k:metrics(v,periods_per_year) if len(v)>1 else None for k,v in publication_split(strategy,publication_date).items()}
    result={'schema_version':1,'project':project,'status':'research_only','protocol':protocol,'protocol_sha256':fingerprint(protocol),
            'provenance':provenance,'periods_per_year':periods_per_year,
            'performance':{k:metrics(v,periods_per_year) for k,v in streams.items()},'paired_inference':inference,
            'by_year':{k:year_spread(v,periods_per_year) for k,v in streams.items()},'publication_split':publication,
            'validation':validation,'deflated_sharpe':None if trial_log is None else deflated_sharpe(strategy,trial_log,periods_per_year=periods_per_year),
            'cost_attribution':trades,'total_costs':costs.sum().to_dict(),'returns':pd.DataFrame(streams),
            'rebalance_sensitivity':rebalance_sensitivity if rebalance_sensitivity is not None else
            {'schema_version':1,'status':'not_run','reason':'No rebalance-date comparison supplied'},
            'limitations':list(limitations),'automatic_promotion':False}
    
    return serialise(result)


def tear_sheet(result):
    esc = lambda value: html.escape(str(value))
    performance = pd.DataFrame(result['performance']).T.to_html(escape=True,float_format=lambda x:f'{x:.6f}')
    annual = pd.DataFrame(result['by_year']['strategy']).to_html(index=False,escape=True)
    costs = pd.DataFrame([result['total_costs']]).to_html(index=False,escape=True)

    details=esc(json.dumps({k:result[k] for k in ['protocol_sha256','provenance','paired_inference','deflated_sharpe','validation','publication_split']},indent=2))
    limits=''.join('<li>'+esc(x)+'</li>' for x in result['limitations'])

    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{esc(result['project'])}</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#14232c}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#f2f5f6;padding:20px}}h1{{font-size:30px}}</style>
<h1>{esc(result['project'])}</h1><p>Research only · schema 1 · no automatic promotion</p>
<h2>Both benchmark comparisons</h2>{performance}<h2>Yearly results</h2>{annual}
<h2>Trade cost attribution totals</h2>{costs}<p>Full per-trade components and return series are in results.json.</p>
<h2>Known limits</h2><ul>{limits}</ul><h2>Protocol and validation evidence</h2><pre>{details}</pre></html>'''


def write_results(directory,result):
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    with lock(root/'report.lock'):
        atomic_json(root/'results.json',serialise(result))
        sensitivity=result.get('rebalance_sensitivity', {'schema_version':1,'status':'not_run','reason':'No rebalance-date comparison supplied'})
        atomic_json(root/'rebalance_sensitivity.json',serialise(sensitivity))

        content=tear_sheet(result)
        path=root/'tear_sheet.html'
        if path.exists() and path.read_text()!=content: 
            raise FileExistsError('Immutable tear sheet differs')
        path.write_text(content)

    return {'json':root/'results.json','tear_sheet':path,'rebalance_sensitivity':root/'rebalance_sensitivity.json'}

__all__ = ["build_results", "metrics", "serialise", "tear_sheet", "write_results"]
