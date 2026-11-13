
#point in time activated signal replay using what we have
import numpy as np
import pandas as pd
from foundation import portfolio, validate
from foundation.costs import CostModel, break_even_cost
from foundation.data import asof_timestamp
from foundation.reporting import build_results, metrics
from foundation.signals import run_events
from foundation.store import fingerprint


COST_COLUMNS=['commission','spread_slippage','sdrt','fx','external','total']


def firing_frame(store, registry, *, as_of, names):
    #first qualifying revision per logical event/trigger.

    cutoff=asof_timestamp(as_of); seen=set(); rows=[]; candidates=[]
    for event in store.timeline(cutoff):
        for name in names:
            spec=registry.get(name)
            at=max(event.available_at,asof_timestamp(event.event_at)) if spec.latency_origin=='event_at' else event.available_at
            if at<=cutoff: candidates.append((at,name,event))
    for at,name,event in sorted(candidates,key=lambda x:(x[0],x[1],x[2].event_id)):
        visible={(e.source,e.event_id):e for e in store.as_of(at)}
        if visible[event.source,event.event_id].content_hash!=event.content_hash: 
            continue
        key=(name,event.source,event.event_id)
        if key in seen: 
            continue
        firings=registry.fire(event,at,names=[name])
        if not firings:
            continue

        seen.add(key)
        for instrument in event.instrument_ids:
            rows.append({'event_id':fingerprint([firings[0].key,instrument]),'instrument_id':instrument,
                         'available_at':at,'trigger':name,'source_event_id':event.event_id})
    return pd.DataFrame(rows,columns=['event_id','instrument_id','available_at','trigger','source_event_id'])


def replay(store,registry,quotes,*,names,holding,cost_model,instruments,volatility, allocation_known_at,initial=10000.):
    #frozen inverse-vol sleeve budgets, with an entry strictly later than the decision

    if quotes.empty or initial<=0: 
        raise ValueError('Quotes and positive capital required')
    q=quotes.copy(); q['at']=q['at'].map(asof_timestamp)
    if asof_timestamp(allocation_known_at)>q['at'].min(): 
        raise ValueError('Allocation learned after replay began')
    vol=pd.Series(volatility,dtype=float)
    if vol.index.has_duplicates or not set(vol.index)<=set(q.instrument_id): 
        raise ValueError('Unique observable allocation universe required')
    
    weights=portfolio.inverse_vol(vol)
    events=firing_frame(store,registry,as_of=q['at'].max(),names=names)
    if not set(events.instrument_id)<=set(weights.index): 
        raise ValueError('Fired instrument lacks predeclared allocation')
    
    runs={instrument:run_events(events[events.instrument_id==instrument],q,holding,cost_model, instruments=instruments,initial=initial*weight,slots=1)
          for instrument,weight in weights.items()}
    ledger=sum((r['ledger'][['equity','cash','cost','stale_positions']] for r in runs.values()))
    ledger['net_return']=ledger.equity/ledger.equity.shift(1,fill_value=initial)-1
    trades=pd.concat([r['trades'] for r in runs.values()],ignore_index=True)

    if trades.empty:
        trades=pd.DataFrame(columns=['instrument_id','notional',*COST_COLUMNS])
    outcomes=pd.concat([r['outcomes'] for r in runs.values()],ignore_index=True)

    return {'ledger':ledger,'trades':trades,'outcomes':outcomes,'events':events,'weights':weights,
            'open_positions':{k:r['open_positions'] for k,r in runs.items()},'initial':initial}


def study(store,registry,quotes,*,variants,primary,protocol,benchmarks,instruments, volatility,allocation_known_at,provenance,trial_log=None,adv_at_trade=None):
    #frozen variants, with 0/10/20/30bp transaction costs sensitivities, against both benchmarks

    if primary not in variants or not variants or not protocol.get('project'):
        raise ValueError('Named project and primary variant required')
    if protocol.get('variants')!={k:list(v) for k,v in variants.items()}:
        raise ValueError('Variants must match frozen protocol')
    log=trial_log or validate.TrialLog(); annual=protocol['periods_per_year']
    def run(names,bps):
        return replay(store,registry,quotes,names=names,holding=protocol['holding'],cost_model=CostModel(bps),
                      instruments=instruments,volatility=volatility,allocation_known_at=allocation_known_at,
                      initial=protocol['initial_capital'])
    runs={name:run(names,10) for name,names in variants.items()}
    aligned=pd.DataFrame({name:r['ledger'].net_return for name,r in runs.items()})
    if len(aligned)<30: 
        raise ValueError('At least 30 replay observations required for validation')
    
    for name,series in aligned.items():
        usable=series.std(ddof=1)>0
        log.record(project=protocol['project'],trial_id=name,protocol=protocol,
                   sharpe_value=validate.sharpe(series,annual) if usable else None,periods_per_year=annual,
                   status='evaluated' if usable else 'rejected',
                   note='activated event variant; historical trial coverage may be incomplete' if usable else 'zero-variance event replay; no inference')
    if aligned[primary].std(ddof=1)<=0:
        raise ValueError('Primary event replay has zero variance; trial recorded as rejected; use replay for raw cash ledger')
    natural,hold=benchmarks(10)
    if not natural.index.equals(aligned.index) or not hold.index.equals(aligned.index):
        raise ValueError('Benchmark dates must match event valuation grid exactly')
    
    chosen=runs[primary]; strategy=aligned[primary]
    draws=validate.circular_indices(len(strategy),protocol['block_size'],protocol['resamples'],protocol['seed'])
    differences=aligned.sub(natural,axis=0).to_numpy()
    observed=differences.mean(axis=0)
    centered=differences[draws].mean(axis=1)-observed
    raw=(1+(centered>=observed).sum(axis=0))/(len(draws)+1)
    corrected,_=validate.holm(raw)
    scenario_rows=[]
    excess=[]
    for bp in (0,10,20,30):
        result=chosen if bp==10 else run(variants[primary],bp)
        bench,buy_hold=benchmarks(bp)
        if not bench.index.equals(strategy.index) or not buy_hold.index.equals(strategy.index): 
            raise ValueError('Scenario benchmark dates differ')
        
        scenario_rows.append({'bps_per_side':bp,**metrics(result['ledger'].net_return,annual)})
        excess.append(float((result['ledger'].net_return-bench).mean()))
    exposure=(1-chosen['ledger'].cash/chosen['ledger'].equity).clip(0,1)
    controls=validate.exposure_controls(natural,exposure,window=protocol['control_window'],seed=protocol['seed'])

    dates=strategy.index; ends=pd.Series(dates,index=dates)
    trades=chosen['trades']
    if len(trades):
        entries=trades[trades.side=='buy']; exits=trades[trades.side=='sell'].set_index('event_id')
        for entry in entries.itertuples():
            end=asof_timestamp(exits.loc[entry.event_id,'at']) if entry.event_id in exits.index else dates[-1]+pd.Timedelta('1ns')
            mask=(ends.index>=asof_timestamp(entry.at))&(ends.index<=end)
            ends.loc[mask]=ends.loc[mask].map(lambda old:max(old,end))
    folds=list(validate.walk_forward(dates,ends,min_train=protocol['min_train'],test_size=protocol['test_size'], purge=protocol['purge'],embargo=protocol['embargo']))
    capacity={'status':'unavailable: no contemporaneous ADV supplied'}
    if len(trades) and adv_at_trade is not None:
        capacity=validate.turnover_capacity(trades.notional,nav=protocol['initial_capital'],adv=adv_at_trade(trades), observations=len(strategy),periods_per_year=annual)
    overfitting={'status':'unavailable: one variant'}
    if len(variants)>1:
        try: 
            overfitting=validate.pbo(aligned,blocks=4,periods_per_year=annual)
        except ValueError as error:
            if str(error)!='Zero variance CSCV candidate':
                raise
            overfitting={'status':'unavailable: zero-variance candidate in a CSCV split; trials retained'}
    validation={'signal_kind':'activated','weights':chosen['weights'],
                'walk_forward':[{'cutoff':f.cutoff,'train_rows':len(f.train),'test_rows':len(f.test)} for f in folds],
                'pbo':overfitting,
                'holm':dict(zip(aligned.columns,corrected)),
                'controls':{'random_skip_mean':float(controls['random_skip'].mean().mean()),
                            'vol_matched_missing':int(controls['warmup_missing'].sum())},
                'cost_scenarios':scenario_rows,'break_even':break_even_cost([0,10,20,30],excess),
                'turnover_capacity':capacity,'event_outcomes':chosen['outcomes'],
                'unresolved_instruments':sum(not e.instrument_ids for e in store.timeline(dates[-1]))}
    report=build_results(project=protocol['project'],protocol=protocol,strategy=strategy,natural_benchmark=natural,
                         buy_and_hold=hold,trades=trades,validation=validation,provenance=provenance,
                         limitations=['Research-only long-only fixed allocation. No automatic admission.',
                                      'Eligibility, identifiers, source rights and benchmark construction require caller-audited inputs.',
                                      'Unknown instruments cannot create positions; missing quotes censor or defer fills.',
                                      'Historical trial coverage is not presumed complete; folds are diagnostics for fixed rules.'],
                         periods_per_year=annual,publication_date=protocol.get('publication_date'),trial_log=log,
                         block_size=protocol['block_size'],resamples=protocol['resamples'])
    return {'report':report,'runs':runs}
