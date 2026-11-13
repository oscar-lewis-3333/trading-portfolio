"""Self-financing, next-observation execution with per-trade cost attribution."""
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from foundation.costs import Trade
from foundation.data import utc


def _trade(asset,quantity,price,at,instruments):
    if asset not in instruments: raise ValueError(f'Missing instrument metadata for {asset}')
    return Trade(instrument_id=asset,side='buy' if quantity>0 else 'sell',quantity=abs(float(quantity)),
                 price=float(price),at=at,**instruments[asset])


def backtest_targets(targets,prices,cost_model,*,instruments,initial=10000.,cash_aer=0.):
    #long-only target weights to execute at first price after the decision timestamp

    prices=prices.copy()
    targets=targets.copy()
    prices.index=pd.DatetimeIndex(utc(prices.index)); targets.index=pd.DatetimeIndex(utc(targets.index))
    if prices.empty or targets.empty or prices.index.has_duplicates or targets.index.has_duplicates or not prices.index.is_monotonic_increasing or not targets.index.is_monotonic_increasing:
        raise ValueError('Nonempty sorted unique clocks required')
    if targets.columns.has_duplicates or prices.columns.has_duplicates:
        raise ValueError('Duplicate instruments')
    if 'CASH' not in targets or not np.isfinite(targets).all().all() or (targets<0).any().any() or not np.allclose(targets.sum(axis=1),1,atol=1e-10,rtol=0):
        raise ValueError('Targets must be finite, nonnegative and sum to one including CASH')
    if initial<=0 or not np.isfinite(initial) or cash_aer<=-1 or not np.isfinite(cash_aer): 
        raise ValueError('Invalid initial capital/cash yield')
    
    assets=sorted(set(targets)-{'CASH'})
    if not set(assets).issubset(prices):
        raise ValueError('Missing instrument prices')
    px=prices[assets].to_numpy(float)
    weights=targets[assets].to_numpy(float)
    decisions=targets.index
    if 'foundation_tranche' in targets.attrs or targets.index.name=='tranche_execution_at':
        from foundation.portfolio.rebalance import _execution_plan
        execution,decisions=_execution_plan(targets,prices.index)
        weights=targets.iloc[execution][assets].to_numpy(float)
    else:
        execution=prices.index.searchsorted(targets.index,side='right')
    if (execution>=len(prices)).any() or len(set(execution))!=len(execution):
        raise ValueError('Every target needs a distinct future executable observation')
    plans=dict(zip(execution,range(len(execution))))
    units=np.zeros(len(assets))
    cash=float(initial)

    prior=float(initial)
    rows=[]; orders=[]
    for i,at in enumerate(prices.index):
        if i:
            cash*=np.exp(np.log1p(cash_aer)*(at-prices.index[i-1]).total_seconds()/(365.25*86400))
        p=px[i]
        active=units!=0
        if i in plans: 
            active=active|(weights[plans[i]]>0)
        if (~np.isfinite(p[active])).any() or (p[active]<=0).any():
            raise ValueError(f'Missing held/traded valuation at {at}')
        safe=np.where(np.isfinite(p)&(p>0),p,1.)
        nav=cash+float(units@safe)
        fee=0.
        turnover=0.
        if i in plans:
            w=weights[plans[i]]
            def evaluate(after):
                new=after*w/safe
                change=new-units

                trades=[(j,_trade(assets[j],change[j],safe[j],at,instruments)) for j in range(len(assets)) if abs(change[j]*safe[j])>1e-9]
                return new,trades,sum(cost_model.quote(t).total for _,t in trades)
            def error(after):
                return after+evaluate(after)[2]-nav
            if error(0)>0:
                raise ValueError('Costs exceed capital')
            after=brentq(error,0,nav,xtol=1e-9)
            new,trades,fee=evaluate(after)
            if abs(after+fee-nav)>1e-6:
                raise ValueError('Discontinuous fee schedule prevents exact target solve')
            for _,t in trades:
                orders.append({**asdict(t),**cost_model.quote(t).to_dict(),'notional':t.notional,'decision_at':decisions[plans[i]]})
                turnover+=t.notional

            cash=nav-fee-float(new@safe)
            units=new
        value=cash+float(units@safe)
        if value<=0 or cash < -1e-6:
            raise ValueError('Nonpositive value or negative cash')
        
        rows.append({'net_return':value/prior-1,'equity':value,'cash':cash,'cost':fee,
                     'traded_notional':turnover,'gross_before_cost':nav/prior-1})
        prior=value
    return {'ledger':pd.DataFrame(rows,index=prices.index),'trades':pd.DataFrame(orders),
            'terminal_units':dict(zip(assets,units)), 'terminal_cash':cash,'initial':initial}


def run_events(events,quotes,holding,cost_model,*,instruments,initial=10000.,slots=10):
    #long-only event slots, entry at next tradable quote, so exit holding rows later

    required={'event_id','instrument_id','available_at'}
    if not required.issubset(events) or not {'instrument_id','at','price','tradable'}.issubset(quotes):
        raise ValueError('Incomplete event/quote contract')
    if holding<1 or slots<1 or initial<=0:
        raise ValueError('Positive horizon/slots/capital required')
    ev=events.copy()
    q=quotes.copy()
    ev['available_at']=utc(ev.available_at)
    q['at']=utc(q['at'])
    if ev.event_id.duplicated().any() or q.duplicated(['instrument_id','at']).any() or q.tradable.isna().any(): 
        raise ValueError('Duplicate or incomplete events/quotes')
    if not pd.api.types.is_bool_dtype(q.tradable): 
        raise ValueError('tradable must be boolean')
    
    usable=q.tradable&np.isfinite(q.price)&q.price.gt(0)
    plans=[]
    outcomes=[]
    for e in ev.sort_values(['available_at','event_id']).itertuples():
        g=q[q.instrument_id==e.instrument_id].sort_values('at').reset_index(drop=True)
        ok=g.tradable&np.isfinite(g.price)&g.price.gt(0)
        entry=np.flatnonzero((g['at']>e.available_at)&ok)
        if not len(entry):
            outcomes.append({'event_id':e.event_id,'status':'no_entry'})
            continue

        j=int(entry[0])
        exit_ix=np.flatnonzero((np.arange(len(g))>=j+holding)&ok)
        k=int(exit_ix[0]) if len(exit_ix) else None

        plans.append(dict(event_id=e.event_id,instrument_id=e.instrument_id,entry=g.at[j,'at'],
                          exit=None if k is None else g.at[k,'at'],entry_price=float(g.at[j,'price']),
                          exit_price=None if k is None else float(g.at[k,'price'])))
    grid=pd.DatetimeIndex(sorted(q['at'].unique()))
    held={}
    cash=float(initial)
    prior=initial
    rows=[]
    orders=[]
    marks={}
    for at in grid:
        fresh=q[(q['at']==at)&usable]
        marks.update(dict(zip(fresh.instrument_id,fresh.price)))
        fee=0.

        for key,p in list(held.items()):
            if p['exit']==at:
                t=_trade(p['instrument_id'],-p['units'],p['exit_price'],at,instruments)
                c=cost_model.quote(t)
                cash+=t.notional-c.total

                fee+=c.total
                orders.append({**asdict(t),**c.to_dict(),'event_id':key,'notional':t.notional})
                outcomes.append({'event_id':key,'status':'closed'})
                del held[key]
        for p in [p for p in plans if p['entry']==at]:
            if len(held)>=slots or p['instrument_id'] in {h['instrument_id'] for h in held.values()}:
                outcomes.append({'event_id':p['event_id'],'status':'capacity_or_overlap'})
                continue
            value=cash+sum(h['units']*marks[h['instrument_id']] for h in held.values())
            budget=min(cash,value/slots)
            if budget<=0: 
                outcomes.append({'event_id':p['event_id'],'status':'no_cash'})
                continue
            def total(units):
                if units==0:
                    return 0.
                t=_trade(p['instrument_id'],units,p['entry_price'],at,instruments)
                return t.notional+cost_model.quote(t).total
            
            amount=brentq(lambda u:total(u)-budget,0,budget/p['entry_price'])
            if abs(total(amount)-budget)>1e-6:
                raise ValueError('Cannot fund event under minimum commission')
            
            t=_trade(p['instrument_id'],amount,p['entry_price'],at,instruments)
            c=cost_model.quote(t)
            cash-=t.notional+c.total; fee+=c.total
            held[p['event_id']]={**p,'units':amount}
            orders.append({**asdict(t),**c.to_dict(),'event_id':p['event_id'],'notional':t.notional})

        value=cash+sum(p['units']*marks[p['instrument_id']] for p in held.values())
        stale=sum(p['instrument_id'] not in set(fresh.instrument_id) for p in held.values())
        rows.append({'equity':value,'cash':cash,'net_return':value/prior-1,'cost':fee,'stale_positions':stale})
        prior=value
    outcomes.extend({'event_id':key,'status':'censored_open'} for key in held)
     
    return {'ledger':pd.DataFrame(rows,index=grid),'trades':pd.DataFrame(orders),
            'outcomes':pd.DataFrame(outcomes),'open_positions':held,'initial':initial}
