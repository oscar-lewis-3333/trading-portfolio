"""Frozen-decision cost research. Spreads are full effective spread; execution uses half per side."""
import numpy as np
import pandas as pd
from reversal_edge_reference import edge_rolling


def audit_bars(panel):
    bars=panel[['Open','High','Low','Close']]
    masks=pd.DataFrame(index=panel.index)
    masks['invalid_price']=~(np.isfinite(bars).all(axis=1)&bars.gt(0).all(axis=1))
    masks['ohlc_flag']=(bars[['Open','Close']].max(axis=1)>bars.High+1e-8)|(bars[['Open','Close']].min(axis=1)<bars.Low-1e-8)
    masks['suspended']=panel.known_suspended|panel.close_suspended
    masks['zero_volume']=panel.Volume.eq(0)
    masks['missing_volume']=~np.isfinite(panel.Volume)
    masks['flat_bar']=panel.High.eq(panel.Low)
    masks['quarantined']=panel.unverified_quote
    masks['valid']=~masks.any(axis=1)&panel.High.gt(panel.Low)&panel.Volume.gt(0)
    return masks


def pair_spreads(bars):
    """Published CS overnight range translation and AR two-day corrected spreads."""
    h,l,c=bars.High,bars.Low,bars.Close
    prev=c.shift()
    gap=(prev-h).where(prev>h,0)+(prev-l).where(prev<l,0)
    th,tl=h+gap,l+gap
    beta=np.log(th/tl)**2+np.log(h.shift()/l.shift())**2
    gamma=np.log(pd.concat([th,h.shift()],axis=1).max(axis=1)/pd.concat([tl,l.shift()],axis=1).min(axis=1))**2
    alpha=(np.sqrt(2*beta)-np.sqrt(beta))/(3-2*np.sqrt(2))-np.sqrt(gamma/(3-2*np.sqrt(2)))
    valid=bars.notna().all(axis=1)&bars.shift().notna().all(axis=1)
    cs=(2*np.tanh(alpha/2)).clip(lower=0).where(valid)
    eta=(np.log(h)+np.log(l))/2
    ar2=4*(np.log(prev)-eta.shift())*(np.log(prev)-eta)
    ar=np.sqrt(ar2.clip(lower=0)).where(valid)
    return pd.DataFrame({'CS':cs,'AR':ar,'AR_signed_squared':ar2.where(valid)})


def estimate_spreads(panel, windows=(21,63), valid_fraction=.8):
    masks=audit_bars(panel); blocks=[]
    for ticker,p in panel.groupby(level='ticker',sort=True):
        b=p[['Open','High','Low','Close']].where(masks.loc[p.index,'valid'],axis=0).droplevel('ticker')
        pairs=pair_spreads(b); out=pd.DataFrame(index=b.index)
        for w in windows:
            n=int(np.ceil(valid_fraction*w)); npair=int(np.ceil(valid_fraction*(w-1)))
            valid=b.notna().all(axis=1).rolling(w,min_periods=w).sum().ge(n)
            for method in ['CS','AR']:
                out[f'{method}{w}']=pairs[method].rolling(w-1,min_periods=npair).mean().where(valid)*10000
            out[f'EDGE{w}']=edge_rolling(b,w,sign=False,min_periods=npair+1).where(valid)*10000
        out['ticker']=ticker;blocks.append(out.set_index('ticker',append=True))
    return pd.concat(blocks).sort_index(),masks


def execution_arrays(market,schedule,decisions,lengths):
    """Prepare a complete calendar; preserve next-open timing and final liquidation."""
    tickers=market.index.get_level_values('ticker').unique().sort_values()
    dates=schedule.index
    def a(col):return market[col].unstack('ticker').reindex(index=dates,columns=tickers).to_numpy()
    weights=decisions.target_weight.unstack('ticker').reindex(index=dates,columns=tickers).to_numpy(float)
    # Missing ticker rows mean zero allocation; missing dates mean no rebalance.
    decision_days=dates.isin(decisions.index.get_level_values('Date'))
    weights[decision_days]=np.nan_to_num(weights[decision_days])
    lens=lengths.reindex(dates).to_numpy()
    return dict(dates=dates,tickers=tickers,open=a('open_gbp'),close=a('close_gbp'),div=a('dividend_gbp'),suspended=a('known_suspended'),usable=a('open_reference_usable'),weights=weights,decision_days=decision_days,lengths=lens)


def replay(a, spread_per_side_bps=10., sdrt_buy_bps=0., cash_aer=.038, initial=10000., record=True):
    """Array implementation of the existing netted, cash-interest ledger, frozen targets.

    Rate arrays are execution-date by ticker, already lagged by the caller. Taxes
    apply only to additions. Cancelled allocations stay in the funding budget.
    """
    dates,tickers=a['dates'],a['tickers']; shape=a['open'].shape
    spread=np.broadcast_to(np.asarray(spread_per_side_bps,float)/10000,shape)
    tax=np.broadcast_to(np.asarray(sdrt_buy_bps,float)/10000,shape)
    if not np.isfinite(spread).all() or not np.isfinite(tax).all() or (spread<0).any() or (tax<0).any() or (spread+tax>=1).any():raise ValueError('Invalid or missing cost per side')
    units=np.zeros(shape[1]); marks=np.zeros(shape[1]); exits=np.full(shape[1],-1)
    cash=float(initial);reserve=0.;pending=None;pending_exit=-1;rows=[];orders=[];contrib=[]
    for i,date in enumerate(dates):
        interest=0. if i==0 else cash*np.expm1(np.log1p(cash_aer)*(date-dates[i-1]).days/365.25)
        cash+=interest; old=units.copy(); oldmarks=marks.copy()
        div=a['div'][i];held=units>0
        if not np.isfinite(div[held]).all():raise ValueError('Unknown held dividend')
        dividends=float(np.dot(units[held],div[held]));reserve+=dividends;marks[held]-=div[held]
        op=a['open'][i];quote=a['usable'][i]&~a['suspended'][i]&np.isfinite(op)&(op>0)
        due=held&(exits<=i)&quote
        buys=np.zeros(shape[1],bool) if pending is None else quote&(~held|due)
        active=due|buys
        current=np.where(due,units*np.nan_to_num(op),0.)
        requested=np.zeros(shape[1]) if pending is None else np.where(buys,pending,0.)
        total=0. if pending is None else float(pending.sum())
        available=cash+current.sum();sr=spread[i];br=sr+tax[i]
        def funding(s):return s*total+np.dot(sr,np.maximum(current-s*requested,0.))
        scale=0. if total==0 else 1.
        if total and funding(1.)>available:
            lo,hi=0.,1.
            for _ in range(60):
                mid=(lo+hi)/2
                if funding(mid)<=available:lo=mid
                else:hi=mid
            scale=lo
        budgets=scale*requested
        target=np.where(budgets<=current,budgets,(budgets+br*current)/(1+br))
        nxt=np.divide(target,op,out=np.zeros(shape[1]),where=active)
        old_due=np.where(due,units,0.)
        same=(old_due>0)&(nxt>0)&np.isclose(nxt,old_due,rtol=1e-12,atol=0)
        nxt[same]=old_due[same]
        change=np.where(active,nxt-old_due,0.)
        notional=np.abs(change)*np.nan_to_num(op)
        scost=notional*sr;tcost=np.where(change>0,notional*tax[i],0.);cost=scost+tcost
        cash+=float(np.sum(np.where(change<0,notional-cost,np.where(change>0,-notional-cost,0.))))
        if cash < -1e-10*max(1,available):raise ArithmeticError('Negative cash')
        cash=max(0.,cash);units[active]=nxt[active];exits[active&(nxt>0)]=pending_exit
        marks[active&(units>0)]=op[active&(units>0)]
        close=a['close'][i];fresh=~a['suspended'][i]&np.isfinite(close)&(close>0)
        marks[(units>0)&fresh]=close[(units>0)&fresh]
        if not np.isfinite(marks[units>0]).all() or (marks[units>0]<=0).any():raise ValueError('Invalid valuation')
        value=units*marks;equity=cash+value.sum()+reserve
        rows.append((equity,cash,float(value.sum()),reserve,float(cost.sum()),float(scost.sum()),float(tcost.sum()),float(notional.sum()),interest))
        if record:
            j=np.flatnonzero(change)
            orders.extend((date,tickers[k],'buy' if change[k]>0 else 'sell',float(notional[k]),float(scost[k]),float(tcost[k]),float(cost[k])) for k in j)
            # Cash-neutral stock P&L, including reserved dividends and execution costs.
            pnl=value-old*oldmarks+np.where(change<0,notional,np.where(change>0,-notional,0.))+np.where(held,old*np.nan_to_num(div),0.)-cost
            contrib.append(pnl)
        pending=None;pending_exit=-1
        if a['decision_days'][i]:
            pending=a['weights'][i]*(cash+value.sum());pending_exit=i+1+int(a['lengths'][i])
            if pending_exit>=len(dates):raise ValueError('Exit outside calendar')
    daily=pd.DataFrame(rows,index=dates,columns=['equity_gbp','cash_gbp','stock_value_gbp','dividend_reserve_gbp','cost_gbp','spread_gbp','sdrt_gbp','notional_gbp','cash_interest_gbp'])
    daily['equity_return']=daily.equity_gbp.pct_change(fill_method=None)
    if units.any():raise ValueError('Unliquidated final position')
    return {'daily':daily,'orders':pd.DataFrame(orders,columns=['Date','ticker','side','notional_gbp','spread_gbp','sdrt_gbp','cost_gbp']),'pnl':pd.DataFrame(contrib,index=dates if record else None,columns=tickers if record else None)}


def fill_and_lag(estimates, liquidity, eligible):
    """Same-decision-close positive peer medians, then one complete-session lag.

    Bucket boundaries are in GBP median traded value per session. No data rows
    are dropped. Warm-up with no valid peers uses an explicit 60bp FULL-spread
    house fallback (30bp per side), never an unlabelled zero.
    """
    buckets=pd.cut(liquidity,[-np.inf,100000,250000,1000000,np.inf],labels=['below_100k','100k_250k','250k_1m','above_1m'],right=False).astype(object).fillna('unknown')
    dates=estimates.index.get_level_values('Date');filled=pd.DataFrame(index=estimates.index);sources=pd.DataFrame(index=estimates.index)
    for method in estimates:
        raw=estimates[method];peers=raw.where(eligible&raw.gt(0))
        bucket=peers.groupby([dates,buckets],observed=True).transform('median')
        universe=peers.groupby(dates).transform('median')
        v=raw.fillna(bucket).fillna(universe).fillna(60.)
        source=pd.Series(np.select([raw.notna(),bucket.notna(),universe.notna()],['direct','bucket_median','universe_median'],default='house_30bp_per_side'),index=raw.index)
        filled[method]=v.groupby(level='ticker').shift(1)/2
        sources[method]=source.groupby(level='ticker').shift(1)
    return filled,sources,buckets
