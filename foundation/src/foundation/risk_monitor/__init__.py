
#pretrade controls, persistent drawdown stop, drift diagnostics

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.stats import ks_2samp, binomtest
from foundation.store import lock, atomic_json


@dataclass(frozen=True)
class Limits:
    max_order_notional: float
    max_position_notional: float
    max_gross_notional: float
    max_net_notional: float
    max_daily_turnover: float
    allow_short: bool = False

    def __post_init__(self):
        numbers=[self.max_order_notional,self.max_position_notional,self.max_gross_notional,self.max_net_notional,self.max_daily_turnover]
        if not np.isfinite(numbers).all() or min(numbers)<=0: raise ValueError('Finite positive limits required')


def pretrade(positions,order,prices,limits,*,daily_turnover=0.,stopped=False):
    if stopped:
        raise PermissionError('Persistent risk stop is latched')
    asset,quantity=order['instrument_id'],float(order['quantity'])
    names=set(positions)|{asset}
    if not names.issubset(prices) or not np.isfinite(quantity) or quantity==0 or not np.isfinite(daily_turnover) or daily_turnover<0:
        raise ValueError('Complete valid pretrade inputs required')
    if any(not np.isfinite(prices[n]) or prices[n]<=0 for n in names) or any(not np.isfinite(v) for v in positions.values()): 
        raise ValueError('Invalid risk prices/positions')
    current=dict(positions); current[asset]=current.get(asset,0.)+quantity
    notional=abs(quantity)*prices[asset]
    exposures=np.array([current[n]*prices[n] for n in current])
    violations=[]
    if not limits.allow_short and (exposures < -1e-9).any():
        violations.append('short_position')
    if notional>limits.max_order_notional+1e-9:
        violations.append('order_notional')
    if (np.abs(exposures)>limits.max_position_notional+1e-9).any():
        violations.append('position_notional')
    if np.abs(exposures).sum()>limits.max_gross_notional+1e-9: 
        violations.append('gross_notional')
    if abs(exposures.sum())>limits.max_net_notional+1e-9: 
        violations.append('net_notional')
    if daily_turnover+notional>limits.max_daily_turnover+1e-9:
        violations.append('daily_turnover')
    if violations: 
        raise PermissionError(','.join(violations))
    return {'allowed':True,'notional':notional,'projected_gross':float(np.abs(exposures).sum())}


class DrawdownStop:
    def __init__(self,path,maximum_drawdown=.15):
        self.path=Path(path)
        if not 0<maximum_drawdown<1: 
            raise ValueError('Invalid drawdown threshold')
        self.maximum=maximum_drawdown

    def update(self,equity,at):
        from foundation.data import asof_timestamp
        at=asof_timestamp(at)
        if not np.isfinite(equity) or equity<=0:
            raise ValueError('Positive finite equity required')
        with lock(self.path.with_suffix('.risk.lock')):
            old=json.loads(self.path.read_text()) if self.path.exists() else {'peak':equity,'stopped':False,'maximum':self.maximum}
            if old['maximum']!=self.maximum: 
                raise ValueError('Risk threshold changed; use a reviewed new state')
            if 'at' in old and at<asof_timestamp(old['at']):
                raise ValueError('Out-of-order equity update')
            peak=max(old['peak'],equity); drawdown=1-equity/peak
            state={'peak':peak,'equity':equity,'drawdown':drawdown,'stopped':bool(old['stopped'] or drawdown>=self.maximum),
                   'at':at.isoformat(),'maximum':self.maximum}
            atomic_json(self.path,state,overwrite=True)
        return state

    def stopped(self):
        if not self.path.exists():
            raise ValueError('Initialise risk monitor before execution')
        
        state=json.loads(self.path.read_text())
        if state['maximum']!=self.maximum: 
            raise ValueError('Risk config mismatch')
        return bool(state['stopped'])


def drift(*,slippage_bps,backtest_slippage_bps,hits,expected_hit_rate,features,reference_features,alpha=.01,slippage_tolerance=5.):
    from foundation.validate import holm
    s=np.asarray(slippage_bps,float)
    h=np.asarray(hits)
    if len(s)<20 or not np.isfinite(s).all() or not np.isin(h,[0,1]).all() or len(h)<20 or not 0<expected_hit_rate<1:
        raise ValueError('At least 20 observed fills/trades and valid reference required')
    
    if not np.isfinite([backtest_slippage_bps,slippage_tolerance]).all() or slippage_tolerance<0: 
        raise ValueError('Invalid slippage reference')
    names=sorted(features)
    if set(names)!=set(reference_features) or not names:
        raise ValueError('Aligned named features required')
    tests=[]
    for name in names:
        a,b=np.asarray(features[name],float),np.asarray(reference_features[name],float)
        if min(len(a),len(b))<20 or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Insufficient finite drift sample')
        
        tests.append(ks_2samp(a,b))
    adj,reject=holm([t.pvalue for t in tests],alpha)
    hit_p=float(binomtest(int(h.sum()),len(h),expected_hit_rate,alternative='less').pvalue)
    alerts=[]
    if s.mean()>backtest_slippage_bps+slippage_tolerance:
        alerts.append('slippage')
    if hit_p<alpha:
        alerts.append('hit_rate')
    alerts.extend('feature:'+n for n,r in zip(names,reject) if r)

    return {'alerts':alerts,'mean_slippage_bps':float(s.mean()),'hit_rate_pvalue':hit_p,
            'features':{n:{'ks':float(t.statistic),'holm_p':float(p)} for n,t,p in zip(names,tests,adj)}}

__all__ = ["DrawdownStop", "Limits", "drift", "pretrade"]
