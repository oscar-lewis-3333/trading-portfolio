
#allocation above immutable strategy streams, without touching the strategy

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from foundation._research.screen import shrunk_covariance as constant_correlation_covariance


def _positive(values):
    s=pd.Series(values,dtype=float)
    if s.empty or s.index.has_duplicates or not np.isfinite(s).all() or (s<=0).any():
        raise ValueError('Unique positive finite risk inputs required')
    return s


def inverse_vol(volatility):
    inverse=1/_positive(volatility)
    return (inverse/inverse.sum()).rename('weight')


def ledoit_wolf(returns):
    f=pd.DataFrame(returns)
    if f.shape[0]<2 or f.empty or f.columns.has_duplicates or not np.isfinite(f).all().all() or (f.std()<=0).any():
        raise ValueError('Complete nonconstant return streams required')
    
    fit=LedoitWolf().fit(f.to_numpy())
    cov=pd.DataFrame(fit.covariance_,index=f.columns,columns=f.columns)
    cov.attrs['shrinkage']=float(fit.shrinkage_)
    cov.attrs['target']='scaled identity'
    return cov


def hrp(covariance):
    c=pd.DataFrame(covariance).sort_index().sort_index(axis=1)
    a=c.to_numpy(float)
    if c.empty or not c.index.equals(c.columns) or not np.isfinite(a).all() or not np.allclose(a,a.T) or (np.diag(a)<=0).any() or np.linalg.eigvalsh(a).min() < -1e-10:
        raise ValueError('Symmetric positive semidefinite named covariance required')
    
    if len(c)==1: return pd.Series(1.,index=c.index,name='weight')
    sd=np.sqrt(np.diag(a))
    corr=np.clip(a/np.outer(sd,sd),-1,1)
    distance=np.sqrt(np.maximum((1-corr)/2,0))
    np.fill_diagonal(distance,0)

    order=leaves_list(linkage(squareform(distance,checks=False),method='single',optimal_ordering=True))
    weight=pd.Series(1.,index=c.index); clusters=[list(c.index[order])]
    def variance(names):
        matrix=c.loc[names,names].to_numpy()
        w=1/np.diag(matrix)
        w/=w.sum()
        return float(np.einsum('i,ij,j->',w,matrix,w))
    while clusters:
        children=[]
        for cluster in clusters:
            if len(cluster)<2:
                continue

            half=len(cluster)//2
            left,right=cluster[:half],cluster[half:]
            lv,rv=variance(left),variance(right)
            if lv+rv<=0: 
                raise ValueError('Degenerate cluster variance')
            
            alpha=rv/(lv+rv)
            weight.loc[left]*=alpha
            weight.loc[right]*=1-alpha
            children.extend([left,right])
        clusters=children

    return weight.rename('weight')


def vol_target(returns,*,target=.10,window=60,periods_per_year=252,max_leverage=1.):
    if min(target,max_leverage)<=0 or window<2 or periods_per_year<=0:
        raise ValueError('Invalid volatility target settings')
    s=pd.Series(returns,dtype=float)
    if np.isinf(s).any():
        raise ValueError('Infinite returns')
    
    vol=s.rolling(window,min_periods=window).std(ddof=1).shift(1)*np.sqrt(periods_per_year)
    scale=(target/vol.where(vol>0)).clip(upper=max_leverage)
    return pd.DataFrame({'scale':scale,'estimated_vol':vol,'unavailable':scale.isna()})


def fractional_kelly(deflated_sharpe_estimate,volatility,*,fraction=.25,cap=1.):

    if not np.isfinite([deflated_sharpe_estimate,volatility,fraction,cap]).all() or volatility<=0 or not 0<=fraction<=.25 or cap<0:
        raise ValueError('Finite risk inputs and Kelly fraction <=0.25 required')
    
    return min(cap,max(0.,fraction*deflated_sharpe_estimate/volatility))


def _weights(values,*,full=False):
    s=pd.Series(values,dtype=float)
    if s.empty or s.index.has_duplicates or not np.isfinite(s).all() or (s<0).any() or s.sum()>1+1e-10 or full and not np.isclose(s.sum(),1,atol=1e-10,rtol=0):
        raise ValueError('Invalid capital weights')
    
    return s


def no_trade_bands(current,target,*,band=.01):
    current=_weights(current,full=True); target=_weights(target,full=True)
    if not np.isfinite(band) or band<0: 
        raise ValueError('Invalid band')
    
    names=current.index.union(target.index); a=current.reindex(names,fill_value=0); b=target.reindex(names,fill_value=0)

    return a if (a-b).abs().max()<=band else b


def core_sweep(sleeves,budgets,core,*,current=None,band=.01):
    core=_weights(core,full=True)
    if 'CASH' in core and core['CASH']>0:
        raise ValueError('ISA core must be invested; use an explicit defensive fund')
    budgets=_weights(budgets)
    if set(sleeves)!=set(budgets.index): 
        raise ValueError('One capital budget per sleeve required')
    
    target=pd.Series(dtype=float)
    for name,weights in sleeves.items():
        w=_weights(weights).drop('CASH',errors='ignore')*budgets[name]
        target=target.add(w,fill_value=0)
    residual=max(0.,1-float(target.sum()))
    target=target.add(core*residual,fill_value=0)
    _weights(target,full=True)
    if current is not None:
        old=_weights(current,full=True)

        #existing idle cash must be swept even when total trade is small
        if old.get('CASH',0)==0:
            target=no_trade_bands(old,target,band=band)
    return target.rename('account_weight')


from .rebalance import tranche, rebalance_sensitivity, monthly_signal_dates  # noqa: E402
from .legacy_rebalance import rebalance_with_costs  # noqa: E402  live for quality_value

__all__ = [
    "core_sweep", "fractional_kelly", "hrp", "inverse_vol", "ledoit_wolf",
    "monthly_signal_dates", "no_trade_bands", "rebalance_sensitivity",
    "rebalance_with_costs", "tranche", "vol_target",
]
