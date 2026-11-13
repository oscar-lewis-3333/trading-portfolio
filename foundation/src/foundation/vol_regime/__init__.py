
#causal variance forecasting and regime inputs

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from foundation.models import estimator, Protocol, decay_weights
from foundation.validate import walk_forward, circular_indices
from foundation.data import utc, asof_timestamp


def har_features(rv,*,log=False):
    #RV observed at index t forecasts RV at t+1. can never shift future RV into features
    
    if not isinstance(rv,pd.Series) or not isinstance(rv.index,pd.DatetimeIndex) or rv.index.has_duplicates or not rv.index.is_monotonic_increasing:
        raise ValueError('Sorted unique realised variance series required')
    utc(rv.index)
    if np.isinf(rv).any() or (rv.dropna()<=0).any(): 
        raise ValueError('Realised variance must be positive, missing may remain')
    
    f=pd.DataFrame({'daily':rv,'weekly':rv.rolling(5,min_periods=5).mean(),'monthly':rv.rolling(22,min_periods=22).mean()})
    return np.log(f) if log else f


def qlike(actual,forecast):
    a,b=np.asarray(actual,float),np.asarray(forecast,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<=0).any() or (b<=0).any():
        raise ValueError('QLIKE needs aligned positive finite variances')
    ratio=a/b
    return ratio-np.log(ratio)-1


def forecast(rv,*,protocol=Protocol(min_train=60,retrain_every=20),challenger=True):

    #HAR OLS, and LogHAR OLS with training residual bias correction, alongside next-RV OOS.
    X=har_features(rv)
    y=rv.shift(-1)
    end=pd.Series(rv.index,index=rv.index).shift(-1)

    valid=X.notna().all(axis=1)&y.notna()&end.notna()
    X=X.loc[valid]
    y=y.loc[valid]

    ends=pd.DatetimeIndex(end.loc[valid])
    names=['har','loghar']+(['xgboost'] if challenger else [])
    out=pd.DataFrame(np.nan,index=X.index,columns=names); fits=[]; clipped={n:0 for n in names}
    for fold in walk_forward(X.index,ends,min_train=protocol.min_train,test_size=protocol.retrain_every,
                             train_window=protocol.train_window,purge=protocol.purge,embargo=protocol.embargo):
        
        weights=decay_weights(X.index[fold.train],protocol.half_life) if protocol.half_life else None
        for name in names:
            train=X.iloc[fold.train]; test=X.iloc[fold.test]; target=y.iloc[fold.train]
            if name=='loghar':
                train=np.log(train); test=np.log(test); target=np.log(target)

            model=estimator('xgboost') if name=='xgboost' else LinearRegression()
            model.fit(train,target,sample_weight=weights)
            pred=model.predict(test)
            if name=='loghar':
                residual=target-model.predict(train)
                variance=np.average(residual**2,weights=weights)
                pred=np.exp(pred+.5*variance)

            floor=float(y.iloc[fold.train].median())*1e-6
            clipped[name]+=int((pred<floor).sum())
            out.iloc[fold.test,out.columns.get_loc(name)]=np.maximum(pred,floor)
        fits.append({'cutoff':fold.cutoff.isoformat(),'last_label_end':ends[fold.train].max().isoformat(),'training_rows':len(fold.train)})
    scored=out.dropna()
    losses=pd.DataFrame({n:qlike(y.loc[scored.index],scored[n]) for n in names},index=scored.index)
    return {'predictions':out,'actual':y,'losses':losses,'mean_qlike':losses.mean().to_dict(),'fits':fits,'clipped':clipped}


def compare_qlike(losses,*,baseline='har',challenger='xgboost',block_size=20,resamples=10000,seed=0):
    a=losses[[baseline,challenger]].to_numpy(float)
    if not np.isfinite(a).all():
        raise ValueError('QLIKE comparisons require complete common OOS rows')
    
    difference=a[:,0]-a[:,1]
    idx=circular_indices(len(a),block_size,resamples,seed)
    draws=difference[idx].mean(axis=1)
    lower=float(np.quantile(draws,.05))
    return {'mean_loss_reduction':float(difference.mean()),'lower_95':lower,'challenger_passes':lower>0}


def regime_features(close,*,trend_window=200,vol_window=20):
    if trend_window<2 or vol_window<2: 
        raise ValueError('Invalid regime windows')
    if np.isinf(close).any() or (close.dropna()<=0).any(): 
        raise ValueError('Invalid price')
    
    returns=close.pct_change(fill_method=None)

    #features at decision day t use only complete observation days strictly before t
    return pd.DataFrame({'trend_ratio':(close/close.rolling(trend_window).mean()-1).shift(1),
                         'realised_vol':returns.rolling(vol_window).std().shift(1),
                         'drawdown':(close/close.cummax()-1).shift(1)})


def gates(at,*,features,vix=None,macro_calendar=None,gamma=None,macro_buffer='30min'):
    #unknown inputs yield None

    at=asof_timestamp(at)
    result={}
    if 'available_at' not in features: 
        raise ValueError('Feature availability required')
    
    f=features.copy()
    f['available_at']=utc(f.available_at)
    f=f[f.available_at<=at]
    result['trend']=None if f.empty or pd.isna(f.sort_values('available_at').iloc[-1].trend_ratio) else bool(f.sort_values('available_at').iloc[-1].trend_ratio>=0)
    if vix is not None:
        v=vix.copy()
        v['available_at']=utc(v.available_at)
        v=v[v.available_at<=at].sort_values('available_at')
        if v.empty or not np.isfinite(v.iloc[-1][['front','three_month']].astype(float)).all() or min(v.iloc[-1].front,v.iloc[-1].three_month)<=0:
            result['vix_contango']=None
        else: result['vix_contango']=bool(v.iloc[-1].front<=v.iloc[-1].three_month)
    if macro_calendar is not None:
        m = macro_calendar.copy()
        m['available_at'] = utc(m.available_at)
        m['event_at'] = utc(m.event_at)
        buffer = pd.Timedelta(macro_buffer)
        if buffer<pd.Timedelta(0):
            raise ValueError('Negative macro buffer')
        
        known=m[m.available_at<=at]
        result['outside_macro_window']=not bool(((known.event_at-at).abs()<=buffer).any())
    if gamma is not None:
        result['positive_gamma_proxy'] = None if gamma.available_at>at or gamma.value is None else bool(gamma.value>0)
    return result


@dataclass(frozen=True)
class GammaProxy:
    value: float | None
    available_at: pd.Timestamp
    sign_assumption: str
    source: str

    def __post_init__(self):
        asof_timestamp(self.available_at)
        if self.value is not None and not np.isfinite(self.value): raise ValueError('Nonfinite gamma proxy')
        if not self.sign_assumption or not self.source: raise ValueError('Explicit sign assumption and provenance required')


def gamma_proxy(snapshot,*,at,inventory_sign=None,assumption=None,source='options_capture'):
    #USD gamma per 1% move
    at=asof_timestamp(at)
    if inventory_sign is None:
        return GammaProxy(None,at,'dealer inventory sign unobserved',source)
    
    required=['gamma','open_interest','spot','multiplier','available_at']
    if not set(required).issubset(snapshot) or not assumption: 
        raise ValueError('Snapshot/sign methodology required')
    
    f=snapshot.copy(); f['available_at']=utc(f.available_at)
    if (f.available_at>at).any() or not np.isfinite(f[required[:-1]]).all().all(): 
        raise ValueError('Future or missing option input')
    if (f[['open_interest','spot','multiplier']]<0).any().any():
        raise ValueError('Negative option inputs')
    
    signs=np.asarray(inventory_sign,float)
    if signs.shape!=(len(f),) or not np.isin(signs,[-1,0,1]).all():
        raise ValueError('One explicit position sign per contract')
    
    value=float(np.sum(signs*f.gamma*f.open_interest*f.spot**2*f.multiplier*.01))
    return GammaProxy(value,at,assumption,source)

__all__ = ["GammaProxy", "compare_qlike", "forecast", "gamma_proxy", "gates", "har_features", "qlike", "regime_features"]
