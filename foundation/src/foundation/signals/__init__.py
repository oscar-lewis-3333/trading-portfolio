
#signal metadata, causal construction, cross-sectional transforms

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from foundation.data import utc, asof_timestamp
from foundation._research.catalogue import catalogue_frame
from foundation._research.synthetic import make_panel


@dataclass(frozen=True)
class SignalSpec:
    name: str
    hypothesis: str
    inputs: tuple[str,...]
    availability_lag: str = '0s'

    def __post_init__(self):
        if not self.name or not self.hypothesis or not self.inputs or pd.Timedelta(self.availability_lag)<pd.Timedelta(0):
            raise ValueError('Name, economic hypothesis, inputs and nonnegative lag required')


class Registry:
    def __init__(self): self._signals={}

    def register(self,spec:SignalSpec,build:Callable):
        if spec.name in self._signals:
            raise ValueError('Duplicate signal')
        
        self._signals[spec.name]=(spec,build)

    def specs(self):
        return [s for s,_ in self._signals.values()]

    def build(self,name,panel,decision_at):
        spec,build=self._signals[name]
        at=asof_timestamp(decision_at)
        if not set(spec.inputs).issubset(panel) or not {'instrument_id','available_at','event_at'}.issubset(panel):
            raise ValueError('Missing signal inputs/clocks')
        
        f=panel.copy()
        f['available_at']=utc(f.available_at)
        f['event_at']=utc(f.event_at)
        f=f[(f.available_at+pd.Timedelta(spec.availability_lag)<=at)&(f.event_at<=at)]
        result=build(f.copy(),at)
        if not isinstance(result,pd.Series) or result.index.has_duplicates or np.isinf(result.to_numpy(float)).any():
            raise ValueError('Builder must return instrument-indexed scores; NaN is an explicit missing signal')
        if not set(result.index).issubset(set(f.instrument_id)): 
            raise ValueError('Signal references an unavailable instrument')

        return pd.DataFrame({'instrument_id':result.index,'decision_at':at,'signal':result.to_numpy(), 'missing_signal':result.isna().to_numpy()})


def normalise(frame,columns,*,method='rank',date='decision_at'):
    f=frame.copy()
    if method not in {'rank','z'}:
        raise ValueError('Unknown normalisation')
    
    for c in columns:
        if np.isinf(f[c].to_numpy(float)).any(): 
            raise ValueError('Infinite signal')
        
        f[c+'_missing']=f[c].isna()
        g=f.groupby(date)[c]
        if method=='rank':
            f[c]=g.rank(pct=True,method='average')-.5
        else:
            sd=g.transform('std'); f[c]=(f[c]-g.transform('mean'))/sd.where(sd>0)
    return f


def neutralise(frame,signal='signal',*,sector='sector',size='size',beta='beta',date='decision_at'):
    f=frame.copy()
    f['neutral_signal']=np.nan
    f['neutral_missing']=True
    for _,g in f.groupby(date,sort=False):
        mask=g[[signal,sector,size,beta]].notna().all(axis=1)&g[size].gt(0)
        sub=g.loc[mask]
        if sub.empty: 
            continue

        x=pd.concat([pd.Series(1.,index=sub.index,name='intercept'), np.log(sub[size]).rename('log_size'),sub[beta],
                     pd.get_dummies(sub[sector],drop_first=True,dtype=float)],axis=1).to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(sub[signal]).all():
            raise ValueError('Nonfinite neutralisation input')
        if len(sub)<=np.linalg.matrix_rank(x): 
            continue

        y=sub[signal].to_numpy(float)
        coef=np.linalg.lstsq(x,y,rcond=None)[0]
        f.loc[sub.index,'neutral_signal']=y-x@coef
        f.loc[sub.index,'neutral_missing']=False

    return f


from .harness import backtest_targets, run_events  #noqa: E402

__all__ = ["Registry", "SignalSpec", "backtest_targets", "neutralise", "normalise", "run_events"]
