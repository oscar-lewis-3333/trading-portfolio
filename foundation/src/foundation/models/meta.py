
#barrier labelling, window calibration and sizing
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from foundation.data import utc
from foundation.costs import Trade
from . import estimator, _fit, feature_audit


def triple_barrier(bars,events,*,profit_take=.02,stop_loss=.01,max_holding=5):
    #OHLC labelling for a single asset, with entries occuring at bar open

    if min(profit_take,stop_loss)<=0 or max(profit_take,stop_loss)>=1 or max_holding<1:
        raise ValueError('Invalid barrier protocol')
    
    b=bars.copy(); b.index=pd.DatetimeIndex(utc(b.index))
    if b.index.has_duplicates or not b.index.is_monotonic_increasing or not {'open','high','low','close'}.issubset(b):
        raise ValueError('Sorted unique OHLC bars required')
    if events.index.has_duplicates:
        raise ValueError('Unique event IDs required')
    
    rows=[]
    for event_id,event in events.iterrows():
        at=utc([event.entry_at])[0]
        side=event.side
        row={'event_id':event_id,'entry_at':at,'label_end':pd.NaT,'label':np.nan,'gross_return':np.nan,'status':'censored'}
        if side not in (-1,1) or at not in b.index:
            raise ValueError('Side must be +/-1 and entry an observed bar open')
        
        start=b.index.get_loc(at)
        entry=float(b.iloc[start]['open'])
        if not np.isfinite(entry) or entry<=0: 
            rows.append({**row,'status':'missing_entry'})
            continue

        for k in range(start,min(start+max_holding,len(b))):
            bar=b.iloc[k]
            prices=bar[['open','high','low','close']].to_numpy(dtype=float)
            if not np.isfinite(prices).all() or (prices<=0).any():
                row['status']='missing_path'
                break
            if bar.high<max(bar.open,bar.close,bar.low) or bar.low>min(bar.open,bar.close,bar.high):
                raise ValueError('Invalid OHLC ordering')
            opening=side*(bar.open/entry-1)
            hi=max(side*(bar.high/entry-1),side*(bar.low/entry-1))
            lo=min(side*(bar.high/entry-1),side*(bar.low/entry-1))
            outcome=None
            status=None
            if opening>=profit_take or opening<=-stop_loss:
                 outcome=opening; status='gap'
            elif hi>=profit_take and lo<=-stop_loss: 
                row.update(label_end=b.index[k],status='ambiguous_both')
                break
            elif hi>=profit_take:
                outcome=profit_take
                status='profit'
            elif lo<=-stop_loss:
                outcome=-stop_loss
                status='stop'
            elif k==start+max_holding-1: 
                outcome=side*(bar.close/entry-1)
                status='vertical'
            if outcome is not None:
                #label_end is bar END, supplied, or next observed bar begins. high/low not known at open 
                if 'available_at' in b:
                    end=utc([bar.available_at])[0]

                elif k+1<len(b):
                    end=b.index[k+1]
                else: 
                    row['status']='missing_label_availability'
                    break
                if end<=b.index[k]: 
                    raise ValueError('OHLC label cannot be available at/before bar open')
                row.update(label_end=end,label=int(outcome>0),gross_return=float(outcome),status=status)
                break

        rows.append(row)
    return pd.DataFrame(rows).set_index('event_id')


class CalibratedMetaModel:
    def __init__(self,family='xgboost'):
        self.family=family
        self.model=None
        self.calibrator=None
        self.ready_at=None

    def fit(self,X,y,label_end,*,available_at,calibration_start,ready_at,sample_weight=None):
        feature_audit(X,available_at,X.index)
        t=pd.DatetimeIndex(utc(X.index))
        end=pd.DatetimeIndex(utc(label_end))
        cal,ready=utc([calibration_start,ready_at])
        if not t.is_monotonic_increasing or cal>=ready or not X.index.equals(y.index) or len(end)!=len(X) or (end<t).any():
            raise ValueError('Invalid chronological calibration contract')
        
        train=(t<cal)&(end<cal)
        calibration=(t>=cal)&(end<ready)&(t<ready)
        if train.sum()<10 or calibration.sum()<10 or y.loc[train].nunique()!=2 or y.loc[calibration].nunique()!=2 or not y.isin([0,1]).all():
            raise ValueError('Both chronological subsets need >=10 observations and both classes')
        self.model=estimator(self.family,classification=True)
        weights=None if sample_weight is None else np.asarray(sample_weight)[train]
        _fit(self.model,X.loc[train],y.loc[train],weights)
        raw=self.model.predict_proba(X.loc[calibration])[:,1]
        logits=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))[:,None]
        self.calibrator=LogisticRegression(C=1.,random_state=0).fit(logits,y.loc[calibration])
        self.ready_at=ready
        self.columns=list(X.columns)
        return self

    def predict_probability(self,X,*,available_at):
        if self.model is None:
            raise ValueError('Fit and calibrate first')
        feature_audit(X,available_at,X.index)
        if list(X.columns)!=self.columns or (pd.DatetimeIndex(X.index)<self.ready_at).any():
            raise ValueError('Prediction precedes calibration availability or features differ')
        raw=self.model.predict_proba(X)[:,1]
        logits=np.log(np.clip(raw,1e-6,1-1e-6)/np.clip(1-raw,1e-6,1-1e-6))[:,None]
        return pd.Series(self.calibrator.predict_proba(logits)[:,1],index=X.index,name='probability')


def cost_gated_size(probability,expected_return,entry_trade,cost_model,*,buffer=0.,maximum=1.,forecast_is_net=False,exit_trade=None):
    #gross forecasts exceed estimated round-trip costs, net forecasts exceed a buffer
    
    if not np.isfinite([probability,expected_return,buffer,maximum]).all() or not 0<=probability<=1 or not 0<=maximum<=1 or buffer<0:
        raise ValueError('Invalid probability/forecast/size')
    if exit_trade is None:
        from dataclasses import replace
        exit_trade=replace(entry_trade,side='sell' if entry_trade.side=='buy' else 'buy')
    if exit_trade.instrument_id!=entry_trade.instrument_id: 
        raise ValueError('Cost legs must refer to the same instrument')
    
    cost=(cost_model.quote(entry_trade).total+cost_model.quote(exit_trade).total)/entry_trade.notional
    net=float(expected_return if forecast_is_net else expected_return-cost)
    return {'size':maximum*max(0.,2*probability-1) if net>buffer else 0.,
            'expected_net_return':net,'estimated_round_trip_cost':cost,'passes_cost_gate':bool(net>buffer)}
