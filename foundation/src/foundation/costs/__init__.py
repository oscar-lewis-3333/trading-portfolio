

#costs in GBP, with rates as decimals.

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from foundation.store import file_hash


@dataclass(frozen=True)
class Trade:
    instrument_id: str
    venue: str
    side: str
    quantity: float
    price: float
    currency: str
    at: pd.Timestamp
    market: str = 'UK'
    asset_type: str = 'stock'
    sdrt: str = 'unknown'  
    liquidity: str = 'taker'
    fx_conversion: bool = False
    usd_per_currency: float = 1.
    multiplier: float = 1.

    def __post_init__(self):
        if not self.instrument_id or self.side not in {'buy','sell'} or self.liquidity not in {'maker','taker'}:
            raise ValueError('Invalid trade identity/side/liquidity')
        if not np.isfinite([self.quantity,self.price,self.usd_per_currency,self.multiplier]).all() or min(self.quantity,self.price,self.usd_per_currency,self.multiplier)<=0:
            raise ValueError('Positive finite quantities, prices, FX and multiplier required')
        if pd.Timestamp(self.at).tzinfo is None or pd.isna(self.at):
            raise ValueError('UTC-aware trade time required')
        if self.sdrt not in {'liable','exempt','unknown'}:
            raise ValueError('Unknown SDRT classification')

    @property
    def notional(self):
        return self.quantity*self.price*self.multiplier


@dataclass(frozen=True)
class CostBreakdown:
    commission: float
    spread_slippage: float
    sdrt: float
    fx: float
    external: float
    currency: str
    source: str
    limitations: tuple = ()

    @property
    def total(self):
        return self.commission+self.spread_slippage+self.sdrt+self.fx+self.external

    def to_dict(self):
        return {**asdict(self), 'total': self.total}


class MeasuredTable:
    #one-sided spread+slip, venue and instrument fees

    def __init__(self, frame, source='measured'):
        required={'instrument_id','venue','side','available_at','valid_from','valid_to','per_side_bps','max_notional','currency'}
        if not required.issubset(frame):
            raise ValueError(f'Missing measured cost columns: {required-set(frame)}')
        self.frame=frame.copy(); self.source=source
        for c in ['available_at','valid_from','valid_to']:
            if any(pd.Timestamp(x).tzinfo is None for x in self.frame[c]):
                raise ValueError('Cost clocks must include timezone')
            
            self.frame[c]=pd.to_datetime(self.frame[c],utc=True)
        if self.frame.empty or self.frame[list(required)].isna().any().any():
            raise ValueError('Empty or incomplete cost table')
        
        if not self.frame.side.isin(['buy','sell','both']).all():
            raise ValueError('Invalid cost side')
        if (self.frame.valid_to<=self.frame.valid_from).any():
            raise ValueError('Invalid validity interval')
        
        a=self.frame[['per_side_bps','max_notional']].to_numpy(float)
        if not np.isfinite(a).all() or (a[:,0]<0).any() or (a[:,0]>=10000).any() or (a[:,1]<=0).any():
            raise ValueError('Invalid measured rates/capacity')
        if self.frame.duplicated(['instrument_id','venue','side','available_at','valid_from']).any():
            raise ValueError('Ambiguous measured cost vintage')

    @classmethod
    def load(cls,path):
        path=Path(path)
        frame=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)

        return cls(frame, f'{path.name}:sha256:{file_hash(path)}')

    def rate(self,t):
        f=self.frame
        rows=f[(f.instrument_id==t.instrument_id)&(f.venue==t.venue)&f.side.isin([t.side,'both'])&
               (f.available_at<=t.at)&(f.valid_from<=t.at)&(f.valid_to>t.at)&
               (f.max_notional>=t.notional)&(f.currency==t.currency)]
        
        if rows.empty:
            raise ValueError(f'No available measured cost within size support for {t.instrument_id}')
        rows=rows.sort_values('available_at'); latest=rows[rows.available_at==rows.available_at.max()]
        if len(latest)!=1:
            raise ValueError('Overlapping cost estimates at the same vintage')
        
        return float(latest.iloc[0].per_side_bps)/10000


class CostModel:
    def __init__(self, flat_bps=10., *, measured=None, ibkr_us_plan='fixed',
                 futures_per_contract=None, external_fee=0.):
        if not np.isfinite(flat_bps) or not 0<=flat_bps<10000 or external_fee<0 or not np.isfinite(external_fee):
            raise ValueError('Invalid flat costs')
        
        if ibkr_us_plan not in {'fixed','tiered'}:
            raise ValueError('Unknown IBKR plan')
        
        self.flat_bps=flat_bps; self.measured=measured; self.ibkr_us_plan=ibkr_us_plan
        self.futures_per_contract=dict(futures_per_contract or {})
        self.external_fee=external_fee

    def quote(self,t):
        n=t.notional; commission=tax=fx=0.; limits=[]
        if t.venue not in {'trading212','ibkr','revolut_x','alpaca_paper','simulation'}:
            raise ValueError('Unsupported venue')
        
        if t.market=='UK' and t.asset_type=='stock' and t.side=='buy':
            if t.sdrt=='unknown':
                raise ValueError('Verify SDRT liability/exemption explicitly; AIM name alone is insufficient')
            tax=n*.005 if t.sdrt=='liable' else 0.
        if t.venue=='trading212':
            if t.asset_type not in {'stock','etf'}:
                raise ValueError('Unsupported Trading 212 instrument')
            fx=n*.0015 if t.fx_conversion else 0.

        elif t.venue=='revolut_x':
            if t.asset_type!='crypto':
                raise ValueError('Revolut X adapter is spot crypto only')
            commission=0. if t.liquidity=='maker' else n*.0009

        elif t.venue=='ibkr':
            if t.asset_type=='future':
                rate=self.futures_per_contract.get(t.instrument_id)
                if rate is None or not np.isfinite(rate) or rate<0:
                    raise ValueError('Explicit product/currency per-contract fee required')
                
                if not float(t.quantity).is_integer(): raise ValueError('Futures require whole contracts')
                commission=t.quantity*rate
            elif t.asset_type in {'stock','etf'} and t.market=='US' and t.currency=='USD':
                if not float(t.quantity).is_integer():
                    commission=max(.01,n*.01)
                else:
                    rate,minimum=(.005,1.) if self.ibkr_us_plan=='fixed' else (.0035,.35)
                    commission=min(max(t.quantity*rate,minimum),n*.01)

            elif t.asset_type in {'stock','etf'} and t.market=='UK' and t.currency=='GBP':
                commission=max(n*.0005,1.)
                limits.append('UK tiered broker commission; exchange/clearing/regulatory fees require external_fee')
            else:
                raise ValueError('Unsupported IBKR product/market/currency combination')
            if t.fx_conversion:
                fx=max(n*.00002,2./t.usd_per_currency)
            limits.append('Dated third-party fees must be supplied separately; financing and borrow excluded')

        rate=self.measured.rate(t) if self.measured else self.flat_bps/10000
        source=self.measured.source if self.measured else f'flat {self.flat_bps:g}bp/side assumption'
        
        return CostBreakdown(commission,n*rate,tax,fx,self.external_fee,t.currency,source,tuple(limits))

    def quote_partial_fill(self, trade, *, prior_quantity=0., prior_notional=0., commission_paid=0.):
        #per order minimum commisions or cap, charged cumalatively

        from dataclasses import replace
        if not np.isfinite([prior_quantity,prior_notional,commission_paid]).all() or min(prior_quantity,prior_notional,commission_paid)<0:
            raise ValueError('Invalid cumulative fill state')
        
        total_quantity=prior_quantity+trade.quantity
        average_price=(prior_notional+trade.notional)/(total_quantity*trade.multiplier)

        cumulative=self.quote(replace(trade,quantity=total_quantity,price=average_price))
        current=self.quote(trade)
        return replace(current,commission=max(0.,cumulative.commission-commission_paid))

    def attribute(self,trades):
        return pd.DataFrame([{**asdict(t), 'notional': t.notional, **self.quote(t).to_dict()} for t in trades])


def scenarios():
    return {bps:CostModel(bps) for bps in (10,20,30)}


def break_even_cost(costs_bps, excess):
    #saved grid mean-excess crossing
    x,y=np.asarray(costs_bps,float),np.asarray(excess,float)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<2 or not np.isfinite(x).all() or (x<0).any() or (np.diff(x)<=0).any():
        raise ValueError('Strictly increasing nonnegative cost grid required')
    
    result={'bps_per_side':None,'method':None,'segment':None}
    if not np.isfinite(y).all(): 
        return {**result,'method':'missing strategy or benchmark bucket attribution'}
    if y[0]<=0: 
        return {**result,'method':'nonpositive contribution at zero cost'}
    if x[0]!=0:
        raise ValueError('Include zero cost')
    
    for i in range(len(x)-1):
        if y[i]>=0 and y[i+1]<=0:
            return {'bps_per_side':float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])),
                    'method':'linear interpolation','segment':[float(x[i]),float(x[i+1])]}
    slope=(y[-1]-y[-2])/(x[-1]-x[-2])
    return {'bps_per_side':float(x[-1]-y[-1]/slope) if slope<0 else None,
            'method':'linear extrapolation beyond saved grid' if slope<0 else 'no declining crossing supported',
            'segment':[float(x[-2]),float(x[-1])]}


def replay_break_even(replay_excess, *, maximum_bps=100.):
    lo, hi = replay_excess(0.), replay_excess(maximum_bps)
    if not np.isfinite([lo,hi]).all():
        raise ValueError('Nonfinite replay objective')
    
    if lo<=0: 
        return {'bps_per_side':None,'method':'nonpositive at zero'}
    if hi>0:
        return {'bps_per_side':None,'method':'above evaluated range','lower_bound_bps':maximum_bps}
    
    return {'bps_per_side':float(brentq(replay_excess,0,maximum_bps,xtol=1e-9)), 'method':'replayed root'}


def fit_quote_table(quotes, *, available_at, valid_from, valid_to, max_notional):
    #median half spread, with depth, impact unmeasured

    required={'instrument_id','venue','currency','bid','ask','received_at'}
    if not required.issubset(quotes): 
        raise ValueError('Incomplete quotes')
    q=quotes.copy()
    for column in ['received_at']:
        if any(pd.Timestamp(x).tzinfo is None for x in q[column]):
            raise ValueError('Naive quote clock')
        
    receipt=pd.to_datetime(q.received_at,utc=True)
    if (receipt>pd.Timestamp(available_at)).any(): 
        raise ValueError('Fit uses future quotes')
    if not np.isfinite(q[['bid','ask']]).all().all() or (q.bid<=0).any() or (q.ask<q.bid).any():
        raise ValueError('Invalid/crossed quote')
    
    q['per_side_bps']=10000*(q.ask-q.bid)/(q.ask+q.bid)
    f=q.groupby(['instrument_id','venue','currency'],as_index=False).per_side_bps.median()
    return MeasuredTable(f.assign(side='both',available_at=available_at,valid_from=valid_from,
                                  valid_to=valid_to,max_notional=max_notional),'median half quoted spread; impact/fills unmeasured')

__all__ = [
    "CostBreakdown", "CostModel", "MeasuredTable", "Trade", "break_even_cost",
    "fit_quote_table", "replay_break_even", "scenarios",
]
