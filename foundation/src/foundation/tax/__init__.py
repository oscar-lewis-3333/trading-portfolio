
#UK post-2008 ordinary share matching in decimals

from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

ZERO=Decimal('0')


def dec(value):
    result=Decimal(str(value))
    if not result.is_finite(): 
        raise ValueError('Finite decimal required')
    
    return result


def tax_date(value):
    #accept only a given UK tax calendar date, not a guessed UTC date

    if isinstance(value,str):
        return date.fromisoformat(value)
    if type(value) is date:
        return value
    
    raise ValueError('Supply UK tax_date as YYYY-MM-DD; convert timestamps explicitly')


def match(transactions,*,as_of,opening_pools=None,final=True):

    end=tax_date(as_of); grouped=defaultdict(dict); seen=set()
    for t in transactions:
        if t['id'] in seen: 
            raise ValueError('Duplicate transaction ID')
        
        seen.add(t['id'])
        if t.get('account_type','taxable')!='taxable' or t.get('uk_resident',True) is not True:
            raise ValueError('Only taxable UK-resident transactions supported')
        if t.get('asset_type','share') not in {'share','fungible_token'}: 
            raise ValueError('Unsupported tax asset/event')
        
        side=t['side']; day=tax_date(t['tax_date'])
        if side not in {'buy','sell'} or day<date(2008,4,6) or day>end: 
            raise ValueError('Unsupported side/date or future record')
        
        key=(t['owner'],t['capacity'],t['instrument_id'])
        if not all(key): 
            raise ValueError('Owner, capacity and legal instrument identity required')
        
        q,amount,fee=map(dec,[t['quantity'],t['amount_gbp'],t.get('fee_gbp',0)])
        if q<=0 or amount<0 or fee<0 or side=='sell' and fee>amount: 
            raise ValueError('Invalid tax amounts')
        
        row=grouped[key].setdefault((day,side),{'day':day,'side':side,'q':ZERO,'value':ZERO,'ids':[]})
        row['q']+=q
        row['value']+=amount+fee if side=='buy' else amount-fee
        row['ids'].append(t['id'])

    opening_pools=opening_pools or {}; matches=[]; disposals=[]; pools={}
    for key in sorted(set(grouped)|set(opening_pools)):
        rows=grouped.get(key,{})
        buys=sorted([dict(r,remaining=r['q'],unit=r['value']/r['q']) for r in rows.values() if r['side']=='buy'],key=lambda r:r['day'])
        sells=sorted([dict(r,remaining=r['q'],cost=ZERO) for r in rows.values() if r['side']=='sell'],key=lambda r:r['day'])

        opening=opening_pools.get(key,{'quantity':0,'cost_gbp':0})
        pool_q,pool_cost=dec(opening['quantity']),dec(opening['cost_gbp'])
        if pool_q<0 or pool_cost<0 or pool_q==0 and pool_cost!=0: 
            raise ValueError('Invalid opening pool')
        if final and any(s['day']+timedelta(days=30)>end for s in sells):
            raise ValueError('30-day acquisition horizon incomplete; request provisional explicitly')
        
        #custody quantity is separate from tax pool matching
        custody=pool_q
        for day in sorted({r['day'] for r in [*buys,*sells]}):
            custody+=sum((b['q'] for b in buys if b['day']==day),ZERO)
            custody-=sum((s['q'] for s in sells if s['day']==day),ZERO)
            if custody<0:
                raise ValueError('Short/unfunded disposal is outside matching scope')
            
        def allocate(s,b,q,kind):
            cost=q*b['unit']; s['cost']+=cost; s['remaining']-=q; b['remaining']-=q
            matches.append({'pool':key,'disposal_date':s['day'].isoformat(),'acquisition_date':b['day'].isoformat(),
                            'rule':kind,'quantity':str(q),'cost_gbp':str(cost),'disposal_ids':s['ids'],'acquisition_ids':b['ids']})
            
        #same day priority takes precedence over all earlier disposal matches
        for s in sells:
            for b in buys:
                if b['day']==s['day']: allocate(s,b,min(s['remaining'],b['remaining']),'same_day')
        for s in sells:
            for b in buys:
                if s['remaining']>0 and b['remaining']>0 and s['day']<b['day']<=s['day']+timedelta(days=30):
                    allocate(s,b,min(s['remaining'],b['remaining']),'30_day')
        for day in sorted({r['day'] for r in [*buys,*sells]}):
            for b in buys:
                if b['day']==day:
                    pool_q+=b['remaining']; pool_cost+=b['remaining']*b['unit']
            for s in sells:
                if s['day']!=day: 
                    continue
                q=s['remaining']
                if q>pool_q:
                    raise ValueError('Insufficient Section 104 pool; missing history or unsupported short')
                
                if q:
                    cost=pool_cost if q==pool_q else pool_cost*q/pool_q
                    pool_q-=q; pool_cost-=cost; s['cost']+=cost
                    matches.append({'pool':key,'disposal_date':day.isoformat(),'acquisition_date':None,'rule':'section_104',
                                    'quantity':str(q),'cost_gbp':str(cost),'disposal_ids':s['ids'],'acquisition_ids':[]})
                disposals.append({'pool':key,'tax_date':day.isoformat(),'quantity':str(s['q']),'net_proceeds_gbp':str(s['value']),
                                  'allowable_cost_gbp':str(s['cost']),'gain_gbp':str(s['value']-s['cost']),
                                  'final':day+timedelta(days=30)<=end,'ids':s['ids']})
        pools[key]={'quantity':str(pool_q),'cost_gbp':str(pool_cost)}

    return {'disposals':disposals,'matches':matches,'pools':[{'pool':key,**value} for key,value in pools.items()],
            'as_of':end.isoformat(),'final':all(d['final'] for d in disposals),'total_gain_gbp':str(sum((dec(d['gain_gbp']) for d in disposals),ZERO))}


def dividends(records):
    out=[]
    ids=set()
    for r in records:
        if r['id'] in ids: 
            raise ValueError('Duplicate dividend')
        
        ids.add(r['id']); gross,withheld=dec(r['gross_gbp']),dec(r.get('withheld_gbp',0))
        if min(gross,withheld)<0 or withheld>gross:
            raise ValueError('Invalid dividend amounts')
        
        out.append({**r,'tax_date':tax_date(r['tax_date']).isoformat(),'gross_gbp':str(gross),
                    'withheld_gbp':str(withheld),'net_cash_gbp':str(gross-withheld)})
    return {'records':out,'gross_gbp':str(sum((dec(r['gross_gbp']) for r in out),ZERO)),
            'withheld_gbp':str(sum((dec(r['withheld_gbp']) for r in out),ZERO))}

__all__ = ["dec", "dividends", "match", "tax_date",]
