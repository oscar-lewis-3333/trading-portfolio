
#persistent simulated broker adapters

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Protocol
import json
import os
import sqlite3
import time
import numpy as np
from foundation.contracts import TargetPositions
from foundation.data import asof_timestamp
from foundation.store import canonical, fingerprint, check_disk, lock
from foundation.risk_monitor import pretrade, Limits

STATES={'CREATED':{'SUBMITTED','CANCELLED','REJECTED'},'SUBMITTED':{'PARTIALLY_FILLED','FILLED','CANCELLED','REJECTED'},
        'PARTIALLY_FILLED':{'PARTIALLY_FILLED','FILLED','CANCELLED'},'FILLED':set(),'CANCELLED':set(),'REJECTED':set()}


def transition(state,event):
    if event not in STATES.get(state,set()): raise ValueError(f'Illegal order transition: {state} -> {event}')
    return event


class BrokerAdapter(Protocol):
    def positions(self) -> dict: ...
    def orders(self) -> list: ...
    def submit(self,order_id:str,payload:dict) -> dict: ...
    def cancel(self,order_id:str) -> None: ...


class PaperBroker:
    venue='simulation'
    def __init__(self,path,*,system,currency='GBP',initial_cash=10000.):
        self.path=Path(path); self.system=system
        self.currency=currency
        if not system or not currency or not np.isfinite(initial_cash) or initial_cash<=0:
            raise ValueError('Invalid paper account')
        check_disk(self.path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS positions(asset TEXT PRIMARY KEY, quantity REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,payload TEXT NOT NULL,state TEXT NOT NULL,filled REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS fills(id TEXT PRIMARY KEY,order_id TEXT NOT NULL,quantity REAL NOT NULL,price REAL NOT NULL,fee REAL NOT NULL,at TEXT NOT NULL,commission REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS targets(id TEXT PRIMARY KEY,hash TEXT NOT NULL,effective_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,order_id TEXT,event TEXT NOT NULL,at TEXT NOT NULL);
            ''')
            config=canonical({'system':system,'currency':currency,'venue':self.venue,'initial_cash':initial_cash})
            old=db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
            if old and old[0]!=config: 
                raise ValueError('Paper account configuration mismatch')
            
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('config',?)",(config,))
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('cash',?)",(str(initial_cash),))

    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.path,timeout=15,isolation_level=None)
        db.row_factory=sqlite3.Row
        try:
            yield db
        finally: 
            db.close()

    @contextmanager
    def transaction(self):
        check_disk(self.path)
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                 yield db; db.commit()
            except BaseException: 
                db.rollback()
                raise

    def positions(self):
        with self._db() as db: 
            return {r['asset']:r['quantity'] for r in db.execute('SELECT * FROM positions') if abs(r['quantity'])>1e-12}

    def cash(self):
        with self._db() as db: 
            return float(db.execute("SELECT value FROM metadata WHERE key='cash'").fetchone()[0])

    def orders(self):
        with self._db() as db:
            return [{**dict(r),'payload':json.loads(r['payload'])} for r in db.execute('SELECT * FROM orders ORDER BY rowid')]

    def fills(self):
        with self._db() as db:
            return [dict(r) for r in db.execute('SELECT * FROM fills ORDER BY rowid')]

    def _submit(self,db,order_id,payload):
        required={'instrument_id','quantity','reference_price','target_id','at'}
        if not required.issubset(payload) or not np.isfinite([payload['quantity'],payload['reference_price']]).all() or payload['quantity']==0 or payload['reference_price']<=0: 
            raise ValueError('Invalid simulated order')
        asof_timestamp(payload['at'])
        data=canonical(payload)
        old=db.execute('SELECT * FROM orders WHERE id=?',(order_id,)).fetchone()

        if old:
            if old['payload']!=data:
                raise ValueError('Idempotency collision: order payload changed')
            return dict(old)
        db.execute('INSERT INTO orders(id,payload,state) VALUES (?,?,?)',(order_id,data,'SUBMITTED'))
        db.execute('INSERT INTO events(order_id,event,at) VALUES (?,?,?)',(order_id,'SUBMITTED',payload['at']))
        return {'id':order_id,'payload':data,'state':'SUBMITTED','filled':0.}

    def submit(self,order_id,payload):
        with self.transaction() as db: return self._submit(db,order_id,payload)

    def cancel(self,order_id):
        with self.transaction() as db:
            row=db.execute('SELECT state FROM orders WHERE id=?',(order_id,)).fetchone()
            if row is None:
                raise KeyError(order_id)
            
            if row['state']=='CANCELLED':
                return
            transition(row['state'],'CANCELLED')
            db.execute("UPDATE orders SET state='CANCELLED' WHERE id=?",(order_id,))
            db.execute('INSERT INTO events(order_id,event,at) VALUES (?,?,?)',(order_id,'CANCELLED',pd_now()))

    def fill(self,order_id,*,fill_id,quantity,price,fee,at,limits,prices,stopped=False,commission=0.):

        #simulated fill + cash/position updates.

        at=asof_timestamp(at).isoformat()
        if not np.isfinite([quantity,price,fee,commission]).all() or quantity<=0 or price<=0 or fee<0 or commission<0 or commission>fee: 
            raise ValueError('Invalid fill')
        with self.transaction() as db:
            old=db.execute('SELECT * FROM fills WHERE id=?',(fill_id,)).fetchone()
            payload={'id':fill_id,'order_id':order_id,'quantity':quantity,'price':price,'fee':fee,'at':at,'commission':commission}
            if old:
                if dict(old)!=payload:
                    raise ValueError('Fill ID collision')
                return
            row=db.execute('SELECT * FROM orders WHERE id=?',(order_id,)).fetchone()
            if row is None: 
                raise KeyError(order_id)
            order=json.loads(row['payload'])
            total=abs(order['quantity'])
            remaining=total-row['filled']

            if quantity>remaining+1e-10:
                raise ValueError('Overfill')
            if asof_timestamp(at)<asof_timestamp(order['at']): 
                raise ValueError('Fill precedes order')
            
            signed=np.sign(order['quantity'])*quantity
            state='FILLED' if abs(quantity-remaining)<1e-10 else 'PARTIALLY_FILLED'; transition(row['state'],state)
            positions={r['asset']:r['quantity'] for r in db.execute('SELECT * FROM positions')}
            day=at[:10]

            turnover=db.execute('SELECT COALESCE(SUM(quantity*price),0) FROM fills WHERE substr(at,1,10)=?',(day,)).fetchone()[0]
            marks={**prices,order['instrument_id']:price}
            pretrade(positions,{'instrument_id':order['instrument_id'],'quantity':signed},marks,limits,daily_turnover=turnover,stopped=stopped)
            cash=float(db.execute("SELECT value FROM metadata WHERE key='cash'").fetchone()[0])-signed*price-fee
            if cash < -1e-8: 
                raise PermissionError('Insufficient paper cash including fees')
            
            db.execute("UPDATE metadata SET value=? WHERE key='cash'",(str(max(0,cash)),))
            asset=order['instrument_id']; position=positions.get(asset,0.)+signed
            db.execute('INSERT INTO positions VALUES (?,?) ON CONFLICT(asset) DO UPDATE SET quantity=excluded.quantity',(asset,position))
            db.execute('UPDATE orders SET filled=filled+?,state=? WHERE id=?',(quantity,state,order_id))
            db.execute('INSERT INTO fills VALUES (?,?,?,?,?,?,?)',(fill_id,order_id,quantity,price,fee,at,commission))
            db.execute('INSERT INTO events(order_id,event,at) VALUES (?,?,?)',(order_id,state,at))


class Trading212Paper(PaperBroker): venue='trading212'
class RevolutXPaper(PaperBroker): venue='revolut_x'
class IBKRPaper(PaperBroker): venue='ibkr'
class AlpacaPaper(PaperBroker): venue='alpaca_paper'


def pd_now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()


class PaperService:
    #one account/system service, targets are persisted atomically before fills

    def __init__(self,broker,limits,*,risk_monitor):
        self.broker=broker
        self.limits=limits
        self.risk=risk_monitor

    def reconcile(self,target,prices,*,at):
        target.validate_at(at)
        if target.system!=self.broker.system or target.currency!=self.broker.currency:
            raise ValueError('Cross-system/currency target')
        
        stopped=self.risk.stopped()
        if stopped:
            raise PermissionError('Persistent risk stop')
        
        with self.broker.transaction() as db:
            old=db.execute('SELECT * FROM targets WHERE id=?',(target.target_id,)).fetchone()
            if old:
                if old['hash']!=target.sha256:
                    raise ValueError('Target ID reused with changed payload')
                return [r['id'] for r in db.execute('SELECT id,payload FROM orders') if json.loads(r['payload'])['target_id']==target.target_id]
            
            previous=db.execute('SELECT effective_at FROM targets ORDER BY rowid DESC LIMIT 1').fetchone()
            if previous and asof_timestamp(target.effective_at)<=asof_timestamp(previous[0]):
                raise ValueError('Stale/out-of-order target')
            positions={r['asset']:r['quantity'] for r in db.execute('SELECT * FROM positions')}
            names=set(positions)|set(target.quantities)
            if not names.issubset(prices): 
                raise ValueError('Complete prices required for reconciliation')
            
            delta={a:target.quantities.get(a,0)-positions.get(a,0) for a in names}
            pending=sorted([(a,q) for a,q in delta.items() if abs(q)>1e-12],key=lambda item:(item[1]>0,item[0]))
            day=asof_timestamp(at).isoformat()[:10]
            turnover=db.execute('SELECT COALESCE(SUM(quantity*price),0) FROM fills WHERE substr(at,1,10)=?',(day,)).fetchone()[0]
            projected=dict(positions)
            ids=[]
            
            for asset,q in pending:
                pretrade(projected,{'instrument_id':asset,'quantity':q},prices,self.limits,daily_turnover=turnover,stopped=stopped)
                projected[asset]=projected.get(asset,0)+q
                turnover+=abs(q)*prices[asset]
            for r in db.execute("SELECT id FROM orders WHERE state IN ('SUBMITTED','PARTIALLY_FILLED')").fetchall():
                db.execute("UPDATE orders SET state='CANCELLED' WHERE id=?",(r['id'],))
                db.execute('INSERT INTO events(order_id,event,at) VALUES (?,?,?)',(r['id'],'CANCELLED',asof_timestamp(at).isoformat()))
            db.execute('INSERT INTO targets VALUES (?,?,?)',(target.target_id,target.sha256,target.effective_at))
            for asset,q in pending:
                payload={'instrument_id':asset,'quantity':float(q),'reference_price':float(prices[asset]),
                         'target_id':target.target_id,'at':asof_timestamp(at).isoformat()}
                order_id=fingerprint({'system':target.system,'target_id':target.target_id,'asset':asset,'quantity':float(q)})
                self.broker._submit(db,order_id,payload)
                ids.append(order_id)
            return ids

    def run_once(self,target_path,prices,*,at,cost_model,instruments,fill_fraction=1.):
        from foundation.costs import Trade
        if not 0<fill_fraction<=1:
            raise ValueError('Invalid simulated fill fraction')
        target=TargetPositions.load(target_path)
        metadata={}
        for asset in set(target.quantities)|set(self.broker.positions()):
            if asset not in instruments:
                raise ValueError('Missing instrument metadata')
            
            item=dict(instruments[asset])
            for field,expected in [('venue',self.broker.venue),('currency',self.broker.currency)]:
                if item.pop(field,expected)!=expected: 
                    raise ValueError('Instrument venue/currency differs from paper account')
            if item.get('asset_type','stock') not in {'stock','etf','crypto'} or item.get('multiplier',1.)!=1.:
                raise ValueError('Paper cash simulator does not model futures/margin; use a reviewed product simulator')
            
            metadata[asset]=item
        ids=self.reconcile(target,prices,at=at)
        by_id={o['id']:o for o in self.broker.orders()}
        for key in ids:
            order=by_id[key]
            if order['state'] not in {'SUBMITTED','PARTIALLY_FILLED'}:
                continue
            p=order['payload']; remaining=abs(p['quantity'])-order['filled']
            quantity=remaining*fill_fraction
            asset=p['instrument_id']

            trade=Trade(asset,self.broker.venue,'buy' if p['quantity']>0 else 'sell',quantity,prices[asset],self.broker.currency,asof_timestamp(at),**metadata[asset])
            prior=[f for f in self.broker.fills() if f['order_id']==key]
            cost=cost_model.quote_partial_fill(trade,prior_quantity=sum(f['quantity'] for f in prior),prior_notional=sum(f['quantity']*f['price'] for f in prior),commission_paid=sum(f['commission'] for f in prior))
            fee=cost.total
            self.broker.fill(key,fill_id=fingerprint({'order':key,'filled_before':order['filled'],'quantity':quantity}),quantity=quantity,
                             price=prices[asset],fee=fee,at=at,limits=self.limits,prices=prices,stopped=self.risk.stopped(),commission=cost.commission)
        return {'positions':self.broker.positions(),'cash':self.broker.cash(),'orders':self.broker.orders(),'simulated':True}


class SharedRateLimiter:
    #one cross process reservation file per venue

    def __init__(self,path,interval=1.,*,captures_coordinated=False):
        self.path=Path(path); self.interval=interval; self.coordinated=captures_coordinated
        if not np.isfinite(interval) or interval<1:
            raise ValueError('At least one second per venue request')

    def reserve(self,*,now=None,real_requests=False):
        if real_requests and not self.coordinated:
            raise PermissionError('BLOCKED_RATE_BUDGET: capture coordination unverified')
        now=time.time() if now is None else now
        if not np.isfinite(now):
            raise ValueError('Invalid clock')
        
        check_disk(self.path)
        with lock(self.path.with_suffix('.lock')):
            state=json.loads(self.path.read_text()) if self.path.exists() else {'next':now,'last_clock':now,'interval':self.interval}
            if state['interval']!=self.interval or now<state['last_clock']: 
                raise ValueError('Rate configuration changed or clock moved backwards')
            
            slot=max(now,state['next'])
            temp=self.path.with_suffix('.tmp')
            temp.write_text(canonical({'next':slot+self.interval,'last_clock':now,'interval':self.interval}))
            temp.replace(self.path)
        return slot


def live_gate(*,LIVE=False,cap_file=None):
    #audits necessary future gates

    reasons=[]
    if LIVE is not True: 
        reasons.append('LIVE flag absent')
    cap=None
    if cap_file is None:
        reasons.append('cap file absent')

    else:
        p=Path(cap_file)
        if p.is_symlink() or not p.is_file():
            reasons.append('cap file missing or symlink')
        elif os.access(p,os.W_OK) or os.access(p.parent,os.W_OK):
            reasons.append('cap is writable or replaceable by execution process')
        else:
            with p.open('r') as stream: 
                cap=json.load(stream)
            if not isinstance(cap.get('max_notional'),(int,float)) or not np.isfinite(cap['max_notional']) or cap['max_notional']<=0: 
                reasons.append('invalid notional cap')
    reasons.append('live transport not implemented; paper only')
    return {'enabled':False,'reasons':reasons,'cap':cap}

__all__ = ["AlpacaPaper", "BrokerAdapter", "IBKRPaper", "PaperBroker", "PaperService", "RevolutXPaper", "SharedRateLimiter", "Trading212Paper", "live_gate"]
