
#official-ibapi headline watcher.

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import threading
import time
import tomllib
import pandas as pd
from foundation.data import asof_timestamp
from foundation.store import file_hash, lock, platform_home
from .core import Event, stamp


class CaptureGuard:
    #conservative quiet hours plus current configs, busy-lock checks

    def __init__(self, *, uk_config, options_config, uk_lock, options_lock, clock=None):
        self.uk_config, self.options_config = Path(uk_config), Path(options_config)
        uk = json.loads(self.uk_config.read_text())
        self.protocol = Path(uk['runtime_root'])/'research'/uk['protocol_file']
        self.uk_activation = Path(uk['runtime_root'])/'data/state/activation.json'

        protocol = json.loads(self.protocol.read_text())
        options = tomllib.loads(self.options_config.read_text())
        if not uk.get('readonly') or options.get('provider') != 'ibkr':
            raise ValueError('Expected read-only UK and IBKR options configuration')
        if Path(options['uk_config']).resolve() != self.uk_config.resolve():
            raise ValueError('Capture configurations disagree')
        
        for value in protocol['schedule']['slots']:
            h, m = map(int, value.split(':')); minutes=h*60+m
            if not 8*60 <= minutes or minutes+25 > 17*60:
                raise ValueError('UK schedule exceeds conservative blackout; review guard')
            
        for spec in options['slots'].values():
            h,m=map(int,spec['new_york_time'].split(':')); minutes=h*60+m
            if minutes < 9*60 or minutes+spec.get('window_minutes',0)>16*60+30:
                raise ValueError('Options schedule exceeds conservative blackout; review guard')
            
        self.hashes = {str(p):file_hash(p) for p in (self.uk_config,self.options_config,self.protocol)}
        self.locks = (Path(uk_lock), Path(options_lock))
        self.reserved = {0, int(uk['client_id']),27201,27301}
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def check(self, *, duration=0):
        if not 0 <= duration <= 60: 
            raise ValueError('Bounded quiet-window horizon required')
        for path,digest in self.hashes.items():
            if file_hash(path)!=digest: 
                raise PermissionError('Capture configuration changed; rebuild reviewed guard')
        if self.uk_activation.exists():
            raise PermissionError('Activated UK daily capture requires a separately reviewed quiet window')
        
        now=asof_timestamp(self.clock()); end=now+pd.Timedelta(seconds=duration+30)
        for zone,start,stop in [('Europe/London','07:40','17:00'),('America/New_York','09:00','16:30')]:
            local=now.tz_convert(zone); last=end.tz_convert(zone)
            for day in pd.date_range(local.normalize(), last.normalize(),freq='D'):
                if day.weekday()>=5: 
                    continue
                #construct wall clocks
                begin=pd.Timestamp(f'{day.date()} {start}',tz=zone)
                finish=pd.Timestamp(f'{day.date()} {stop}',tz=zone)
                if now<finish and end>=begin: 
                    raise PermissionError('UK/options capture blackout')
        for path in self.locks:
            if not path.is_file():
                raise PermissionError('Capture lock absent; coordination is unconfirmed')
            with path.open('rb') as stream:
                try:
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: 
                    raise PermissionError('Capture is busy') from None
                finally: fcntl.flock(stream,fcntl.LOCK_UN)


class NewsWirePolicy:
    #enforce tiny allowlist for legacy, protobuf encoders

    def __init__(self, guard): 
        self.guard,self.expected=guard,None

    def check(self, message_id):
        self.guard.check()
        if int(message_id)!=71 and int(message_id)!=self.expected:
            raise PermissionError('Only explicitly scoped news requests and START_API are allowed')
        
    @contextmanager
    def request(self, message_id):
        if message_id not in {1,2,85} or self.expected is not None:
            raise PermissionError('Unsupported or overlapping news request')
        
        self.expected=message_id
        try: 
            yield
        finally:
            self.expected=None


def official_client(on_news, guard):
    #lazy official SDK import

    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.common import PROTOBUF_MSG_ID
    class Client(EWrapper,EClient):
        def __init__(self):
            EWrapper.__init__(self)
            EClient.__init__(self,self)

            self.policy=NewsWirePolicy(guard)
            self.ready=threading.Event()
            self.failures=[]
            self.providers=[]; self.providers_ready=threading.Event()

        def sendMsg(self,msgId,msg):
            self.policy.check(msgId)
            return super().sendMsg(msgId,msg)
        
        def sendMsgProtoBuf(self,msgId,msg):
            self.policy.check(msgId-PROTOBUF_MSG_ID)
            return super().sendMsgProtoBuf(msgId,msg)
        
        def placeOrder(self,*args,**kwargs):
            raise PermissionError('News reader cannot place orders')
        def cancelOrder(self,*args,**kwargs):
            raise PermissionError('News reader cannot cancel orders')
        def reqGlobalCancel(self,*args,**kwargs):
            raise PermissionError('News reader cannot cancel orders')
        def exerciseOptions(self,*args,**kwargs):
            raise PermissionError('News reader cannot exercise options')
        
        def nextValidId(self,orderId):
            self.ready.set()  #connection readiness only
        def managedAccounts(self,accountsList):
            pass  # discard unsolicitied data

        def newsProviders(self,newsProviders):
            self.providers=[{'code':p.code,'name':p.name} for p in newsProviders]
            self.providers_ready.set()

        def tickNews(self,tickerId,timeStamp,providerCode,articleId,headline,extraData):

            on_news(tickerId,timeStamp,providerCode,articleId,headline,extraData)
        def error(self,*args):

            code=args[2] if len(args)>=4 else (args[1] if len(args)>=2 else None)
            if code not in {2104,2106,2107,2108,2158}: self.failures.append(code)

    return Client()


def news_providers(*, guard, readonly_confirmed=False, client_id=27401, host='127.0.0.1', port=4001, timeout=10,
                   client_factory=official_client, session_lock=None):
    """One bounded reqNewsProviders call; does not subscribe or request headlines.

    The callback establishes the account's API provider list, not RNS coverage
    or non-display rights. No credentials/account identifiers are retained.
    """
    if not readonly_confirmed or guard is None:
        raise PermissionError('Read-only confirmation and capture coordination required')
    if host not in {'127.0.0.1','localhost'} or not 1<=port<=65535 or not 0<timeout<=20:
        raise ValueError('Bounded loopback request required')
    if client_id<=0 or client_id in {27101,27201,27301} | guard.reserved:
        raise ValueError('Dedicated news client ID required')
    
    with lock(Path(session_lock or platform_home()/'events/ibkr_news_session.lock')):
        guard.check(duration=timeout+5)
        client=client_factory(lambda *args:None,guard)
        stopped=threading.Event()
        errors=[]
        def connect():
            try:
                client.connect(host,port,client_id)
                if stopped.is_set(): 
                    client.disconnect()
                    return
                client.run()
            except Exception:
                errors.append('IBKR connection/reader failed')
        thread=threading.Thread(target=connect,daemon=True)
        deadline=time.monotonic()+timeout
        def wait_for(callback):
            while not callback.wait(.05):
                guard.check()
                if errors:
                    raise ConnectionError(errors[0])
                if client.failures: 
                    raise RuntimeError(f'IBKR provider query failed: codes {client.failures}')
                if time.monotonic()>=deadline: 
                    raise TimeoutError('IBKR provider callback incomplete')
                    
            guard.check()
            if client.failures:
                raise RuntimeError(f'IBKR provider query failed: codes {client.failures}')
        try:
            thread.start(); wait_for(client.ready)
            with client.policy.request(85): client.reqNewsProviders()
            wait_for(client.providers_ready)
            return {'received_at':stamp(datetime.now(timezone.utc)), 'client_id':client_id,
                    'providers':client.providers,'server_version':client.serverVersion(),
                    'transport':'official ibapi','request':'reqNewsProviders'}
        finally:
            stopped.set(); client.disconnect(); thread.join(timeout=2)


class IBKRNewsWatcher:
    #broad tape news only, with explicit stable identities supplied downstream

    def __init__(self, *, provider, instrument_ids=(), fixture=None, guard=None,
                 client_id=27401, port=4001, host='127.0.0.1', seconds=5,
                 network_enabled=False, readonly_confirmed=False, entitlement_confirmed=False,
                 client_factory=official_client, session_lock=None):
        
        if not re.fullmatch(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', provider or '') or host not in {'127.0.0.1','localhost'}:
            raise ValueError('Explicit news provider and loopback Gateway required')
        if not 0<seconds<=30 or client_id<=0 or not 1<=port<=65535:
            raise ValueError('Bounded duration, dedicated positive client ID and valid port required')
        if client_id in {27101,27201,27301} or (guard and client_id in guard.reserved):
            raise ValueError('News client ID overlaps capture/diagnostic client')
        
        self.provider,self.ids,self.fixture=provider,tuple(instrument_ids),fixture
        self.guard,self.client_id,self.port,self.host=guard,client_id,port,host
        self.seconds,self.enabled,self.readonly,self.entitled=seconds,network_enabled,readonly_confirmed,entitlement_confirmed
        self.factory=client_factory
        self.session_lock=Path(session_lock or platform_home()/'events/ibkr_news_session.lock')

    def _event(self,row):
        published=pd.to_datetime(row['timestamp_ms'],unit='ms',utc=True)
        if published<pd.Timestamp('2000-01-01',tz='UTC') or published>asof_timestamp(row['received_at'])+pd.Timedelta('5min'):
            raise ValueError('IBKR epoch unit or clock is invalid; expected milliseconds')
        return Event('ibkr_news',f"{row['provider']}:{row['article_id']}",'headline',tuple(row.get('instrument_ids',self.ids)),
                     stamp(published),row['received_at'],{'provider':row['provider'],'article_id':row['article_id'],
                     'headline':row['headline'],'extra_data':row.get('extra_data','')},stamp(published),
                     {'transport':'official ibapi tickNews','coverage':'provider-specific; no RNS coverage assumption'})

    def poll(self,at):
        if self.fixture is not None:
            return [e for e in map(self._event,self.fixture) if e.available_at<=asof_timestamp(at)]
        if not all((self.enabled,self.readonly,self.entitled,self.guard is not None)):
            raise PermissionError('Network off: read-only Gateway, entitlement and capture coordination required')
        
        with lock(self.session_lock):
            self.guard.check(duration=self.seconds+10)
            events=[]
            def callback(request_id,published,provider,article_id,headline,extra):
                events.append(self._event({'timestamp_ms':published,'provider':provider,'article_id':article_id,
                                          'headline':headline,'extra_data':extra,'received_at':stamp(datetime.now(timezone.utc))}))
            client=self.factory(callback,self.guard)
            thread=None; stopped=threading.Event()
            try:
                stopped=threading.Event(); errors=[]
                def connect_and_read():
                    try:
                        client.connect(self.host,self.port,self.client_id)
                        if stopped.is_set(): 
                            client.disconnect()
                            return
                        client.run()
                    except Exception:
                        errors.append('IBKR connection/reader failed')

                thread=threading.Thread(target=connect_and_read,daemon=True); thread.start()
                handshake_deadline=time.monotonic()+5
                while not client.ready.wait(timeout=.05):
                    self.guard.check(duration=self.seconds)
                    if errors: 
                        raise ConnectionError(errors[0])
                    if time.monotonic()>=handshake_deadline: 
                        raise TimeoutError('IBKR news handshake incomplete')
                    
                self.guard.check(duration=self.seconds)
                from ibapi.contract import Contract

                contract=Contract()
                contract.symbol=f'{self.provider}:{self.provider}_ALL'
                contract.secType='NEWS'
                contract.exchange=self.provider

                with client.policy.request(1):
                    client.reqMktData(9001,contract,'mdoff,292',False,False,[])
                deadline=time.monotonic()+self.seconds
                while time.monotonic()<deadline:
                    self.guard.check()
                    if errors:
                        raise ConnectionError(errors[0])
                    if client.failures:
                        raise RuntimeError(f'IBKR news failed: codes {client.failures}')
                    if not client.isConnected(): 
                        raise ConnectionError('IBKR news disconnected')
                    
                    time.sleep(min(.1,max(0,deadline-time.monotonic())))
                with client.policy.request(2):
                    client.cancelMktData(9001)
            finally:
                stopped.set()
                client.disconnect()
                if thread: thread.join(timeout=2)
            return events
