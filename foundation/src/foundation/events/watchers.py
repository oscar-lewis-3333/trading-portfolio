
#paper event ingestion, alongside read-only SEC transport

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from foundation.data import asof_timestamp
from foundation.store import atomic_json, file_hash, lock, platform_home
from .core import Event, stamp


class Watcher(Protocol):
    def poll(self, at) -> list[Event]: ...


class PaperWatcher:
    #replay a documented observation
    def __init__(self, events):
        self.events = tuple(events)

    def poll(self, at):
        return [event for event in self.events if event.available_at <= asof_timestamp(at)]


def _xml(raw):
    if len(raw) > 2_000_000 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('Oversized XML or unsupported entity declarations')
    
    return ET.fromstring(raw)


def parse_edgar_atom(raw, *, received_at, issuer_ids=None):
    #latest-filings. owner ids not guessed to be issuer ids
    
    root = _xml(raw)
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    result = []
    for entry in root.findall('a:entry', ns):
        category = entry.find('a:category', ns)
        form = category.get('term', '') if category is not None else ''
        if form not in {'4', '4/A', '8-K', '8-K/A'}:
            continue
        
        identity = entry.findtext('a:id', namespaces=ns)
        published = entry.findtext('a:updated', namespaces=ns)
        if not identity or not published:
            raise ValueError('EDGAR entry lacks stable identity or publication clock')
        
        link = entry.find('a:link', ns)
        payload = {'form': form, 'title': entry.findtext('a:title', default='', namespaces=ns),
                   'url': '' if link is None else link.get('href', ''), 'classification': 'filing_metadata_only'}
        result.append(Event('sec_edgar', identity, 'form4' if form.startswith('4') else '8k',
                            tuple((issuer_ids or {}).get(identity, ())), published, received_at, payload, published,
                            {'transport': 'EDGAR Atom', 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                             'issuer_mapping': 'explicit accession mapping; otherwise unresolved'}))
    return result


def parse_edgar_submissions(payload, *, received_at, issuer_ids):
    cik = str(payload['cik']).zfill(10)
    rows = payload['filings']['recent']
    required = ['accessionNumber', 'form', 'acceptanceDateTime', 'primaryDocument']
    if len({len(rows[k]) for k in required}) != 1:
        raise ValueError('Unaligned EDGAR recent-filing arrays')
    
    result = []
    for accession, form, published, document in zip(*(rows[k] for k in required)):
        if form not in {'4', '4/A', '8-K', '8-K/A'}:
            continue
        result.append(Event('sec_edgar', accession, 'form4' if form.startswith('4') else '8k',
                            tuple(issuer_ids.get(cik, ())), published, received_at,
                            {'form': form, 'cik': cik, 'primary_document': document, 'classification': 'filing_metadata_only'}, published,
                            {'transport': 'SEC submissions JSON', 'issuer_mapping': 'caller-audited issuer CIK mapping'}))
    return result


def parse_form4_document(raw):
    #distinguish purchases from other code

    root = _xml(raw)
    issuer = root.findtext('.//issuerCik')
    if not issuer or not issuer.isdigit() or len(issuer)>10:
        raise ValueError('Form 4 issuer CIK missing or invalid')
    transactions = []
    for row in root.findall('.//nonDerivativeTransaction'):
        transactions.append({'code': row.findtext('.//transactionCode'),
                             'acquired_disposed': row.findtext('.//transactionAcquiredDisposedCode/value'),
                             'shares': row.findtext('.//transactionShares/value'),
                             'price': row.findtext('.//transactionPricePerShare/value'),
                             'transaction_date': row.findtext('.//transactionDate/value')})
    return {'issuer_cik': str(issuer).zfill(10), 'transactions': transactions,
            'reported_purchase': any(t['code'] == 'P' and t['acquired_disposed'] == 'A' for t in transactions),
            'raw_sha256': hashlib.sha256(raw).hexdigest()}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise PermissionError('SEC transport refuses redirects; review a changed endpoint')


class SECClient:
    #one request per second across local clients

    def __init__(self, *, user_agent, state_path=None, network_enabled=False,
                 shared_budget_confirmed=False, timeout=15, opener=None, clock=time.time, sleeper=time.sleep):
        if not re.search(r'[^\s@]+@[^\s@]+\.[^\s@]+', user_agent or '') or '\n' in user_agent or '\r' in user_agent:
            raise ValueError('Declare a descriptive SEC User-Agent with a real contact email')
        if not 0 < timeout <= 30:
            raise ValueError('SEC timeout must be in (0,30] seconds')
        self.user_agent, self.enabled, self.coordinated = user_agent, network_enabled, shared_budget_confirmed
        self.state = Path(state_path or platform_home()/'events/sec_http.json')
        self.timeout, self.clock, self.sleep = timeout, clock, sleeper
        self.opener = opener or build_opener(_NoRedirect()).open

    def get(self, url):
        if not self.enabled or not self.coordinated:
            raise PermissionError('SEC network disabled or shared user-wide budget not confirmed')
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in {'www.sec.gov', 'data.sec.gov'} or parsed.username or parsed.port not in {None, 443}:
            raise ValueError('Only official HTTPS SEC endpoints are supported')
        with lock(self.state.with_suffix('.reservation.lock')):
            now = self.clock()
            state = json.loads(self.state.read_text()) if self.state.exists() else {}
            if now < state.get('last_clock', now):
                raise ValueError('SEC clock moved backwards')
            if now < state.get('cooldown_until', 0):
                raise PermissionError('SEC cooldown active; no retry sent')
            slot = max(now, state.get('next', now))
            if slot-now > 30:
                raise PermissionError('SEC request queue saturated; defer this poll')
            self.sleep(max(0., slot-self.clock()))
            sent = self.clock()
            if sent < slot:
                raise ValueError('SEC sleeper returned before reserved slot')
            atomic_json(self.state, {**state, 'next': sent+1., 'last_clock': sent}, overwrite=True)
        request = Request(url, headers={'User-Agent': self.user_agent, 'Accept-Encoding': 'identity'})
        try:
            with self.opener(request, timeout=self.timeout) as response:
                raw = response.read(10_000_001)
                if len(raw) > 10_000_000:
                    raise ValueError('SEC response exceeds bounded reader')
                return raw
        except HTTPError as error:
            if error.code in {403, 429}:
                wait = 600.
                retry = error.headers.get('Retry-After') if error.headers else None
                if retry:
                    try: wait = max(wait, float(retry))
                    except ValueError:
                        try: wait = max(wait, parsedate_to_datetime(retry).timestamp()-self.clock())
                        except (ValueError, TypeError): pass
                with lock(self.state.with_suffix('.reservation.lock')):
                    state = json.loads(self.state.read_text())
                    atomic_json(self.state, {**state, 'cooldown_until': self.clock()+wait, 'http_status': error.code}, overwrite=True)
            raise


class EDGARWatcher:
    def __init__(self, client, *, form, issuer_ids=None, fixture=None):
        if form not in {'4', '8-K'}:
            raise ValueError('Supported EDGAR forms are 4 and 8-K')
        self.client, self.form, self.issuer_ids, self.fixture = client, form, issuer_ids or {}, fixture

    def poll(self, at):
        if self.fixture is not None:
            raw = self.fixture
            received = at
        else:
            query = urlencode({'action': 'getcurrent', 'type': self.form, 'owner': 'only' if self.form == '4' else 'exclude', 'count': 100, 'output': 'atom'})
            raw = self.client.get('https://www.sec.gov/cgi-bin/browse-edgar?'+query)
            received = datetime.now(timezone.utc)

        return [e for e in parse_edgar_atom(raw, received_at=received, issuer_ids=self.issuer_ids)
                if e.payload['form'].removesuffix('/A') == self.form]


class NSMRNSReader:
    #local authorised CSV/JSONL exports, or things recieved from licensed feeds
    def __init__(self, path, *, source, mapping, expected_sha256=None):
        if source not in {'fca_nsm', 'uk_rns'}:
            raise ValueError('Choose explicit NSM or licensed RNS provenance')
        self.path, self.source, self.mapping, self.sha = Path(path), source, dict(mapping), expected_sha256

    def poll(self, at):
        sha = file_hash(self.path)
        if self.sha is not None and sha != self.sha:
            raise ValueError('News export changed from frozen hash')
        
        frame = pd.read_json(self.path, lines=True) if self.path.suffix == '.jsonl' else pd.read_csv(self.path, dtype=str, keep_default_na=False)
        frame = frame.rename(columns=self.mapping)
        required = {'event_id', 'published_at', 'received_at', 'headline', 'instrument_id'}
        if not required.issubset(frame):
            raise ValueError('Explicit source publication and actual receipt clocks required')
        
        events = []
        for row in frame.to_dict('records'):
            event = Event(self.source, row['event_id'], row.get('kind') or 'announcement',
                          (row['instrument_id'],) if row['instrument_id'] else (), row['published_at'], row['received_at'],
                          {k: v for k, v in row.items() if k not in {'received_at', 'published_at', 'event_id', 'instrument_id'}},
                          row['published_at'], {'file_sha256': sha, 'transport': 'authorised local export; not a live-site scrape'})
            if event.available_at <= asof_timestamp(at): events.append(event)
        if file_hash(self.path) != sha:
            raise ValueError('News export changed while reading')
        return events


class PriceMonitor:
    #completed bars/quotes from existing exports only

    def __init__(self, observations, *, move_threshold=.02, vol_threshold=2., window=20, source='price_monitor'):
        if not 0 < move_threshold < 1 or vol_threshold <= 0 or window < 3:
            raise ValueError('Invalid monitor thresholds/window')
        self.frame = observations.copy()
        self.move, self.vol, self.window, self.source = move_threshold, vol_threshold, window, source

    def poll(self, at):
        required = {'instrument_id', 'event_at', 'available_at', 'price'}
        if not required.issubset(self.frame):
            raise ValueError('Timestamped completed price observations required')
        
        f = self.frame.copy()
        for key in ['event_at', 'available_at']:
            f[key] = f[key].map(asof_timestamp)
        f = f[(f.available_at <= asof_timestamp(at)) & (f.event_at <= asof_timestamp(at))]
        if (f.available_at < f.event_at).any() or f.duplicated(['instrument_id', 'event_at']).any():
            raise ValueError('Ambiguous bar versions or availability before observation')
        if 'completed' in f and not f.completed.eq(True).all():
            raise ValueError('Incomplete bar supplied')
        if not np.isfinite(f.price).all() or (f.price <= 0).any():
            raise ValueError('Positive finite price required')
        
        result = []
        for instrument, rows in f.groupby('instrument_id'):
            rows = rows.sort_values('event_at')
            for i, row in enumerate(rows.itertuples()):
                prior = rows.iloc[:i]
                prior = prior[prior.available_at <= row.available_at]
                reference = getattr(row, 'reference_price', None)
                reference_at = getattr(row, 'reference_available_at', None)
                if reference is not None and not pd.isna(reference):
                    if reference_at is None or asof_timestamp(reference_at) > row.event_at or not np.isfinite(reference) or reference <= 0:
                        raise ValueError('Gap reference must be positive and known at observation')
                    move = row.price/reference-1
                    if abs(move) >= self.move:
                        result.append(self._event(row, 'price_gap', {'move': float(move), 'reference_price': float(reference)}))
                if len(prior) >= self.window:
                    history = prior.tail(self.window).price.to_numpy(float)
                    changes = np.diff(np.log(history))
                    sigma = float(np.std(changes, ddof=1))
                    shock = abs(math.log(row.price/history[-1]))/sigma if sigma > 0 else None
                    if shock is not None and shock >= self.vol:
                        result.append(self._event(row, 'volatility_shock', {'z_abs': shock, 'past_volatility': sigma, 'window': self.window}))
                    if row.price > max(history):
                        result.append(self._event(row, 'range_breakout', {'past_high': float(max(history)), 'window': self.window}))
        return result

    def _event(self, row, kind, payload):
        return Event(self.source, f'{row.instrument_id}:{stamp(row.event_at)}:{kind}', kind, (row.instrument_id,),
                     stamp(row.event_at), stamp(row.available_at), payload, stamp(row.event_at),
                     {'basis': 'existing observations; past-only reference', 'price': float(row.price)})


class MacroCalendar:
    #schedule chosen records known before release
    def __init__(self, releases): self.releases = tuple(releases)

    def poll(self, at):
        result = []
        for row in self.releases:
            event = Event('macro_calendar', row['release_id'], 'scheduled_release', tuple(row.get('instrument_ids', ())),
                          row['scheduled_at'], row['received_at'],
                          {'name': row['name'], 'scheduled_at': stamp(row['scheduled_at']), 'country': row['country']},
                          row['known_at'], {'basis': 'calendar vintage; no future released value'})
            if event.available_at <= asof_timestamp(at): result.append(event)
        return result


class CryptoMetricsMonitor:
    #fundings observations remain information
    def __init__(self, observations, *, funding_threshold=.001, vol_ratio_threshold=2.):
        if funding_threshold <= 0 or vol_ratio_threshold <= 0:
            raise ValueError('Positive crypto monitor thresholds required')
        self.rows = tuple(observations)
        self.funding, self.vol = funding_threshold, vol_ratio_threshold

    def poll(self, at):
        result = []
        for row in self.rows:
            observed, available = map(asof_timestamp, (row['event_at'], row['available_at']))
            if available < observed: raise ValueError('Crypto availability precedes observation')
            if available > asof_timestamp(at): continue
            values = []
            funding = row.get('funding_rate')
            if funding is not None:
                if not math.isfinite(funding) or not math.isfinite(row['funding_period_hours']) or row['funding_period_hours'] <= 0:
                    raise ValueError('Finite funding and positive period required')
                if abs(funding) >= self.funding: values.append(('funding_extreme', {'funding_rate': funding, 'funding_period_hours': row['funding_period_hours']}))
            volatility = row.get('realised_vol')
            if volatility is not None:
                reference = row['reference_vol']
                if not np.isfinite([volatility, reference]).all() or min(volatility, reference) <= 0 or asof_timestamp(row['reference_available_at']) > observed:
                    raise ValueError('Positive past-only crypto volatility reference required')
                if volatility/reference >= self.vol: values.append(('crypto_volatility', {'ratio': volatility/reference}))
            for kind, payload in values:
                result.append(Event('crypto_metrics', f"{row['instrument_id']}:{stamp(observed)}:{kind}", kind,
                                    (row['instrument_id'],), stamp(observed), stamp(available), payload, stamp(observed),
                                    {'venue': row['venue'], 'basis': 'existing exported observations; no extra polling'}))
        return result
