
#source observations, data revisions, trigger definitions

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import sqlite3
from types import MappingProxyType
import pandas as pd
from foundation.data import asof_timestamp
from foundation.store import canonical, check_disk, fingerprint, utc_now


def stamp(value):
    return asof_timestamp(value).isoformat()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class Event:
    source: str
    event_id: str
    kind: str
    instrument_ids: tuple[str, ...]
    event_at: str
    received_at: str
    payload: dict
    published_at: str | None = None
    provenance: dict | None = None

    def __post_init__(self):
        if not all(isinstance(x, str) and x.strip() for x in (self.source, self.event_id, self.kind)):
            raise ValueError('Source, source event identity and kind are required')
        if len(set(self.instrument_ids)) != len(self.instrument_ids) or any(not isinstance(i, str) or not i for i in self.instrument_ids):
            raise ValueError('Unique stable instrument identities required')
        
        for field in ('event_at', 'received_at', 'published_at'):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, stamp(value))
        object.__setattr__(self, 'instrument_ids', tuple(self.instrument_ids))
        if not isinstance(self.payload, (dict, MappingProxyType)):
            raise ValueError('Event payload must be a mapping')
        
        #JSON validation rejects NaN and arbitrary objects
        object.__setattr__(self, 'payload', _freeze(json.loads(canonical(plain(self.payload)))))
        object.__setattr__(self, 'provenance', _freeze(json.loads(canonical(plain(self.provenance or {})))))

    @property
    def available_at(self):
        return max(asof_timestamp(self.received_at), asof_timestamp(self.published_at or self.received_at))

    @property
    def content_hash(self):
        #repeat polling cannot change an observation into a revision

        fields = self.to_dict()
        fields.pop('received_at')
        fields.pop('provenance')
        return fingerprint(fields)

    def to_dict(self):
        return {k: plain(getattr(self, k)) for k in self.__dataclass_fields__}


class EventStore:

    #own SQLite store, not a capture, and rows cannot be rewritten

    def __init__(self, path):
        self.path = Path(path)
        check_disk(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS event_versions (
                    seq INTEGER PRIMARY KEY, source TEXT NOT NULL, event_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL, available_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, body TEXT NOT NULL,
                    UNIQUE(source,event_id,content_hash));
                CREATE INDEX IF NOT EXISTS events_available ON event_versions(available_at);
                CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON event_versions
                    BEGIN SELECT RAISE(ABORT,'Immutable event version'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON event_versions
                    BEGIN SELECT RAISE(ABORT,'Immutable event version'); END;
            ''')

    @contextmanager
    def transaction(self):
        check_disk(self.path)
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def append(self, event, *, recorded_at=None):
        recorded = asof_timestamp(recorded_at or utc_now())
        if event.available_at > recorded:
            raise ValueError('Cannot store a future receipt/publication')
        with self.transaction() as db:
            old = db.execute('SELECT body FROM event_versions WHERE source=? AND event_id=? AND content_hash=?',
                             (event.source, event.event_id, event.content_hash)).fetchone()
            if old:
                return Event(**json.loads(old['body'])), False
            prior = db.execute('SELECT available_at FROM event_versions WHERE source=? AND event_id=? ORDER BY seq DESC LIMIT 1', (event.source, event.event_id)).fetchone()
            if prior and event.available_at <= asof_timestamp(prior['available_at']):
                raise ValueError('Changed content needs a strictly later availability; preserve revisions')
            
            db.execute('INSERT INTO event_versions(source,event_id,content_hash,available_at,recorded_at,body) VALUES (?,?,?,?,?,?)',
                       (event.source, event.event_id, event.content_hash, stamp(event.available_at), stamp(recorded), canonical(event.to_dict())))
        return event, True

    def timeline(self, as_of, *, known_at=None):
        cutoff = asof_timestamp(as_of)
        with sqlite3.connect(self.path) as db:
            rows = db.execute('SELECT recorded_at,body FROM event_versions ORDER BY seq').fetchall()
        result = [Event(**json.loads(body)) for recorded, body in rows
                  if known_at is None or asof_timestamp(recorded) <= asof_timestamp(known_at)]
        return sorted((e for e in result if e.available_at <= cutoff), key=lambda e: (e.available_at, e.source, e.event_id))

    def as_of(self, at, *, known_at=None):
        latest = {}
        for event in self.timeline(at, known_at=known_at):
            latest[event.source, event.event_id] = event
        return list(latest.values())


@dataclass(frozen=True)
class TriggerSpec:
    name: str
    source: str
    hypothesis: str
    availability_rule: str
    latency_budget_seconds: float | None
    kinds: tuple[str, ...]
    latency_origin: str = 'published_at'
    version: int = 1
    latency_policy: str = 'elapsed_seconds'

    def __post_init__(self):
        if not re.fullmatch(r'[a-z][a-z0-9_]*', self.name) or not self.source or not self.hypothesis.strip():
            raise ValueError('Named trigger, source and economic hypothesis required')
        if self.availability_rule != 'max(published_at, received_at)':
            raise ValueError('Clock rule must preserve publication and receipt')
        if self.latency_origin not in {'published_at', 'available_at', 'event_at'}:
            raise ValueError('Explicit latency origin required')
        if self.latency_policy == 'elapsed_seconds':
            if self.latency_budget_seconds is None or not 0 < self.latency_budget_seconds <= 86400:
                raise ValueError('Finite latency budget up to one day required')
        elif self.latency_policy == 'next_xlon_open':
            if self.latency_budget_seconds is not None or self.latency_origin != 'published_at':
                raise ValueError('Next XLON open requires publication origin and no fixed seconds budget')
            
        else:
            raise ValueError('Unknown latency policy')
        if not self.kinds or self.version < 1:
            raise ValueError('Kinds and protocol version required')
        object.__setattr__(self, 'kinds', tuple(self.kinds))

    @property
    def sha256(self):
        fields = asdict(self)
        #preserve identifies of time protocols and outboxes

        if self.latency_policy == 'elapsed_seconds':
            fields.pop('latency_policy')
        return fingerprint(fields)

    def origin(self, event):
        if self.latency_policy == 'next_xlon_open' and event.published_at is None:
            raise ValueError('Next XLON open requires a documented publication timestamp')
        return asof_timestamp(getattr(event, self.latency_origin) or event.received_at)

    def deadline(self, event):
        origin = self.origin(event)
        if self.latency_policy == 'next_xlon_open':
            from foundation.data.calendar import next_open

            #LSE is 'pandas_market_calendars' london stock exchange (XLON)
            return next_open(origin, exchange='LSE')
        return origin + pd.Timedelta(seconds=self.latency_budget_seconds)


@dataclass(frozen=True)
class Firing:
    spec: TriggerSpec
    event: Event
    decision_at: str
    latency_seconds: float

    @property
    def key(self):
        return fingerprint({'trigger': self.spec.sha256, 'source': self.event.source, 'id': self.event.event_id, 'version': self.event.content_hash})

    def to_dict(self):
        return {'spec': asdict(self.spec), 'event': self.event.to_dict(),
                'decision_at': self.decision_at, 'latency_seconds': self.latency_seconds, 'key': self.key}


class TriggerRegistry:
    def __init__(self):
        self._entries = {}

    def register(self, spec, predicate):
        if spec.name in self._entries or not callable(predicate):
            raise ValueError('Unique trigger and callable predicate required')
        
        self._entries[spec.name] = (spec, predicate)

    def describe(self):
        return [{**asdict(spec), 'signal_kind': 'activated'} for spec, _ in self._entries.values()]

    def get(self, name):
        return self._entries[name][0]

    def fire(self, event, at, *, names=None):
        now = asof_timestamp(at)
        if event.available_at > now:
            return []
        fired = []
        for name in sorted(self._entries if names is None else names):
            spec, predicate = self._entries[name]
            if event.source != spec.source or event.kind not in spec.kinds:
                continue
            age = (now - spec.origin(event)).total_seconds()
            if age < 0 or now > spec.deadline(event):
                continue
            if predicate(event, now):
                fired.append(Firing(spec, event, stamp(now), age))
                
        return fired
