
#paper decision outbox, i.e, we get the contract to be ordered, but don't order

from dataclasses import dataclass
import json
import time
from pathlib import Path
import pandas as pd
from foundation.contracts import TargetPositions
from foundation.data import asof_timestamp
from foundation.store import atomic_json, canonical, fingerprint, lock
from .core import Event, Firing, TriggerSpec, stamp


@dataclass(frozen=True)
class DecisionContext:
    run_id: str
    system: str
    decision_at: str
    firings: tuple[Firing, ...]
    protocol: dict


@dataclass(frozen=True)
class Decision:
    #account quantities, including scheduled sleeves

    quantities: dict[str, float]
    valid_for_seconds: float = 300


class EventOrchestrator:
    def __init__(self, store, registry, output_dir, *, system, currency, protocol, paper=True, timer=time.monotonic):
        if paper is not True or not system or not currency or not protocol:
            raise ValueError('Explicit paper system, currency and frozen protocol required')
        
        self.store, self.registry, self.timer = store, registry, timer
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.system, self.currency = system, currency
        self.protocol = json.loads(canonical(protocol))
        self.protocol_hash = fingerprint(self.protocol)
        atomic_json(self.root/'protocol.json', self.protocol)
        with store.transaction() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS event_system_bindings (system TEXT PRIMARY KEY, binding TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decision_runs (
                    run_id TEXT PRIMARY KEY, system TEXT NOT NULL, at TEXT NOT NULL,
                    inputs TEXT NOT NULL, status TEXT NOT NULL, target TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS delivered_triggers (
                    system TEXT NOT NULL, protocol_hash TEXT NOT NULL,
                    firing_key TEXT NOT NULL, run_id TEXT NOT NULL,
                    PRIMARY KEY(system,protocol_hash,firing_key));
            ''')
            binding = canonical({'output': str(self.root.resolve()), 'currency': currency, 'protocol': self.protocol_hash})
            old = db.execute('SELECT binding FROM event_system_bindings WHERE system=?', (system,)).fetchone()
            if old and old[0] != binding:
                raise ValueError('System already bound to another output/currency/protocol; migrate explicitly')
            
            db.execute('INSERT OR IGNORE INTO event_system_bindings VALUES (?,?)', (system, binding))

    def poll(self, watcher, handler, *, at):
        events = list(watcher.poll(at))
        finished = max([asof_timestamp(at), *(asof_timestamp(e.received_at) for e in events)])
        for event in events:
            self.store.append(event, recorded_at=finished)
        return self.run(handler, at=finished)

    def _publish(self, run, now):
        target = TargetPositions(**json.loads(run['target']))
        firings = json.loads(run['inputs'])['firings']
        expired_signal = any(now > TriggerSpec(**f['spec']).deadline(Event(**f['event'])) for f in firings)
        if asof_timestamp(target.expires_at) <= now or expired_signal:
            status = 'expired_before_publish'
        else:
            target.publish(self.root/'targets'/f'{target.target_id}.json')
            current = self.root/'current_target.json'
            previous = TargetPositions.load(current) if current.exists() else None
            if previous and asof_timestamp(previous.effective_at) > asof_timestamp(target.effective_at):
                status = 'superseded'
            elif previous and previous.effective_at == target.effective_at and previous.target_id != target.target_id:
                raise ValueError('Two account targets at one instant: batch simultaneous signals')
            
            else:
                atomic_json(current, target.to_dict(), overwrite=True)
                status = 'published'
        with self.store.transaction() as db:
            db.execute('UPDATE decision_runs SET status=? WHERE run_id=?', (status, run['run_id']))
        return {'run_id': run['run_id'], 'status': status, 'paper': True}

    def run(self, handler, *, at):
        #handlers are deterministic, read-only, and use this context

        now = asof_timestamp(at)
        with lock(self.root/'orchestrator.process.lock'):
            with self.store.transaction() as db:
                pending = db.execute("SELECT * FROM decision_runs WHERE system=? AND status IN ('running','ready') ORDER BY at", (self.system,)).fetchall()
            results = []
            if pending:
                for run in pending:
                    results.append(self._execute(run, handler, now))
                return results
            firings = [f for event in self.store.as_of(now, known_at=now) for f in self.registry.fire(event, now)]
            with self.store.transaction() as db:
                seen = {r[0] for r in db.execute('SELECT firing_key FROM delivered_triggers WHERE system=? AND protocol_hash=?',
                                                (self.system, self.protocol_hash))}
                fresh = sorted((f for f in firings if f.key not in seen), key=lambda f: f.key)
                if not fresh:
                    return []
                
                run_id = fingerprint({'system': self.system, 'protocol': self.protocol_hash, 'firings': [f.key for f in fresh]})
                inputs = {'protocol': self.protocol, 'firings': [f.to_dict() for f in fresh]}
                db.execute('INSERT INTO decision_runs VALUES (?,?,?,?,?,?,?)',
                           (run_id, self.system, stamp(now), canonical(inputs), 'running', None, None))
                for f in fresh:
                    db.execute('INSERT INTO delivered_triggers VALUES (?,?,?,?)',
                               (self.system, self.protocol_hash, f.key, run_id))
                run = db.execute('SELECT * FROM decision_runs WHERE run_id=?', (run_id,)).fetchone()

            return [self._execute(run, handler, now)]

    def _execute(self, run, handler, now):
        if run['status'] == 'ready':
            return self._publish(run, now)
        inputs = json.loads(run['inputs'])
        firings = tuple(Firing(TriggerSpec(**x['spec']), Event(**x['event']), x['decision_at'], x['latency_seconds']) for x in inputs['firings'])
        if any(now > f.spec.deadline(f.event) for f in firings):
            with self.store.transaction() as db:
                db.execute("UPDATE decision_runs SET status='expired_before_decision' WHERE run_id=?", (run['run_id'],))
            return {'run_id': run['run_id'], 'status': 'expired_before_decision', 'paper': True}
        context = DecisionContext(run['run_id'], self.system, run['at'], firings, inputs['protocol'])
        try:
            started = self.timer()
            decision = handler(context)
            elapsed = self.timer()-started
            if elapsed < 0:
                raise ValueError('Processing clock moved backwards')
            
            finished = now+pd.Timedelta(seconds=elapsed)
            if any(finished > f.spec.deadline(f.event) for f in firings):
                with self.store.transaction() as db:
                    db.execute("UPDATE decision_runs SET status='expired_after_decision' WHERE run_id=?", (run['run_id'],))
                return {'run_id': run['run_id'], 'status': 'expired_after_decision', 'paper': True}
            
            target = None
            if decision is not None:
                if not 0 < decision.valid_for_seconds <= 86400:
                    raise ValueError('Target duration must be finite, positive and <= one day')
                
                target = TargetPositions(self.system, run['run_id'], stamp(finished), stamp(finished),
                                        stamp(finished+pd.Timedelta(seconds=decision.valid_for_seconds)),
                                        self.currency, decision.quantities)
            status = 'no_action' if target is None else 'ready'
            with self.store.transaction() as db:
                db.execute('UPDATE decision_runs SET status=?,target=? WHERE run_id=?',
                           (status, None if target is None else canonical(target.to_dict()), run['run_id']))
                updated = db.execute('SELECT * FROM decision_runs WHERE run_id=?', (run['run_id'],)).fetchone()

        except Exception as error:
            with self.store.transaction() as db:
                db.execute("UPDATE decision_runs SET status='failed',error=? WHERE run_id=?", (str(error), run['run_id']))
            raise
        
        if target is None:
            return {'run_id': run['run_id'], 'status': 'no_action', 'paper': True}
        
        return self._publish(updated, finished)
