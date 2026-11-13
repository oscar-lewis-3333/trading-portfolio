
#local evidence, atomic publication, cross-process locking

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def platform_home():
    return Path(os.environ.get('FOUNDATION_HOME', Path.home() / 'Desktop/python/trading_portfolio/foundation')).resolve()


def check_disk(path, minimum=8_000_000_000):
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    free = shutil.disk_usage(p).free
    if free < minimum:
        raise OSError(f'Free disk {free} bytes is below {minimum}; stop.')
    
    return free


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lock(path):
    #OS releases advisory lock after crashes

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if handle.tell() == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, value, *, overwrite=False):
    path = Path(path)
    check_disk(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with lock(path.with_suffix(path.suffix + '.lock')):
        if path.exists() and not overwrite:
            if json.loads(path.read_text()) != value:
                raise FileExistsError(f'Immutable file differs: {path}')
            return path
        
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)
    return path


class EvidenceLog:
    #hash chain JSONL, rejects torn/changed evidence on read
    def __init__(self, path):
        self.path = Path(path)

    def _read(self):
        if not self.path.exists():
            return []
        
        raw = self.path.read_text()
        if raw and not raw.endswith('\n'):
            raise ValueError('Torn evidence line; preserve file and repair explicitly')
        
        entries, previous = [], None
        for line in raw.splitlines():
            entry = json.loads(line)
            digest = entry.pop('sha256')
            if entry.get('previous') != previous or fingerprint(entry) != digest:
                raise ValueError('Evidence chain mismatch')
            
            entry['sha256'] = digest; previous = digest; entries.append(entry)
        return entries

    def read(self):
        with lock(self.path.with_suffix('.lock')):
            return self._read()

    def append(self, record):
        check_disk(self.path)
        with lock(self.path.with_suffix('.lock')):
            prior = self._read()
            entry = {'recorded_at': utc_now(), 'previous': prior[-1]['sha256'] if prior else None, 'record': record}
            entry['sha256'] = fingerprint(entry)
            with self.path.open('a') as stream:
                stream.write(canonical(entry) + '\n'); stream.flush(); os.fsync(stream.fileno())
        return entry
