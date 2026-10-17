"""Durable actuator audit. Never called while a valve is intentionally held open."""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class FiringHistory:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, started TEXT NOT NULL, outcome TEXT NOT NULL, opened INTEGER NOT NULL DEFAULT 0, data TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY, attempt_id TEXT, timestamp TEXT NOT NULL, data TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS attempts_started ON attempts(started DESC)')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('since', utc_now()))
            pending = db.execute("SELECT id,data FROM attempts WHERE outcome='in_progress'").fetchall()
            for attempt_id, raw in pending:
                data = json.loads(raw)
                data.update(outcome='interrupted', error='Process ended before final outcome; consult pulse evidence.', finished_at=utc_now())
                self._save(db, attempt_id, data, 'recovered_after_restart')

    def _save(self, db, attempt_id, data, event):
        raw = json.dumps(data, allow_nan=False)
        db.execute('INSERT INTO attempts VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET outcome=excluded.outcome,opened=excluded.opened,data=excluded.data',
                   (attempt_id, data['started_at'], data['outcome'], int(data.get('valve_open_returned', False)), raw))
        record = dict(data, event=event, timestamp=utc_now())
        db.execute('INSERT INTO events(attempt_id,timestamp,data) VALUES (?,?,?)',
                   (attempt_id, record['timestamp'], json.dumps(record, allow_nan=False)))

    def begin(self, **details):
        attempt_id = uuid4().hex
        data = dict(details, id=attempt_id, started_at=utc_now(), outcome='in_progress',
                    pulse_evidence='not_requested', valve_open_returned=False)
        with self._lock, self._connect() as db:
            self._save(db, attempt_id, data, 'attempt_started')
        return attempt_id

    def update(self, attempt_id, event, **details):
        with self._lock, self._connect() as db:
            row = db.execute('SELECT data FROM attempts WHERE id=?', (attempt_id,)).fetchone()
            if row is None:
                raise ValueError('unknown firing attempt')
            data = json.loads(row[0])
            data.update(details)
            self._save(db, attempt_id, data, event)

    def snapshot(self, limit=20):
        with self._lock, self._connect() as db:
            since = db.execute("SELECT value FROM metadata WHERE key='since'").fetchone()[0]
            counts = dict(db.execute('SELECT outcome,count(*) FROM attempts GROUP BY outcome'))
            shots = db.execute('SELECT count(*) FROM attempts WHERE opened=1').fetchone()[0]
            last = db.execute('SELECT data FROM attempts WHERE opened=1 ORDER BY started DESC LIMIT 1').fetchone()
            recent = [json.loads(row[0]) for row in db.execute('SELECT data FROM attempts ORDER BY started DESC LIMIT ?', (limit,))]
            return dict(since=since, shots=shots, attempts=sum(counts.values()), outcomes=counts,
                        last_shot=json.loads(last[0]) if last else None, recent=recent,
                        evidence='Software valve commands; no water-flow sensor.')

    def export(self):
        # Page by immutable event sequence; no long read transaction or unbounded memory.
        with self._connect() as db:
            end = db.execute('SELECT coalesce(max(sequence),0) FROM events').fetchone()[0]
        sequence = 0
        while sequence < end:
            with self._connect() as db:
                rows = db.execute('SELECT sequence,data FROM events WHERE sequence>? AND sequence<=? ORDER BY sequence LIMIT 100', (sequence, end)).fetchall()
            if not rows:
                break
            for sequence, data in rows:
                yield data + '\n'
