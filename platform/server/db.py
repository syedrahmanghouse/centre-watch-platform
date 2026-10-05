"""Persistent application records and append-only audit history."""
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def uid(prefix):
    return f'{prefix}-{uuid.uuid4().hex}'


class Database:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.root / 'centrewatch.sqlite3', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA foreign_keys=ON')
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,email TEXT UNIQUE NOT NULL,name TEXT NOT NULL,role TEXT NOT NULL,institution_id TEXT,password_hash TEXT NOT NULL,specialty TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id),csrf TEXT NOT NULL,expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS records(kind TEXT NOT NULL,id TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS audit(id TEXT PRIMARY KEY,time TEXT NOT NULL,actor_id TEXT,institution_id TEXT,class_id TEXT,type TEXT NOT NULL,message TEXT NOT NULL,data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS edge_events(event_id TEXT PRIMARY KEY,device_id TEXT NOT NULL,received_at TEXT NOT NULL,data TEXT NOT NULL);
        ''')
        self.conn.commit()

    @contextmanager
    def transaction(self):
        with self.lock:
            try:
                yield self
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def execute(self, sql, params=()):
        return self.conn.execute(sql, params)

    def one(self, sql, params=()):
        with self.lock:
            row = self.conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def all(self, sql, params=()):
        with self.lock:
            return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def get(self, kind, record_id):
        row = self.one('SELECT data FROM records WHERE kind=? AND id=?', (kind, record_id))
        return json.loads(row['data']) if row else None

    def list(self, kind):
        return [json.loads(r['data']) for r in self.all('SELECT data FROM records WHERE kind=? ORDER BY rowid DESC', (kind,))]

    def put(self, kind, record):
        self.execute('INSERT INTO records(kind,id,data) VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data', (kind, record['id'], json.dumps(record)))
        return record

    def audit(self, actor, institution_id, class_id, type_, message, data=None):
        self.execute('INSERT INTO audit VALUES(?,?,?,?,?,?,?,?)', (uid('audit'), now(), actor, institution_id, class_id, type_, message, json.dumps(data or {})))

    def activities(self):
        return [{'id':r['id'], 'time':r['time'], 'actorId':r['actor_id'], 'institutionId':r['institution_id'], 'classId':r['class_id'], 'type':r['type'], 'message':r['message'], 'data':json.loads(r['data'])} for r in self.all('SELECT * FROM audit ORDER BY time DESC LIMIT 1000')]
