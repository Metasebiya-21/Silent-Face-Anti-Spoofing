"""Persistent audit trail of finished sessions: decision, scores and timings, never raw media.

Redis only caches live sessions; this is the record that outlives them. SQLite for the PoC; anything with a
`record(AuditRecord)` method can replace it (e.g. Postgres) without touching the worker.
"""
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class AuditRecord:
    session_id: str
    decision: str              # live / spoof / retry
    outcome: str | None        # colour-challenge outcome; None if analysis didn't run
    scores: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)  # unix seconds: created_at, uploaded_at, analysed_at, ...
    error: str | None = None   # internal reason, never sent to the client


class AuditStore(Protocol):
    def record(self, entry: AuditRecord) -> None: ...


class SQLiteAuditStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS liveness_audit (
                session_id TEXT PRIMARY KEY, decision TEXT NOT NULL, outcome TEXT,
                scores TEXT NOT NULL, timings TEXT NOT NULL, error TEXT,
                recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def _connect(self):
        return sqlite3.connect(self.path)

    def record(self, entry):
        row = asdict(entry)
        with self._connect() as db:  # commits on success
            db.execute("INSERT OR REPLACE INTO liveness_audit (session_id, decision, outcome, scores, timings, error)"
                       " VALUES (?, ?, ?, ?, ?, ?)",
                       (row["session_id"], row["decision"], row["outcome"], json.dumps(row["scores"]),
                        json.dumps(row["timings"]), row["error"]))

    def get(self, session_id):
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM liveness_audit WHERE session_id = ?", (session_id,)).fetchone()
        return None if row is None else {**dict(row), "scores": json.loads(row["scores"]),
                                         "timings": json.loads(row["timings"])}
