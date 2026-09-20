"""Model registry backed by in-process SQLite."""

import json
import sqlite3
import threading


class ModelStore:
    """Thread-safe store of named HMM model specs."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS models ("
                "  name TEXT PRIMARY KEY,"
                "  spec TEXT NOT NULL"
                ")"
            )
            self._conn.commit()

    def save(self, spec: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO models (name, spec) VALUES (?, ?)",
                (spec["name"], json.dumps(spec)),
            )
            self._conn.commit()

    def get(self, name: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT spec FROM models WHERE name = ?", (name,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def names(self) -> list:
        with self._lock:
            rows = self._conn.execute("SELECT name FROM models ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
