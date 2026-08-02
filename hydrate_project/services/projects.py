"""
projects.py
-----------
SQLite-backed project store -- named bundles of Setup-page input config
(gas/liq composition, T range, EOS selection, exp-data mode), analogous to
a COMSOL model file / Aspen HYSYS case. Lives in the same hydrate_cache.db
file as services/cache.py's ResultsCache (separate table, no FK coupling).

Public API
----------
    store = ProjectStore()
    store.list()                        # -> list[dict] (summaries, newest first)
    store.create(name)                  # -> dict (full project, empty config)
    store.get(project_id)                # -> dict | None
    store.update_config(project_id, cfg) # -> None
    store.rename(project_id, name)       # -> None
    store.delete(project_id)             # -> None
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_FILE = Path("hydrate_cache.db")


class ProjectStore:
    def __init__(self, db_path: Path = DB_FILE):
        self._db = Path(db_path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, timeout=10)

    def _init_db(self):
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL,
                    config_json TEXT    NOT NULL DEFAULT '{}',
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                    updated_at  TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
                )
                """
            )

    def _row_to_dict(self, row: tuple) -> dict:
        return {
            "id": row[0],
            "name": row[1],
            "config": json.loads(row[2]),
            "created_at": row[3],
            "updated_at": row[4],
        }

    def list(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, name, config_json, created_at, updated_at "
                "FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def create(self, name: str) -> dict:
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO projects (name, config_json) VALUES (?, '{}')",
                (name,),
            )
            pid = cur.lastrowid
        return self.get(pid)

    def get(self, project_id: int) -> Optional[dict]:
        with self._conn() as con:
            row = con.execute(
                "SELECT id, name, config_json, created_at, updated_at "
                "FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def update_config(self, project_id: int, config: dict) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE projects SET config_json = ?, updated_at = datetime('now','localtime') "
                "WHERE id = ?",
                (json.dumps(config), project_id),
            )

    def rename(self, project_id: int, name: str) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE projects SET name = ?, updated_at = datetime('now','localtime') "
                "WHERE id = ?",
                (name, project_id),
            )

    def delete(self, project_id: int) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM projects WHERE id = ?", (project_id,))


_store: Optional[ProjectStore] = None


def get_project_store() -> ProjectStore:
    global _store
    if _store is None:
        _store = ProjectStore()
    return _store
