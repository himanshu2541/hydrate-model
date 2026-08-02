"""
cache.py
--------
SQLite-backed FIFO (First-In-First-Out) cache for hydrate equilibrium results.

Design
------
Each cached entry is identified by a deterministic SHA-256 hash of:
    (gas_comp, liq_comp, T_min, T_max, T_step, eos_name)

All results for a *single run* (all EOS models together) are stored as a
JSON blob so one hash lookup retrieves everything.

FIFO eviction
-------------
When the cache exceeds MAX_ENTRIES, the oldest inserted record (lowest id /
earliest timestamp) is evicted first — first in, first out.  This is the
natural queue behaviour: new runs push old forgotten experiments out the back.

Public API
----------
    cache = ResultsCache()
    hit = cache.get(run_params)          # -> dict | None
    cache.put(run_params, results_dict)
    cache.clear()
    info = cache.info()                  # -> CacheInfo namedtuple
"""

from __future__ import annotations

import sqlite3
import hashlib
import json
import io
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

DB_FILE = Path("hydrate_cache.db")
MAX_ENTRIES = 30  # max number of cached runs (one run = all EOS together)
SCHEMA_VER = 1


# ── helpers ───────────────────────────────────────────────────────────────────


def _round_T_range(T_range: np.ndarray) -> tuple[float, float, float]:
    """Reduce a T array to (min, max, step) for hashing."""
    arr = np.asarray(T_range, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), 0.0
    step = float(round(arr[1] - arr[0], 6))
    return float(round(arr[0], 6)), float(round(arr[-1], 6)), step


def _make_hash(
    gas_comp: dict,
    liq_comp: dict,
    T_range: np.ndarray,
    eos_names: list[str],
) -> str:
    """Deterministic SHA-256 fingerprint for a run configuration."""
    tmin, tmax, tstep = _round_T_range(T_range)
    payload = {
        "gas": dict(sorted((k, round(v, 8)) for k, v in gas_comp.items())),
        "liq": dict(sorted((k, round(v, 8)) for k, v in liq_comp.items())),
        "Tmin": tmin,
        "Tmax": tmax,
        "Tstep": tstep,
        "eos": sorted(eos_names),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ── DataFrame serialisation ───────────────────────────────────────────────────


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(
        buf, index=False, engine="pyarrow" if _has_pyarrow() else "fastparquet"
    )
    return buf.getvalue()


def _bytes_to_df(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data))


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _results_to_json(results: dict[str, pd.DataFrame]) -> str:
    """Serialise {eos_name: DataFrame} → JSON string (CSV inside JSON)."""
    out = {}
    for name, df in results.items():
        out[name] = df.to_csv(index=False)
    return json.dumps(out)


def _json_to_results(blob: str) -> dict[str, pd.DataFrame]:
    raw = json.loads(blob)
    return {name: pd.read_csv(io.StringIO(csv)) for name, csv in raw.items()}


# ── cache info ────────────────────────────────────────────────────────────────


@dataclass
class CacheInfo:
    total_entries: int
    max_entries: int
    db_path: str
    last_access: Optional[str]  # ISO timestamp string or None
    hits_session: int = field(default=0)
    misses_session: int = field(default=0)


# ── main class ────────────────────────────────────────────────────────────────


class ResultsCache:
    """Thread-safe SQLite LIFO cache for hydrate run results."""

    def __init__(self, db_path: Path = DB_FILE, max_entries: int = MAX_ENTRIES):
        self._db = Path(db_path)
        self._max = max_entries
        self._hits = 0
        self._misses = 0
        self._init_db()

    # ── internal ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, timeout=10)

    def _init_db(self):
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_hash    TEXT    NOT NULL UNIQUE,
                    gas_json    TEXT    NOT NULL,
                    liq_json    TEXT    NOT NULL,
                    T_min       REAL    NOT NULL,
                    T_max       REAL    NOT NULL,
                    T_step      REAL    NOT NULL,
                    eos_json    TEXT    NOT NULL,
                    results_json TEXT   NOT NULL,
                    inserted_at TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                    accessed_at TEXT
                )
            """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_hash ON runs (run_hash)")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key  TEXT PRIMARY KEY,
                    val  TEXT
                )
            """
            )
            con.execute(
                "INSERT OR IGNORE INTO meta(key,val) VALUES('schema_ver',?)",
                (str(SCHEMA_VER),),
            )

    # ── public API ────────────────────────────────────────────────────────────

    def get(
        self,
        gas_comp: dict,
        liq_comp: dict,
        T_range: np.ndarray,
        eos_names: list[str],
    ) -> Optional[dict[str, pd.DataFrame]]:
        """Return cached results or None if not found (cache miss)."""
        h = _make_hash(gas_comp, liq_comp, T_range, eos_names)
        with self._conn() as con:
            row = con.execute(
                "SELECT results_json FROM runs WHERE run_hash = ?", (h,)
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            # Update access timestamp
            con.execute(
                "UPDATE runs SET accessed_at = datetime('now','localtime') "
                "WHERE run_hash = ?",
                (h,),
            )
        self._hits += 1
        try:
            return _json_to_results(row[0])
        except Exception as exc:
            log.warning("Cache deserialisation failed (%s); treating as miss.", exc)
            self._misses += 1
            return None

    def put(
        self,
        gas_comp: dict,
        liq_comp: dict,
        T_range: np.ndarray,
        eos_names: list[str],
        results: dict[str, pd.DataFrame],
    ) -> None:
        """Insert results into the cache.  Evicts LIFO entries if over capacity."""
        h = _make_hash(gas_comp, liq_comp, T_range, eos_names)
        tmin, tmax, tstep = _round_T_range(T_range)
        blob = _results_to_json(results)

        with self._conn() as con:
            # Upsert (replace existing entry with same hash)
            con.execute(
                """
                INSERT INTO runs
                    (run_hash, gas_json, liq_json, T_min, T_max, T_step,
                     eos_json, results_json, inserted_at)
                VALUES (?,?,?,?,?,?,?,?,datetime('now','localtime'))
                ON CONFLICT(run_hash) DO UPDATE SET
                    results_json = excluded.results_json,
                    inserted_at  = excluded.inserted_at,
                    accessed_at  = NULL
            """,
                (
                    h,
                    json.dumps(dict(sorted(gas_comp.items()))),
                    json.dumps(dict(sorted(liq_comp.items()))),
                    tmin,
                    tmax,
                    tstep,
                    json.dumps(sorted(eos_names)),
                    blob,
                ),
            )

            # FIFO eviction: while over capacity, delete the lowest-id row
            # (the oldest inserted entry, i.e. the front of the queue)
            while True:
                count = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
                if count <= self._max:
                    break
                # FIFO: delete the oldest entry (lowest id = first in)
                victim = con.execute(
                    "SELECT id FROM runs ORDER BY id ASC LIMIT 1"
                ).fetchone()
                if victim is None:
                    break
                con.execute("DELETE FROM runs WHERE id = ?", (victim[0],))
                log.debug("FIFO eviction: removed oldest cache entry id=%d", victim[0])

    def clear(self) -> None:
        """Wipe all cached entries."""
        with self._conn() as con:
            con.execute("DELETE FROM runs")
        self._hits = self._misses = 0

    def info(self) -> CacheInfo:
        with self._conn() as con:
            total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            last = con.execute(
                "SELECT accessed_at FROM runs ORDER BY accessed_at DESC LIMIT 1"
            ).fetchone()
        return CacheInfo(
            total_entries=total,
            max_entries=self._max,
            db_path=str(self._db.resolve()),
            last_access=last[0] if last and last[0] else None,
            hits_session=self._hits,
            misses_session=self._misses,
        )

    def list_entries(self) -> list[dict]:
        """Return a list of summary dicts for all cached entries (newest first)."""
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT id, gas_json, liq_json, T_min, T_max, T_step,
                       eos_json, inserted_at, accessed_at
                FROM runs ORDER BY id DESC
            """
            ).fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "gas": json.loads(row[1]),
                    "liq": json.loads(row[2]),
                    "T_min": row[3],
                    "T_max": row[4],
                    "T_step": row[5],
                    "eos": json.loads(row[6]),
                    "inserted_at": row[7],
                    "accessed_at": row[8],
                }
            )
        return result


# ── module-level singleton ────────────────────────────────────────────────────
_cache: Optional[ResultsCache] = None


def get_cache() -> ResultsCache:
    """Return the module-level singleton cache instance."""
    global _cache
    if _cache is None:
        _cache = ResultsCache()
    return _cache
