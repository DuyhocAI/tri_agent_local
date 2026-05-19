"""
HERMES AGENT — memory/memory_manager.py  (v2 — SQLite backend)

WHY this replaces the flat-JSON design
──────────────────────────────────────
  Old design problems:
    • One JSON file per session  → full file rewritten on every message
    • Entire list scanned in RAM to prune/dedup  → O(n) on every write
    • Messages truncated at 300 chars in orchestrator before being stored
    • No way to search past exchanges
    • Two separate glob() calls on startup to reconstruct state

  New design (SQLite + WAL mode):
    • Single DB file, never rewritten fully — only rows inserted/deleted
    • Indexed queries: O(log n) lookups by session_id + timestamp
    • Full message content stored — no truncation
    • search_short_term() for keyword recall mid-session
    • UNIQUE constraint on (session_id, topic) replaces manual dedup loop
    • Thread-safe via WAL mode + module-level write lock

Schema
──────
  short_term  id · session_id · role · user_msg · assistant_msg · ts
  long_term   id · session_id · topic · value · ts
                UNIQUE(session_id, topic) ON CONFLICT REPLACE

Storage: ~/.hermes/memory/session_memory.db

Public API is a drop-in replacement — all method signatures unchanged.
"""

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import MEMORY_CONFIG

logger = logging.getLogger("hermes.memory")

# ── Storage ───────────────────────────────────────────────────────────────────

MEMORY_DIR = Path.home() / ".hermes" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = MEMORY_DIR / "session_memory.db"

SHORT_MAX      = MEMORY_CONFIG["short_term_max"]   # 30
LONG_MAX       = MEMORY_CONFIG["long_term_max"]    # 200
EXPIRE_SECONDS = MEMORY_CONFIG["expire_seconds"]   # 7200

# Single write-lock — keeps multi-threaded writes serialised without a
# connection pool. SQLite WAL allows concurrent reads at no extra cost.
_LOCK = threading.Lock()


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS short_term (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'exchange',
    user_msg      TEXT    NOT NULL DEFAULT '',
    assistant_msg TEXT    NOT NULL DEFAULT '',
    ts            TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_st_session_ts
    ON short_term (session_id, ts);

CREATE TABLE IF NOT EXISTS long_term (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    topic      TEXT NOT NULL,
    value      TEXT NOT NULL DEFAULT '',
    ts         TEXT NOT NULL,
    UNIQUE (session_id, topic) ON CONFLICT REPLACE
);
CREATE INDEX IF NOT EXISTS idx_lt_session
    ON long_term (session_id);
"""


def _open() -> sqlite3.Connection:
    """Open a connection to the DB.  check_same_thread=False is safe here
    because every write is protected by _LOCK."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _LOCK:
        c = _open()
        c.executescript(_SCHEMA)
        c.commit()
        c.close()


_init()   # run once at import time


def _cutoff_ts() -> str:
    """ISO string for the oldest short-term entry worth keeping."""
    dt = datetime.now(timezone.utc) - timedelta(seconds=EXPIRE_SECONDS)
    return dt.replace(tzinfo=None).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
#  MemoryManager
# ══════════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """
    Drop-in replacement for the v1 flat-JSON MemoryManager.
    All public method signatures are identical.
    """

    # ── Short-term: read ──────────────────────────────────────────────────────

    def get_short_term(self, session_id: str) -> list:
        """
        Return all non-expired entries for this session, oldest → newest.
        Each entry dict has keys: role, user, assistant, ts.
        """
        with _LOCK:
            c = _open()
            try:
                rows = c.execute(
                    """
                    SELECT role,
                           user_msg      AS user,
                           assistant_msg AS assistant,
                           ts
                    FROM   short_term
                    WHERE  session_id = ?
                      AND  ts >= ?
                    ORDER  BY id ASC
                    LIMIT  ?
                    """,
                    (session_id, _cutoff_ts(), SHORT_MAX),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                c.close()

    # ── Short-term: write ─────────────────────────────────────────────────────

    def add_short_term(self, session_id: str, entry: dict):
        """
        Insert one exchange.  Expected keys: role, user, assistant, ts.
        Automatically prunes expired rows and enforces SHORT_MAX cap.
        """
        if not isinstance(entry, dict):
            return

        role     = entry.get("role", "exchange")
        user_msg = entry.get("user", "")
        asst_msg = entry.get("assistant", "")
        ts       = entry.get("ts") or datetime.now().isoformat()

        with _LOCK:
            c = _open()
            try:
                # Insert new row
                c.execute(
                    """
                    INSERT INTO short_term
                        (session_id, role, user_msg, assistant_msg, ts)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, role, user_msg, asst_msg, ts),
                )

                # Prune rows older than EXPIRE_SECONDS
                c.execute(
                    "DELETE FROM short_term WHERE session_id = ? AND ts < ?",
                    (session_id, _cutoff_ts()),
                )

                # Enforce hard cap — keep newest SHORT_MAX rows per session
                c.execute(
                    """
                    DELETE FROM short_term
                    WHERE  session_id = ?
                      AND  id NOT IN (
                               SELECT id FROM short_term
                               WHERE  session_id = ?
                               ORDER  BY id DESC
                               LIMIT  ?
                           )
                    """,
                    (session_id, session_id, SHORT_MAX),
                )
                c.commit()
            except Exception as e:
                logger.warning(f"add_short_term error: {e}")
                c.rollback()
            finally:
                c.close()

    # ── Long-term: read ───────────────────────────────────────────────────────

    def get_long_term(self, session_id: str) -> list:
        """Return all long-term facts for this session. Each entry: {topic, value, ts}."""
        with _LOCK:
            c = _open()
            try:
                rows = c.execute(
                    """
                    SELECT topic, value, ts
                    FROM   long_term
                    WHERE  session_id = ?
                    ORDER  BY id DESC
                    LIMIT  ?
                    """,
                    (session_id, LONG_MAX),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                c.close()

    # ── Long-term: write ──────────────────────────────────────────────────────

    def add_long_term(self, session_id: str, entry: dict):
        """
        Upsert a fact by (session_id, topic).
        The UNIQUE ON CONFLICT REPLACE constraint handles dedup — no loop.
        """
        if not isinstance(entry, dict):
            return

        topic = entry.get("topic", "").strip()
        value = entry.get("value", "")
        ts    = entry.get("ts") or datetime.now().isoformat()

        if not topic:
            return

        with _LOCK:
            c = _open()
            try:
                c.execute(
                    """
                    INSERT OR REPLACE INTO long_term
                        (session_id, topic, value, ts)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, topic, value, ts),
                )

                # Enforce hard cap — keep newest LONG_MAX rows per session
                c.execute(
                    """
                    DELETE FROM long_term
                    WHERE  session_id = ?
                      AND  id NOT IN (
                               SELECT id FROM long_term
                               WHERE  session_id = ?
                               ORDER  BY id DESC
                               LIMIT  ?
                           )
                    """,
                    (session_id, session_id, LONG_MAX),
                )
                c.commit()
            except Exception as e:
                logger.warning(f"add_long_term error: {e}")
                c.rollback()
            finally:
                c.close()

    # ── Keyword search (new) ──────────────────────────────────────────────────

    def search_short_term(self, session_id: str, query: str, limit: int = 10) -> list:
        """
        Search past exchanges containing `query` in either side of the message.
        Returns up to `limit` results, newest first.
        Useful for the orchestrator to surface relevant context on long sessions.
        """
        pattern = f"%{query}%"
        with _LOCK:
            c = _open()
            try:
                rows = c.execute(
                    """
                    SELECT role,
                           user_msg      AS user,
                           assistant_msg AS assistant,
                           ts
                    FROM   short_term
                    WHERE  session_id = ?
                      AND  (user_msg LIKE ? OR assistant_msg LIKE ?)
                    ORDER  BY id DESC
                    LIMIT  ?
                    """,
                    (session_id, pattern, pattern, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                c.close()

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear(self, session_id: str, scope: str = "short"):
        """scope: 'short' | 'long' | 'all'"""
        with _LOCK:
            c = _open()
            try:
                if scope in ("short", "all"):
                    c.execute(
                        "DELETE FROM short_term WHERE session_id = ?",
                        (session_id,),
                    )
                if scope in ("long", "all"):
                    c.execute(
                        "DELETE FROM long_term WHERE session_id = ?",
                        (session_id,),
                    )
                c.commit()
            except Exception as e:
                logger.warning(f"clear error: {e}")
                c.rollback()
            finally:
                c.close()

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_sessions(self) -> list:
        """Return all session IDs that have any stored data."""
        with _LOCK:
            c = _open()
            try:
                rows = c.execute(
                    """
                    SELECT DISTINCT session_id FROM short_term
                    UNION
                    SELECT DISTINCT session_id FROM long_term
                    ORDER BY session_id
                    """
                ).fetchall()
                return [r[0] for r in rows]
            finally:
                c.close()

    def session_summary(self, session_id: str) -> dict:
        with _LOCK:
            c = _open()
            try:
                sc = c.execute(
                    "SELECT COUNT(*) FROM short_term WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
                lc = c.execute(
                    "SELECT COUNT(*) FROM long_term WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
                return {
                    "session_id":  session_id,
                    "short_count": sc,
                    "long_count":  lc,
                    "db_path":     str(DB_PATH),
                }
            finally:
                c.close()

    def db_stats(self) -> dict:
        """Global DB stats — exposed via /api/system if desired."""
        with _LOCK:
            c = _open()
            try:
                st = c.execute("SELECT COUNT(*) FROM short_term").fetchone()[0]
                lt = c.execute("SELECT COUNT(*) FROM long_term").fetchone()[0]
                sz = DB_PATH.stat().st_size if DB_PATH.exists() else 0
                return {
                    "short_term_rows": st,
                    "long_term_rows":  lt,
                    "db_size_kb":      round(sz / 1024, 1),
                    "db_path":         str(DB_PATH),
                }
            finally:
                c.close()