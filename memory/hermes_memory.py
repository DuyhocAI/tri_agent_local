"""
Hermes cross-session memory — memory/hermes_memory.py  (P3)

Additive layer on top of the existing MemoryManager (SQLite per-session).
Uses a separate DB file so the original memory_manager.py is never touched.

Schema additions:
  user_facts   — persistent facts about the user across all sessions
  episodes     — session summaries for long-term recall
  skill_usage  — analytics: which skills/tools are used most

All writes are synchronous; reads are O(log n) via indexes.
WAL mode allows concurrent reads while writes are in progress.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from config import HERMES_CONFIG

logger = logging.getLogger("hermes.memory")

_DB_PATH = Path(HERMES_CONFIG["memory_dir"]) / "hermes_memory.db"
_WRITE_LOCK = threading.Lock()

_DDL = """
CREATE TABLE IF NOT EXISTS user_facts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT    NOT NULL DEFAULT 'local',
    topic          TEXT    NOT NULL,
    value          TEXT    NOT NULL DEFAULT '',
    confidence     REAL    NOT NULL DEFAULT 1.0,
    source_session TEXT,
    ts             TEXT    NOT NULL,
    UNIQUE(user_id, topic) ON CONFLICT REPLACE
);
CREATE INDEX IF NOT EXISTS idx_uf_user ON user_facts (user_id);

CREATE TABLE IF NOT EXISTS episodes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    user_id    TEXT    NOT NULL DEFAULT 'local',
    summary    TEXT    NOT NULL,
    agent      TEXT,
    skill_tags TEXT,
    ts         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ep_user ON episodes (user_id, ts);
CREATE INDEX IF NOT EXISTS idx_ep_session ON episodes (session_id);

CREATE TABLE IF NOT EXISTS skill_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name TEXT    NOT NULL,
    agent      TEXT,
    session_id TEXT,
    success    INTEGER NOT NULL DEFAULT 1,
    latency_ms REAL,
    ts         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_su_skill ON skill_usage (skill_name, ts);
"""


class HermesMemory:
    """
    Cross-session memory and analytics store.

    Thread-safe via a module-level write lock + SQLite WAL mode.
    All public methods are safe to call from background threads.
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with _WRITE_LOCK:
            con = sqlite3.connect(str(self._db_path))
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_DDL)
            con.commit()
            con.close()
        logger.info(f"HermesMemory initialised at {self._db_path}")

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), timeout=10)
        con.row_factory = sqlite3.Row
        return con

    # ── User facts ────────────────────────────────────────────────────────────

    def get_user_facts(self, user_id: str = "local", limit: int = 20) -> list[dict]:
        """Return the most recent facts about a user, capped to avoid prompt bloat."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT topic, value, confidence, ts FROM user_facts "
                "WHERE user_id = ? ORDER BY ts DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_user_fact(
        self,
        topic: str,
        value: str,
        confidence: float = 1.0,
        session_id: str | None = None,
        user_id: str = "local",
    ) -> None:
        """Insert or replace a user fact (UNIQUE on user_id + topic)."""
        ts = datetime.now().isoformat()
        with _WRITE_LOCK:
            with self._conn() as con:
                con.execute(
                    "INSERT OR REPLACE INTO user_facts "
                    "(user_id, topic, value, confidence, source_session, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, topic, value, confidence, session_id, ts),
                )
                con.commit()

    def delete_user_fact(self, topic: str, user_id: str = "local") -> None:
        with _WRITE_LOCK:
            with self._conn() as con:
                con.execute(
                    "DELETE FROM user_facts WHERE user_id = ? AND topic = ?",
                    (user_id, topic),
                )
                con.commit()

    # ── Episodes ─────────────────────────────────────────────────────────────

    def add_episode(
        self,
        session_id: str,
        summary: str,
        agent: str | None = None,
        skill_tags: list[str] | None = None,
        user_id: str = "local",
    ) -> None:
        """Record a session episode summary for long-term recall."""
        ts   = datetime.now().isoformat()
        tags = json.dumps(skill_tags or [])
        with _WRITE_LOCK:
            with self._conn() as con:
                con.execute(
                    "INSERT INTO episodes (session_id, user_id, summary, agent, skill_tags, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, user_id, summary, agent, tags, ts),
                )
                con.commit()

    def get_recent_episodes(
        self,
        user_id: str = "local",
        limit: int = 10,
    ) -> list[dict]:
        """Return the most recent N episodes for a user."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT session_id, summary, agent, skill_tags, ts FROM episodes "
                "WHERE user_id = ? ORDER BY ts DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["skill_tags"] = json.loads(d["skill_tags"] or "[]")
            except Exception:
                d["skill_tags"] = []
            result.append(d)
        return result

    # ── Skill usage analytics ─────────────────────────────────────────────────

    def log_skill_usage(
        self,
        skill_name: str,
        agent: str | None = None,
        session_id: str | None = None,
        success: bool = True,
        latency_ms: float | None = None,
    ) -> None:
        ts = datetime.now().isoformat()
        with _WRITE_LOCK:
            with self._conn() as con:
                con.execute(
                    "INSERT INTO skill_usage (skill_name, agent, session_id, success, latency_ms, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (skill_name, agent, session_id, int(success), latency_ms, ts),
                )
                con.commit()

    def get_skill_stats(self) -> list[dict]:
        """Return skill usage counts grouped by skill_name."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT skill_name, COUNT(*) as total, "
                "SUM(success) as successes, AVG(latency_ms) as avg_latency_ms "
                "FROM skill_usage GROUP BY skill_name ORDER BY total DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Context compression (lossless context management) ─────────────────────

    def compress_context(
        self,
        messages: list[dict],
        max_context_msgs: int,
        llm_collect_fn,
    ) -> list[dict]:
        """
        If messages exceed 2*max_context_msgs+2, summarise the oldest half
        into a single [Context summary] block using the reviewer LLM.

        Preserves: system message (index 0) + most recent max_context_msgs*2 messages.
        The summary replaces the discarded messages as a synthetic user message.

        llm_collect_fn: callable(prompt: str) -> str
        """
        if len(messages) <= max_context_msgs * 2 + 2:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]

        keep_count  = max_context_msgs * 2
        to_compress = non_system[:-keep_count] if keep_count else non_system
        to_keep     = non_system[-keep_count:] if keep_count else []

        if not to_compress:
            return messages

        context_block = "\n".join(
            f"{m['role'].upper()}: {m.get('content','')[:400]}"
            for m in to_compress
        )
        prompt = (
            f"Summarise the following conversation excerpt in 3-5 sentences, "
            f"focusing on key decisions, facts, and context that affect future replies:\n\n"
            f"{context_block}"
        )
        try:
            summary = llm_collect_fn(prompt)
        except Exception as e:
            logger.warning(f"Context compression LLM failed: {e}")
            return messages

        summary_msg = {
            "role":    "user",
            "content": f"[Context summary of earlier conversation]: {summary}",
        }
        return system_msgs + [summary_msg] + to_keep

    def compress_session(
        self,
        session_id: str,
        memory_manager,
        llm_collect_fn,
        user_id: str = "local",
    ) -> str:
        """
        Summarise a completed session and store it as an episode.
        Runs in a background daemon thread — never blocks the SSE stream.
        """
        try:
            short_mem = memory_manager.get_short_term(session_id)
            if not short_mem:
                return ""

            blocks = []
            for m in short_mem[-15:]:
                if isinstance(m, dict):
                    u = m.get("user", "")[:200]
                    a = m.get("assistant", "")[:200]
                    if u or a:
                        blocks.append(f"User: {u}\nAssistant: {a}")
            if not blocks:
                return ""

            prompt = (
                "Summarise this session in 2-3 sentences. "
                "Include: main topic, any decisions made, any code/files created.\n\n"
                + "\n---\n".join(blocks)
            )
            summary = llm_collect_fn(prompt)
            if summary:
                self.add_episode(session_id, summary, user_id=user_id)
            return summary
        except Exception as e:
            logger.debug(f"Session compression failed: {e}")
            return ""

    # ── DB stats ──────────────────────────────────────────────────────────────

    def db_stats(self) -> dict:
        """Return row counts and file size — used by subconscious health checks."""
        try:
            size_mb = self._db_path.stat().st_size / 1_048_576
        except Exception:
            size_mb = 0.0
        with self._conn() as con:
            facts    = con.execute("SELECT COUNT(*) FROM user_facts").fetchone()[0]
            episodes = con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            skills   = con.execute("SELECT COUNT(*) FROM skill_usage").fetchone()[0]
        return {
            "size_mb":  round(size_mb, 2),
            "facts":    facts,
            "episodes": episodes,
            "skill_uses": skills,
        }

    def prune_old_episodes(self, keep_days: int = 90) -> int:
        """Remove episodes older than keep_days. Returns number deleted."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
        with _WRITE_LOCK:
            with self._conn() as con:
                cur = con.execute(
                    "DELETE FROM episodes WHERE ts < ?", (cutoff,)
                )
                con.commit()
        deleted = cur.rowcount
        if deleted:
            logger.info(f"Pruned {deleted} old episodes")
        return deleted
