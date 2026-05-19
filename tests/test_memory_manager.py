"""Tests for MemoryManager — SQLite-backed session memory."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from memory.memory_manager import MemoryManager

SID  = "test-session-1"
SID2 = "test-session-2"


@pytest.fixture
def mm(tmp_path):
    """Fresh MemoryManager backed by a temporary DB (auto-deleted after each test)."""
    return MemoryManager(db_path=tmp_path / "test_session_memory.db")


# ── Short-term ────────────────────────────────────────────────────────────────

class TestShortTerm:
    def test_add_and_get(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "hello", "assistant": "hi"})
        rows = mm.get_short_term(SID)
        assert len(rows) == 1
        assert rows[0]["user"] == "hello"
        assert rows[0]["assistant"] == "hi"
        assert rows[0]["role"] == "exchange"

    def test_multiple_entries_oldest_first(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "first",  "assistant": "a1"})
        mm.add_short_term(SID, {"role": "exchange", "user": "second", "assistant": "a2"})
        rows = mm.get_short_term(SID)
        assert len(rows) == 2
        assert rows[0]["user"] == "first"
        assert rows[1]["user"] == "second"

    def test_session_isolation(self, mm):
        mm.add_short_term(SID,  {"role": "exchange", "user": "A", "assistant": "rA"})
        mm.add_short_term(SID2, {"role": "exchange", "user": "B", "assistant": "rB"})
        assert len(mm.get_short_term(SID))  == 1
        assert len(mm.get_short_term(SID2)) == 1
        assert mm.get_short_term(SID)[0]["user"]  == "A"
        assert mm.get_short_term(SID2)[0]["user"] == "B"

    def test_empty_session_returns_empty_list(self, mm):
        assert mm.get_short_term("nonexistent") == []

    def test_ignores_non_dict_entry(self, mm):
        mm.add_short_term(SID, "not a dict")
        assert mm.get_short_term(SID) == []


class TestShortTermSearch:
    def test_search_finds_match_in_user(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "python tips",       "assistant": "use list comprehensions"})
        mm.add_short_term(SID, {"role": "exchange", "user": "java hello world",  "assistant": "System.out.println"})
        results = mm.search_short_term(SID, "python")
        assert len(results) == 1
        assert "python" in results[0]["user"]

    def test_search_finds_match_in_assistant(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "how to sort?", "assistant": "use sorted() in Python"})
        results = mm.search_short_term(SID, "sorted")
        assert len(results) == 1

    def test_search_no_match_returns_empty(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "hello", "assistant": "world"})
        assert mm.search_short_term(SID, "xxxxnotfound") == []

    def test_search_respects_session(self, mm):
        mm.add_short_term(SID,  {"role": "exchange", "user": "python", "assistant": "ok"})
        mm.add_short_term(SID2, {"role": "exchange", "user": "java",   "assistant": "ok"})
        assert len(mm.search_short_term(SID, "python")) == 1
        assert len(mm.search_short_term(SID, "java"))   == 0


# ── Long-term ─────────────────────────────────────────────────────────────────

class TestLongTerm:
    def test_add_and_get(self, mm):
        mm.add_long_term(SID, {"topic": "user_name", "value": "Alice"})
        rows = mm.get_long_term(SID)
        assert len(rows) == 1
        assert rows[0]["topic"] == "user_name"
        assert rows[0]["value"] == "Alice"

    def test_upsert_same_topic(self, mm):
        mm.add_long_term(SID, {"topic": "lang", "value": "Python"})
        mm.add_long_term(SID, {"topic": "lang", "value": "Go"})
        rows = mm.get_long_term(SID)
        assert len(rows) == 1
        assert rows[0]["value"] == "Go"

    def test_multiple_topics(self, mm):
        mm.add_long_term(SID, {"topic": "lang",  "value": "Python"})
        mm.add_long_term(SID, {"topic": "theme", "value": "dark"})
        assert len(mm.get_long_term(SID)) == 2

    def test_empty_topic_ignored(self, mm):
        mm.add_long_term(SID, {"topic": "", "value": "should be ignored"})
        assert mm.get_long_term(SID) == []

    def test_session_isolation(self, mm):
        mm.add_long_term(SID,  {"topic": "pref", "value": "dark"})
        mm.add_long_term(SID2, {"topic": "pref", "value": "light"})
        assert mm.get_long_term(SID)[0]["value"]  == "dark"
        assert mm.get_long_term(SID2)[0]["value"] == "light"

    def test_empty_session_returns_empty_list(self, mm):
        assert mm.get_long_term("nonexistent") == []


# ── Clear ─────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_short_keeps_long(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "hi", "assistant": "hello"})
        mm.add_long_term(SID, {"topic": "pref", "value": "dark mode"})
        mm.clear(SID, "short")
        assert mm.get_short_term(SID) == []
        assert len(mm.get_long_term(SID)) == 1

    def test_clear_long_keeps_short(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "hi", "assistant": "hello"})
        mm.add_long_term(SID, {"topic": "pref", "value": "dark mode"})
        mm.clear(SID, "long")
        assert len(mm.get_short_term(SID)) == 1
        assert mm.get_long_term(SID) == []

    def test_clear_all(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "hi", "assistant": "hello"})
        mm.add_long_term(SID, {"topic": "pref", "value": "dark mode"})
        mm.clear(SID, "all")
        assert mm.get_short_term(SID) == []
        assert mm.get_long_term(SID) == []

    def test_clear_only_affects_target_session(self, mm):
        mm.add_short_term(SID,  {"role": "exchange", "user": "a", "assistant": "b"})
        mm.add_short_term(SID2, {"role": "exchange", "user": "c", "assistant": "d"})
        mm.clear(SID, "all")
        assert mm.get_short_term(SID)  == []
        assert len(mm.get_short_term(SID2)) == 1


# ── Utility ───────────────────────────────────────────────────────────────────

class TestUtility:
    def test_list_sessions(self, mm):
        mm.add_short_term(SID,  {"role": "exchange", "user": "a", "assistant": "b"})
        mm.add_long_term(SID2, {"topic": "t", "value": "v"})
        sessions = mm.list_sessions()
        assert SID  in sessions
        assert SID2 in sessions

    def test_session_summary(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "a", "assistant": "b"})
        mm.add_long_term(SID,  {"topic": "t", "value": "v"})
        summary = mm.session_summary(SID)
        assert summary["session_id"]  == SID
        assert summary["short_count"] == 1
        assert summary["long_count"]  == 1
        assert "db_path" in summary

    def test_db_stats(self, mm):
        mm.add_short_term(SID, {"role": "exchange", "user": "a", "assistant": "b"})
        mm.add_long_term(SID,  {"topic": "t", "value": "v"})
        stats = mm.db_stats()
        assert stats["short_term_rows"] == 1
        assert stats["long_term_rows"]  == 1
        assert "db_size_kb" in stats
        assert "db_path"    in stats
