"""Tests for MemoryManager including compatibility wrappers."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from memory.memory_manager import MemoryManager


@pytest.fixture
def mm(tmp_path):
    """Create a MemoryManager with temp storage."""
    return MemoryManager(storage_dir=str(tmp_path))


class TestCoreAPI:
    def test_store_and_get_short_term(self, mm):
        mm.store_short(role="user", content="hello")
        mm.store_short(role="assistant", content="hi there")
        result = mm.get_short_term(last_n=5)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["content"] == "hi there"

    def test_get_context_string(self, mm):
        mm.store_short(role="user", content="test message")
        ctx = mm.get_context_string(last_n=5)
        assert "test message" in ctx

    def test_store_long_and_recall(self, mm):
        mm.store_long(content="important fact", metadata={"tag": "test"})
        results = mm.recall(query="important", top_k=5)
        assert len(results) >= 1
        assert "important fact" in str(results)

    def test_clear_short(self, mm):
        mm.store_short(role="user", content="temp")
        mm.clear(scope="short")
        assert len(mm.get_short_term(last_n=100)) == 0

    def test_clear_all(self, mm):
        mm.store_short(role="user", content="temp")
        mm.store_long(content="perm", metadata={})
        mm.clear(scope="all")
        assert len(mm.get_short_term(last_n=100)) == 0

    def test_stats(self, mm):
        s = mm.stats()
        assert "short_term_count" in s
        assert "long_term_count" in s

    def test_flush(self, mm):
        mm.store_short(role="user", content="data")
        mm.flush()  # Should not raise


class TestCompatibilityWrappers:
    def test_get_context_for_session(self, mm):
        mm.store_short(role="user", content="session msg")
        ctx = mm.get_context_for_session(session_id="abc123")
        assert "session msg" in ctx

    def test_store_interaction(self, mm):
        mm.store_interaction(session_id="s1", role="user", content="hi")
        result = mm.get_short_term(last_n=5)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hi"

    def test_get_long_term(self, mm):
        mm.store_long(content="memory1", metadata={})
        mm.store_long(content="memory2", metadata={})
        result = mm.get_long_term(session_id="s1", top_k=10)
        assert len(result) >= 2

    def test_get_long_term_empty(self, mm):
        result = mm.get_long_term()
        assert result == []