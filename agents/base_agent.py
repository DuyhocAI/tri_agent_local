"""
Hermes BaseAgent — agents/base_agent.py

Contains all shared infrastructure extracted from the monolithic Orchestrator:
  - ToolResult data class
  - OllamaUnavailableError
  - Ollama streaming helpers (_stream_chat, _collect, _parse_json)
  - Tool-tag parsing utilities (_TOOL_RE, _parse_tool_calls, _strip_tool_tags)
  - BaseAgent class: chunk factories, agent LED control, supervisor gate,
    and the core _run_tool_loop

All specialist agents (BuilderAgent, WriterAgent, AnalystAgent) inherit from BaseAgent.
The legacy Orchestrator class also inherits from BaseAgent for a safe P1 transition.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import requests

from config import OLLAMA_URL, RESOURCE_LIMITS

logger = logging.getLogger("hermes.base_agent")

_TIMEOUT = RESOURCE_LIMITS["request_timeout_s"]
_MAX_TOOL_ROUNDS = 8
_TOOL_RE = re.compile(r"<TOOL>(.*?)</TOOL>", re.DOTALL)


# ── Fundamental types ─────────────────────────────────────────────────────────

class ToolResult:
    """Returned by every tool/skill implementation."""
    def __init__(self, ok: bool, data: str):
        self.ok   = ok
        self.data = data
    def __str__(self):
        return f"[{'OK' if self.ok else 'ERROR'}] {self.data}"


def _make_tool_error(msg: str) -> ToolResult:
    return ToolResult(False, msg)


class OllamaUnavailableError(Exception):
    def __init__(self, msg: str = ""):
        self.msg = msg or (
            f"Ollama is not running — start it with: ollama serve\n"
            f"(tried {OLLAMA_URL})"
        )
        super().__init__(self.msg)
    def __str__(self): return self.msg


# ── Ollama streaming helpers ──────────────────────────────────────────────────

def _stream_chat(model: str, messages: list, options: dict):
    """Yield (delta, done, prompt_tokens, eval_tokens) from Ollama /api/chat.

    Raises OllamaUnavailableError on connection failure.
    """
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "options":    {k: v for k, v in options.items() if k != "keep_alive"},
        "keep_alive": options.get("keep_alive", "0"),
    }
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=_TIMEOUT
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OllamaUnavailableError()
    except requests.exceptions.Timeout:
        raise OllamaUnavailableError("Ollama timed out — model may still be loading")
    for raw in resp.iter_lines():
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        delta = obj.get("message", {}).get("content", "")
        done  = obj.get("done", False)
        p_tok = obj.get("prompt_eval_count", 0) if done else 0
        e_tok = obj.get("eval_count", 0) if done else 0
        yield delta, done, p_tok, e_tok
        if done:
            break


def _collect(model: str, prompt: str, options: dict, monitor=None, role: str = "") -> str:
    """Collect /api/generate streaming response as a single string."""
    _t0 = time.time()
    payload = {
        "model":      model,
        "prompt":     prompt,
        "stream":     True,
        "options":    {k: v for k, v in options.items() if k != "keep_alive"},
        "keep_alive": options.get("keep_alive", "0"),
    }
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate", json=payload, stream=True, timeout=_TIMEOUT
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OllamaUnavailableError()
    except requests.exceptions.Timeout:
        raise OllamaUnavailableError("Ollama timed out — model may still be loading")
    full   = ""
    _e_tok = 0
    for raw in resp.iter_lines():
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        delta = obj.get("response", "")
        done  = obj.get("done", False)
        p_tok = obj.get("prompt_eval_count", 0) if done else 0
        e_tok = obj.get("eval_count", 0) if done else 0
        if delta:
            full += delta
            if monitor:
                monitor.add_tokens(0, max(1, len(delta) // 4))
        if done and monitor and p_tok:
            monitor.add_tokens(p_tok, 0)
        if done:
            _e_tok = e_tok
            break
    if role:
        try:
            from core.metrics import log_request
            log_request(role, model, round((time.time() - _t0) * 1000, 1),
                        _e_tok or max(1, len(full) // 4), success=True)
        except Exception:
            pass
    return full


def _parse_json(text: str, fallback: dict) -> dict:
    """Extract the first JSON object from text. Returns fallback on failure."""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return fallback


# ── Tool-tag utilities ────────────────────────────────────────────────────────

def _parse_tool_calls(text: str) -> list[dict]:
    """Extract all <TOOL>{...}</TOOL> blocks from text."""
    calls = []
    for m in _TOOL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1).strip())
            if "name" in obj:
                calls.append(obj)
        except Exception:
            pass
    return calls


def _strip_tool_tags(text: str) -> str:
    """Remove all <TOOL>...</TOOL> blocks from text."""
    return _TOOL_RE.sub("", text).strip()


# ── Tools description (shared by all specialist agents) ──────────────────────

_TOOLS_DESCRIPTION = """
You have access to the following tools. Emit them inside <TOOL> tags anywhere in your response.

  Read a file:
    <TOOL>{"name":"read_file","args":{"path":"C:/path/to/file.py"}}</TOOL>

  Write / create a file  ← use for NEW files or FULL rewrites:
    <TOOL>{"name":"write_file","args":{"path":"C:/path/to/file.py","content":"...full file content..."}}</TOOL>

  Edit a file  ← PREFERRED for modifying existing code (surgical find-and-replace):
    <TOOL>{"name":"edit_file","args":{"path":"C:/path/to/file.py","old_text":"exact text to find","new_text":"replacement text"}}</TOOL>

  Append to a file  ← add content to end of existing file:
    <TOOL>{"name":"append_file","args":{"path":"C:/path/to/file.py","content":"...text to append..."}}</TOOL>

  Delete a file or folder (requires supervisor approval):
    <TOOL>{"name":"delete_file","args":{"path":"C:/path/to/target"}}</TOOL>

  List directory contents:
    <TOOL>{"name":"list_dir","args":{"path":"C:/some/dir"}}</TOOL>

  Check whether a path exists:
    <TOOL>{"name":"file_exists","args":{"path":"C:/some/path"}}</TOOL>

  Search the web (returns real page content, not just a snippet):
    <TOOL>{"name":"web_search","args":{"q":"your search query"}}</TOOL>

  Fetch a specific URL and read its full text:
    <TOOL>{"name":"fetch_url","args":{"url":"https://example.com/docs/page"}}</TOOL>

  Get current date/time:
    <TOOL>{"name":"get_datetime","args":{}}</TOOL>

  Run a shell command (python, pytest, npm, node — safe prefixes only):
    <TOOL>{"name":"run_command","args":{"cmd":"pytest tests/ -v","cwd":"/path/to/project","timeout":60}}</TOOL>

MANDATORY RULES — follow these exactly:

1. CODE RESPONSES — always save code to disk:
   - NEW file or full rewrite → use write_file with full content.
   - MODIFYING existing code → read_file first, then use edit_file (one surgical change per call).
   - APPENDING to a file → use append_file.
   - After every write/edit/append, tell the user the exact path.
   - Never just print code without also writing it — printing alone is not enough.

2. EDITING WORKFLOW:
   a. read_file to see exact current content.
   b. edit_file with old_text = exact substring to replace, new_text = new code.
   c. Chain multiple edit_file calls for multiple changes in one response.

3. WEB SEARCH: Use web_search for anything current or uncertain, then fetch_url on the best result.

4. CHAINING: Emit multiple <TOOL> tags in one response — they execute in order.

5. FILE OPERATIONS: Always use tools for file/folder tasks. Never guess at file contents.
"""


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Shared infrastructure for all Hermes agents.

    Subclasses must set before calling methods that use LLMs:
        self.models            dict[str, str]  — role → model name
        self._primary_options  dict            — Ollama options for primary role
        self._supervisor_options dict          — Ollama options for supervisor role
        self.skill_registry    SkillRegistry   — for tool dispatch

    These are normally set in Orchestrator.__init__ and in specialist agent __init__.
    """

    def __init__(self, socketio, monitor):
        self.socketio = socketio
        self.monitor  = monitor
        self._agent_states: dict[str, str] = {}
        # Set by subclass __init__
        self.models: dict[str, str] = {}
        self._primary_options: dict = {}
        self._supervisor_options: dict = {}
        self.skill_registry = None  # SkillRegistry injected by subclass

    # ── Chunk factories ───────────────────────────────────────────────────────

    @staticmethod
    def _c(role: str, text: str, progress: int, **kw) -> dict:
        return {"role": role, "text": text, "progress": progress,
                "ts": time.time(), **kw}

    def _sys(self, text: str, progress: int) -> dict:
        return self._c("system", text, progress)

    def _err(self, text: str, progress: int) -> dict:
        logger.warning(f"[error] {text}")
        return self._c("error", text, progress)

    def _tok_chunk(self, role: str, delta: str, progress: int) -> dict:
        return self._c(role, delta, progress,
                       streaming=True,
                       tokens=self.monitor.get_token_snapshot())

    def _set_agent(self, agent: str, state: str) -> None:
        try:
            self.socketio.emit(
                "agent_status", {"agent": agent, "state": state}, namespace="/"
            )
        except Exception as e:
            logger.debug(f"agent_status emit error: {e}")
        self._agent_states[agent] = state

    def _agent_chunk(self, agent: str, state: str) -> dict:
        """Return an SSE chunk that updates an agent LED via the stream."""
        self._agent_states[agent] = state
        try:
            self.socketio.emit(
                "agent_status", {"agent": agent, "state": state}, namespace="/"
            )
        except Exception:
            pass
        return {"role": "agent_status", "agent": agent, "state": state,
                "text": "", "progress": -1, "ts": time.time()}

    # ── Supervisor: destructive tool gate ────────────────────────────────────

    def _sup_approve_tool(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Ask supervisor whether a destructive tool call is safe."""
        if self.skill_registry and not self.skill_registry.is_destructive(tool_name):
            return True, ""
        prompt = (
            f"A tool call is about to execute:\n"
            f"  Tool: {tool_name}\n"
            f"  Args: {json.dumps(args)}\n\n"
            "Is this safe? Does the target look like a system file or important data?\n"
            "Reply ONLY with JSON: "
            '{"approve":true,"reason":""}'
        )
        try:
            result = _collect(
                self.models.get("supervisor", ""),
                prompt,
                self._supervisor_options,
                self.monitor,
                role="supervisor",
            )
            j = _parse_json(result, {"approve": True, "reason": ""})
            return j.get("approve", True), j.get("reason", "")
        except Exception:
            return True, ""  # default allow on supervisor failure

    # ── Core tool execution loop ──────────────────────────────────────────────

    def _run_tool_loop(
        self,
        messages: list,
        progress_start: int,
        progress_end: int,
    ):
        """
        Stream the primary LLM, intercept <TOOL> calls, execute them,
        inject results back into conversation, and continue streaming.

        Uses:
            self.models["primary"], self._primary_options
            self.skill_registry  (for tool dispatch + destructive check)
            self.monitor         (for token accounting)

        Yields (chunk_dict | None, accumulated_full_text).
        Final yield is (None, full_text) as sentinel.
        """
        full_text   = ""
        tool_rounds = 0
        registry    = self.skill_registry

        while tool_rounds <= _MAX_TOOL_ROUNDS:
            chunk_text = ""
            progress = progress_start + int(
                (tool_rounds / (_MAX_TOOL_ROUNDS + 1)) * (progress_end - progress_start)
            )

            # Stream from primary LLM
            try:
                _pending = ""
                _e_tok   = 0
                _t0      = time.time()
                for delta, done, p_tok, e_tok in _stream_chat(
                    self.models.get("primary", ""),
                    messages,
                    self._primary_options,
                ):
                    if delta:
                        chunk_text += delta
                        self.monitor.add_tokens(0, max(1, len(delta) // 4))
                        _pending += delta
                        # Flush safe (non-tool) prefix to UI, suppress <TOOL> blocks
                        while True:
                            open_pos  = _pending.find("<TOOL>")
                            close_pos = _pending.find("</TOOL>")
                            if open_pos == -1:
                                if _pending:
                                    yield self._tok_chunk("assistant", _pending, progress), None
                                    _pending = ""
                                break
                            if open_pos > 0:
                                yield self._tok_chunk(
                                    "assistant", _pending[:open_pos], progress
                                ), None
                                _pending = _pending[open_pos:]
                            if close_pos == -1:
                                break  # tag not yet closed — wait
                            end = _pending.find("</TOOL>") + len("</TOOL>")
                            _pending = _pending[end:]
                    if done:
                        if p_tok:
                            self.monitor.add_tokens(p_tok, 0)
                        _e_tok = e_tok
                        if _pending and not _pending.strip().startswith("<TOOL>"):
                            clean = _TOOL_RE.sub("", _pending).strip()
                            if clean:
                                yield self._tok_chunk("assistant", clean, progress), None
                        break
                try:
                    from core.metrics import log_request
                    log_request("primary", self.models.get("primary", ""),
                                round((time.time() - _t0) * 1000, 1), _e_tok, success=True)
                except Exception:
                    pass
            except Exception as e:
                yield self._err(f"Primary LLM error: {e}", progress), None
                break

            full_text += chunk_text

            tool_calls = _parse_tool_calls(chunk_text)
            if not tool_calls:
                break  # no tools — done

            # Execute each tool call
            tool_results: list[str] = []
            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args", {})

                yield self._c("system", f"🔧 Calling tool: {name}({args})", progress), None

                if registry and registry.is_destructive(name):
                    approved, reason = self._sup_approve_tool(name, args)
                    if not approved:
                        result = ToolResult(False, f"Blocked by supervisor: {reason}")
                        yield self._c(
                            "supervisor", f"Blocked {name}: {reason}", progress
                        ), None
                    else:
                        result = registry.execute(name, args) if registry else ToolResult(
                            False, "No skill registry"
                        )
                else:
                    result = registry.execute(name, args) if registry else ToolResult(
                        False, "No skill registry"
                    )

                status = "✓" if result.ok else "✗"
                yield self._c(
                    "system", f"{status} {name}: {result.data[:120]}", progress
                ), None
                tool_results.append(f"Tool: {name}\nResult: {result}")

            # Inject tool results back into conversation
            tool_context = "\n\n".join(tool_results)
            messages = messages + [
                {"role": "assistant", "content": chunk_text},
                {"role": "user",      "content":
                 f"Tool results:\n{tool_context}\n\nNow continue your response."},
            ]
            tool_rounds += 1

        yield None, full_text  # sentinel
