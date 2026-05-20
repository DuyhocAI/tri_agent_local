"""
HERMES AGENT — core/orchestrator.py  (v6 / Hermes P1-P2)

Refactored from Trinity v5 monolith into Hermes multi-agent architecture:
  - ToolResult, streaming helpers, BaseAgent  → agents/base_agent.py
  - Skill definitions (YAML)                  → skills/
  - SkillRegistry (metadata + dispatch)       → skills/registry.py
  - Orchestrator now inherits BaseAgent and uses SkillRegistry

Specialist agents (Builder/Writer/Analyst) are in agents/ and inherit BaseAgent.
"""

import json
import logging
import os
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import requests

from config import (
    OLLAMA_URL,
    MODEL_CANDIDATES,
    PRIMARY_OPTIONS,
    REVIEWER_OPTIONS,
    SUPERVISOR_OPTIONS,
    RESOURCE_LIMITS,
)
from agents.base_agent import (
    ToolResult,
    OllamaUnavailableError,
    BaseAgent,
    _stream_chat,
    _collect,
    _parse_json,
    _parse_tool_calls,
    _strip_tool_tags,
    _TOOL_RE,
    _MAX_TOOL_ROUNDS,
    _TOOLS_DESCRIPTION,
)
from agents.builder_agent import BuilderAgent
from agents.writer_agent  import WriterAgent
from agents.analyst_agent import AnalystAgent

logger = logging.getLogger("hermes.orchestrator")

# ── Ollama availability ───────────────────────────────────────────────────────

def _ollama_is_up() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# Resolved at boot — populated by _resolve_models() in _boot thread
MODELS: dict[str, str] = {
    "primary":    "qwen3:latest",
    "reviewer":   "mistral:v0.3",
    "supervisor": "mistral:v0.3",
}

_TIMEOUT = RESOURCE_LIMITS["request_timeout_s"]


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS  (tool implementations remain here; SkillRegistry wraps them)
# ══════════════════════════════════════════════════════════════════════════════

def _tool_read_file(path: str) -> ToolResult:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return ToolResult(False, f"File not found: {path}")
        if p.stat().st_size > 2_000_000:
            return ToolResult(False, f"File too large (>{2}MB): {path}")
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 4000:
            content = content[:4000] + f"\n... [truncated, total {p.stat().st_size} bytes]"
        return ToolResult(True, content)
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, str(e))


def _tool_write_file(path: str, content: str) -> ToolResult:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Written {len(content)} chars to {p}")
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, str(e))


def _tool_edit_file(path: str, old_text: str, new_text: str) -> ToolResult:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return ToolResult(False, f"File not found: {path}")
        content = p.read_text(encoding="utf-8", errors="replace")
        if old_text not in content:
            return ToolResult(False,
                f"Text not found in {path}. Use read_file first to verify the exact text.")
        count   = content.count(old_text)
        updated = content.replace(old_text, new_text, 1)
        p.write_text(updated, encoding="utf-8")
        return ToolResult(True,
            f"Replaced 1 occurrence (of {count}) in {p}. "
            f"File is now {len(updated)} chars.")
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, str(e))


def _tool_append_file(path: str, content: str) -> ToolResult:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return ToolResult(True, f"Appended {len(content)} chars to {p}")
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, str(e))


def _tool_delete_file(path: str) -> ToolResult:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return ToolResult(False, f"Not found: {path}")
        if p.is_dir():
            shutil.rmtree(p)
            return ToolResult(True, f"Directory deleted: {p}")
        p.unlink()
        return ToolResult(True, f"File deleted: {p}")
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, str(e))


def _tool_list_dir(path: str) -> ToolResult:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return ToolResult(False, f"Path not found: {path}")
        if not p.is_dir():
            return ToolResult(False, f"Not a directory: {path}")
        entries = []
        for item in sorted(p.iterdir()):
            kind = "DIR " if item.is_dir() else "FILE"
            size = ""
            if item.is_file():
                try:
                    size = f"  ({item.stat().st_size:,} bytes)"
                except Exception:
                    pass
            entries.append(f"  {kind}  {item.name}{size}")
        if not entries:
            return ToolResult(True, f"Directory is empty: {p}")
        return ToolResult(True, f"Contents of {p}:\n" + "\n".join(entries))
    except PermissionError:
        return ToolResult(False, f"Permission denied: {path}")
    except Exception as e:
        return ToolResult(False, str(e))


def _tool_file_exists(path: str) -> ToolResult:
    p     = Path(path).expanduser()
    exists = p.exists()
    kind   = "directory" if p.is_dir() else "file" if p.is_file() else "not found"
    return ToolResult(exists, f"{path} → {kind}")


def _fetch_page_text(url: str, char_limit: int = 2000) -> str:
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header",
                              "footer", "aside", "form", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
        lines  = [l.strip() for l in text.splitlines() if l.strip()]
        result = "\n".join(lines)
        if len(result) > char_limit:
            result = result[:char_limit] + "\n… [truncated]"
        return result
    except Exception as e:
        return f"[fetch error: {e}]"


def _tool_web_search(q: str) -> ToolResult:
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": q, "b": "", "kl": "en-us"},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()

        result_urls:     list[str] = []
        result_titles:   list[str] = []
        result_snippets: list[str] = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for res in soup.select(".result")[:6]:
                a       = res.select_one(".result__title a")
                snippet = res.select_one(".result__snippet")
                if not a:
                    continue
                href = str(a.get("href", "") or "")
                if "uddg=" in href:
                    href = urllib.parse.unquote(
                        href.split("uddg=")[-1].split("&")[0]
                    )
                if href.startswith("http"):
                    result_urls.append(href)
                    result_titles.append(a.get_text(strip=True))
                    result_snippets.append(
                        snippet.get_text(strip=True) if snippet else ""
                    )
        except ImportError:
            for m in re.finditer(r"uddg=([^&\"]+)", resp.text):
                url = urllib.parse.unquote(m.group(1))
                if url.startswith("http") and url not in result_urls:
                    result_urls.append(url)
                    result_titles.append(url)
                    result_snippets.append("")
                if len(result_urls) >= 6:
                    break

        if not result_urls:
            return ToolResult(False, f"No results found for: {q}")

        header_lines = [f"Web search results for: {q}\n"]
        for i, (title, snippet, url) in enumerate(
            zip(result_titles, result_snippets, result_urls), 1
        ):
            header_lines.append(f"[{i}] {title}\n    {snippet}\n    {url}")
        header = "\n".join(header_lines[:5])

        page_blocks: list[str] = []
        for url in result_urls[:2]:
            text = _fetch_page_text(url, char_limit=1500)
            if not text.startswith("[fetch error"):
                page_blocks.append(f"─── {url} ───\n{text}")

        combined = header
        if page_blocks:
            combined += "\n\n" + "\n\n".join(page_blocks)
        if len(combined) > 4000:
            combined = combined[:4000] + "\n… [truncated]"
        return ToolResult(True, combined)

    except Exception as e:
        return ToolResult(False, f"Search failed: {e}")


def _tool_fetch_url(url: str) -> ToolResult:
    text = _fetch_page_text(url, char_limit=4000)
    if text.startswith("[fetch error"):
        return ToolResult(False, text)
    return ToolResult(True, f"Content from {url}:\n{text}")


def _tool_get_datetime() -> ToolResult:
    now = datetime.now()
    return ToolResult(True, (
        f"Current datetime: {now.strftime('%A, %B %d, %Y  %H:%M:%S')}\n"
        f"ISO format: {now.isoformat()}\n"
        f"Weekday: {now.strftime('%A')}\n"
        f"Unix timestamp: {int(now.timestamp())}"
    ))


def _tool_run_command(cmd: str, cwd: str | None = None, timeout: int = 60) -> ToolResult:
    import subprocess as _sp
    _SAFE_PREFIXES = (
        "python ", "python3 ", "pytest", "pip install",
        "pip3 install", "node ", "npm ", "npm test",
        "npm run", "npx ", "node --check",
        "flask ", "uvicorn ", "curl ",
    )
    cmd_stripped = cmd.strip()
    if not any(cmd_stripped.startswith(p) for p in _SAFE_PREFIXES):
        return ToolResult(False,
            f"Command blocked for safety: '{cmd_stripped}'. "
            f"Allowed prefixes: {list(_SAFE_PREFIXES)}")
    try:
        r = _sp.run(
            cmd_stripped, shell=True,
            capture_output=True, text=True,
            timeout=timeout,
            cwd=cwd,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if len(out) > 3000:
            out = out[:3000] + "\n... [truncated]"
        success = r.returncode == 0
        return ToolResult(success,
            f"[exit {r.returncode}]\n{out}" if out else f"[exit {r.returncode}] (no output)")
    except Exception as e:
        return ToolResult(False, f"Command failed: {e}")


# ── Tool dispatcher ───────────────────────────────────────────────────────────

_TOOLS: dict = {
    "read_file":    lambda a: _tool_read_file(a["path"]),
    "write_file":   lambda a: _tool_write_file(a["path"], a["content"]),
    "edit_file":    lambda a: _tool_edit_file(a["path"], a["old_text"], a["new_text"]),
    "append_file":  lambda a: _tool_append_file(a["path"], a["content"]),
    "delete_file":  lambda a: _tool_delete_file(a["path"]),
    "list_dir":     lambda a: _tool_list_dir(a["path"]),
    "file_exists":  lambda a: _tool_file_exists(a["path"]),
    "web_search":   lambda a: _tool_web_search(a["q"]),
    "fetch_url":    lambda a: _tool_fetch_url(a["url"]),
    "get_datetime": lambda a: _tool_get_datetime(),
    "run_command":  lambda a: _tool_run_command(
                        a["cmd"], a.get("cwd"), int(a.get("timeout", 60))
                    ),
}

# Only delete_file requires supervisor approval (data loss, no undo).
# run_command is supervisor-gated via is_destructive in the registry YAML.
_DESTRUCTIVE_TOOLS: set[str] = {"delete_file", "run_command"}


def _execute_tool(name: str, args: dict) -> ToolResult:
    if name not in _TOOLS:
        return ToolResult(False, f"Unknown tool: {name}")
    try:
        return _TOOLS[name](args)
    except KeyError as e:
        return ToolResult(False, f"Missing argument for {name}: {e}")
    except Exception as e:
        return ToolResult(False, f"Tool {name} error: {e}")


# ── Model resolution ──────────────────────────────────────────────────────────

def _resolve_models() -> dict[str, str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        raw_names: list[str] = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        logger.error(f"Cannot reach Ollama: {e}")
        return dict(MODELS)

    exact_set: set[str] = set(raw_names)
    base_map: dict[str, str] = {}
    for n in raw_names:
        base = n.split(":")[0]
        if base not in base_map:
            base_map[base] = n

    resolved: dict[str, str] = {}
    for role, candidates in MODEL_CANDIDATES.items():
        chosen = None
        for c in candidates:
            if c in exact_set:
                chosen = c
                break
            base = c.split(":")[0]
            if base in base_map:
                chosen = base_map[base]
                break
        resolved[role] = chosen or MODELS.get(role, candidates[0])
        if not chosen:
            logger.warning(f"Model not found for '{role}'. Tried: {candidates}")
    return resolved


# ══════════════════════════════════════════════════════════════════════════════
#  HERMES ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class Orchestrator(BaseAgent):
    """
    HermesOrchestrator — routes requests to specialist agents, synthesises results.

    Inherits from BaseAgent:
        - Streaming helpers (_stream_chat, _collect)
        - Chunk factories (_c, _sys, _err, _tok_chunk)
        - Agent LED control (_set_agent, _agent_chunk)
        - Supervisor gate (_sup_approve_tool)
        - Tool execution loop (_run_tool_loop)

    run_chat delegates primary generation to specialist agents (Builder/Writer/Analyst).
    run_build stays here — it drives a multi-phase pipeline with its own step loop.
    """

    def __init__(self, socketio, monitor):
        super().__init__(socketio, monitor)

        # Set BaseAgent required attributes
        self.models               = dict(MODELS)      # will be updated by _boot
        self._primary_options     = PRIMARY_OPTIONS
        self._supervisor_options  = SUPERVISOR_OPTIONS

        # Pipeline state
        self._feedback: dict[str, list[str]] = {}
        self._active:   dict[str, bool]      = {}

        # Build SkillRegistry backed by the legacy _TOOLS dict
        from skills.registry import SkillRegistry
        self.skill_registry = SkillRegistry(
            skill_dirs=[Path(__file__).parent.parent / "skills"],
            tools_dict=_TOOLS,
            destructive_set=_DESTRUCTIVE_TOOLS,
        )

        # Resource guard (VRAM/RAM watchdog + audio monitor)
        from core.resource_guard import get_guard
        self._resource_guard = get_guard()

        # Specialist agents — models/options/registry synced after _boot resolves them
        self._builder  = BuilderAgent(socketio, monitor)
        self._writer   = WriterAgent(socketio, monitor)
        self._analyst  = AnalystAgent(socketio, monitor)

        threading.Thread(
            target=self._boot, daemon=True, name="hermes-model-resolver"
        ).start()

    def _boot(self):
        global MODELS
        time.sleep(1.5)
        resolved = _resolve_models()
        MODELS = resolved
        self.models = resolved   # keep BaseAgent in sync

        # Propagate resolved models and shared resources to specialist agents
        for specialist in (self._builder, self._writer, self._analyst):
            specialist.models              = self.models
            specialist._primary_options    = self._primary_options
            specialist._supervisor_options = self._supervisor_options
            specialist.skill_registry      = self.skill_registry

        print()
        print("  ── Hermes model resolution ───────────────────")
        for role, name in self.models.items():
            print(f"  ✓  {role:12s} → {name}")
        print(f"  ✓  skills       → {len(self.skill_registry)} loaded")
        print("  ─────────────────────────────────────────────")
        print()

    def _get_specialist(self, role: str) -> BaseAgent:
        return {
            "builder":  self._builder,
            "writer":   self._writer,
            "analyst":  self._analyst,
        }.get(role, self._builder)

    # ── Agent routing (P2 seed) ───────────────────────────────────────────────

    _BUILDER_KEYWORDS = frozenset([
        "code", "build", "implement", "debug", "deploy", "script",
        "function", "class", "refactor", "fix", "install", "program",
        "develop", "create", "write", "make",
    ])
    _ANALYST_KEYWORDS = frozenset([
        "analyse", "analyze", "data", "chart", "graph", "statistics",
        "market", "price", "trend", "report", "metric", "calculate",
        "forecast", "predict",
    ])
    _WRITER_KEYWORDS = frozenset([
        "write an article", "blog post", "essay", "email draft",
        "social media", "caption", "brand", "content", "copy",
        "newsletter", "announcement",
    ])

    def route(self, message: str) -> str:
        """Return the name of the specialist agent best suited for this message."""
        msg = message.lower()
        # Analyst keywords are checked before builder to catch data requests
        if any(kw in msg for kw in self._ANALYST_KEYWORDS):
            return "analyst"
        if any(kw in msg for kw in self._WRITER_KEYWORDS):
            return "writer"
        if any(kw in msg for kw in self._BUILDER_KEYWORDS):
            return "builder"
        return "builder"  # default

    # ─────────────────────────────────────────────────────────────────────────
    #  CHAT MODE
    # ─────────────────────────────────────────────────────────────────────────

    def run_chat(self, user_message: str, session_id: str, memory_manager,
                 hermes_memory=None):
        """
        Full pipeline:
          1. Reviewer  → strategy + token_budget (short JSON)
          2. Primary   → stream answer + tool loop  [routed to specialist]
          3. Supervisor→ quality gate
          4. Reviewer  → memory update
        """
        if not _ollama_is_up():
            yield self._err(
                f"Ollama is not running.\n"
                f"Start it with:  ollama serve\n"
                f"Then try again. (Could not connect to {OLLAMA_URL})",
                0,
            )
            return

        self.monitor.reset_tokens()
        _rg_ok, _rg_msg = self._resource_guard.check_resources_ok()
        if not _rg_ok:
            yield self._c("supervisor", f"⚠ Resource warning: {_rg_msg}", 2)

        # ── Step 1: Reviewer — strategy + budget (computed locally, no LLM call) ──
        yield self._sys("Reviewer analysing request…", 5)
        yield self._agent_chunk("reviewer", "active")

        short_mem  = memory_manager.get_short_term(session_id)
        long_mem   = memory_manager.get_long_term(session_id)
        user_facts = hermes_memory.get_user_facts() if hermes_memory else []

        _np         = PRIMARY_OPTIONS.get("num_predict", -1)
        _budget_cap = _np if _np > 0 else 128_000

        _is_code = any(kw in user_message.lower() for kw in (
            "code", "write", "implement", "build", "leetcode",
            "algorithm", "function", "class", "script", "program",
            "create", "make", "develop", "fix", "debug", "refactor",
        ))
        strategy     = "code" if _is_code else "direct"
        token_budget = _budget_cap if _is_code else max(300, min(512, _budget_cap))
        agent_role   = self.route(user_message)

        yield self._agent_chunk("reviewer", "idle")

        agent_display = {"builder": "Builder", "writer": "Writer",
                         "analyst": "Analyst"}.get(agent_role, "Builder")

        yield self._c("reviewer",
                      f"Strategy: {strategy}  |  Budget: {token_budget} tokens  "
                      f"|  Agent: {agent_display}", 15)

        # ── Step 2: Primary — answer with tool loop ───────────────────────────
        yield self._sys(f"{agent_display} generating response…", 22)
        yield self._agent_chunk("primary", "active")

        specialist    = self._get_specialist(agent_role)
        system_prompt = specialist.get_system_prompt(token_budget, long_mem, user_facts)
        ctx = [{"role": "system", "content": system_prompt}]
        for m in short_mem[-(RESOURCE_LIMITS["max_context_msgs"]):]:
            if isinstance(m, dict) and "user" in m and "assistant" in m:
                ctx.append({"role": "user",      "content": m["user"]})
                ctx.append({"role": "assistant",  "content": m["assistant"]})
        ctx.append({"role": "user", "content": user_message})

        self.monitor.add_tokens(sum(max(1, len(m["content"]) // 4) for m in ctx), 0)

        full_response = ""
        for item, sentinel in specialist.run(ctx, 22, 70):
            if sentinel is not None:
                full_response = sentinel
            elif item is not None:
                yield item

        yield self._agent_chunk("primary", "idle")

        # ── Step 3: Supervisor — quality check (skipped for direct/short responses) ──
        sup = {"approved": True, "score": 85, "issues": [], "refinement": ""}
        if strategy == "code":
            yield self._sys("Supervisor evaluating quality…", 74)
            yield self._agent_chunk("supervisor", "active")

            sup_prompt = _p_supervisor_chat(user_message, full_response)
            self.monitor.add_tokens(max(1, len(sup_prompt) // 4), 0)

            sup_text = '{"approved":true,"score":85,"issues":[],"refinement":""}'
            try:
                sup_text = _collect(
                    self.models["supervisor"], sup_prompt, SUPERVISOR_OPTIONS, self.monitor,
                    role="supervisor",
                )
            except Exception as e:
                yield self._err(f"Supervisor error: {e}", 74)

            yield self._agent_chunk("supervisor", "idle")

            sup = _parse_json(sup_text, sup)
            if sup.get("issues") and isinstance(sup["issues"], list):
                yield self._c("supervisor", "Issues: " + "; ".join(sup["issues"]), 78)

            if not sup.get("approved", True) and sup.get("refinement"):
                yield self._c("supervisor", f"Refining: {sup['refinement']}", 80)
                yield self._agent_chunk("primary", "active")
                refine_ctx = ctx + [
                    {"role": "assistant", "content": _strip_tool_tags(full_response)},
                    {"role": "user",      "content": f"Please improve: {sup['refinement']}"},
                ]
                full_response = ""
                for item, sentinel in specialist.run(refine_ctx, 80, 92):
                    if sentinel is not None:
                        full_response = sentinel
                    elif item is not None:
                        yield item
                yield self._agent_chunk("primary", "idle")

        # ── Step 4: Reviewer — memory update ─────────────────────────────────
        yield self._sys("Updating memory…", 93)
        yield self._agent_chunk("reviewer", "active")

        clean_response = _strip_tool_tags(full_response)

        memory_manager.add_short_term(session_id, {
            "role":      "exchange",
            "user":      user_message,
            "assistant": clean_response,
            "ts":        datetime.now().isoformat(),
        })

        mem_prompt = _p_memory_update(user_message, clean_response)
        self.monitor.add_tokens(max(1, len(mem_prompt) // 4), 0)
        try:
            mem_text = _collect(
                self.models["reviewer"], mem_prompt, REVIEWER_OPTIONS, self.monitor,
                role="reviewer",
            )
            mem = _parse_json(mem_text, {})
            if mem.get("topic") and mem.get("value"):
                memory_manager.add_long_term(session_id, mem)
                if hermes_memory:
                    hermes_memory.upsert_user_fact(
                        mem["topic"], mem["value"], session_id=session_id
                    )
        except Exception as e:
            logger.debug(f"Memory update failed: {e}")

        yield self._agent_chunk("reviewer", "idle")

        # Compress session into a cross-session episode summary
        if hermes_memory:
            _rev_model      = self.models.get("reviewer", "")
            _summary_opts   = {**REVIEWER_OPTIONS, "num_predict": 512}
            _mon            = self.monitor
            threading.Thread(
                target=hermes_memory.compress_session,
                args=(session_id, memory_manager,
                      lambda p: _collect(_rev_model, p, _summary_opts, _mon)),
                daemon=True,
            ).start()

        yield {
            "role":             "done",
            "text":             "",
            "progress":         100,
            "tokens":           self.monitor.get_token_snapshot(),
            "supervisor_score": sup.get("score", 85),
            "session_id":       session_id,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  BUILD MODE
    # ─────────────────────────────────────────────────────────────────────────

    def run_build(
        self,
        project_request: str,
        output_path: str,
        session_id: str,
        memory_manager,
        hermes_memory=None,
    ):
        self._feedback[session_id] = []
        self._active[session_id]   = True

        if not _ollama_is_up():
            yield self._err(
                f"Ollama is not running — start it with: ollama serve\n"
                f"(Could not connect to {OLLAMA_URL})",
                0,
            )
            self._active[session_id] = False
            return

        self.monitor.reset_tokens()
        _rg_ok, _rg_msg = self._resource_guard.check_resources_ok()
        if not _rg_ok:
            yield self._c("supervisor", f"⚠ Resource warning: {_rg_msg}", 2)

        output_dir = Path(output_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        built_files: list[str] = []

        # Phase 1: Supervisor safety
        yield self._sys("Supervisor: safety check…", 3)
        yield self._agent_chunk("supervisor", "active")

        safety_prompt = _p_safety(project_request, str(output_dir))
        self.monitor.add_tokens(max(1, len(safety_prompt) // 4), 0)
        safety = {"safe": True, "warnings": []}
        try:
            st = _collect(
                self.models["supervisor"], safety_prompt, SUPERVISOR_OPTIONS, self.monitor,
                role="supervisor",
            )
            safety = _parse_json(st, safety)
        except Exception as e:
            yield self._err(f"Supervisor unavailable: {e}", 3)

        yield self._agent_chunk("supervisor", "idle")
        if not safety.get("safe", True):
            yield self._err(f"Safety blocked: {safety.get('reason','')}", 3)
            return
        for w in safety.get("warnings", []):
            yield self._c("supervisor", f"⚠ {w}", 4)

        # Phase 2: Primary builds the plan
        yield self._sys("Planning project structure…", 8)
        yield self._agent_chunk("primary", "active")

        plan_prompt = _p_build_plan(project_request)
        self.monitor.add_tokens(max(1, len(plan_prompt) // 4), 0)
        plan = {"steps": [{"name": "Generate project",
                           "desc": project_request, "files": []}],
                "tech_stack": "auto"}
        try:
            pt = _collect(self.models["primary"], plan_prompt, {
                **PRIMARY_OPTIONS,
                "num_predict": 2048,
                "keep_alive": "0",
                "temperature": 0.2,
            }, self.monitor, role="primary")
            plan = _parse_json(pt, plan)
        except Exception as e:
            yield self._err(f"Planning error: {e}", 8)

        yield self._agent_chunk("primary", "idle")

        steps = plan.get("steps", [])
        total = max(len(steps), 1)
        yield self._c("plan", json.dumps(plan), 12, total_steps=total)
        yield self._sys(f"Plan: {total} step(s) | Stack: {plan.get('tech_stack','auto')}", 13)

        # Phase 3: Primary executes each step
        base_pct: int = 15
        step: dict = {}
        test_results: list = []
        for idx, step in enumerate(steps):
            if not self._active.get(session_id, True):
                yield self._sys("Build cancelled.", -1)
                return

            base_pct = 15 + int(idx / total * 65)
            yield self._sys(f"Step {idx+1}/{total}: {step.get('name','...')}", base_pct)

            pending = self._feedback.pop(session_id, [])
            fb_ctx  = ("\n\nUser feedback:\n" + "\n".join(f"- {f}" for f in pending)
                       if pending else "")

            for rel in step.get("files", []):
                tgt = output_dir / rel
                if tgt.exists():
                    yield self._c("supervisor", f"Backing up: {rel}", base_pct)
                    try:
                        tgt.with_suffix(tgt.suffix + ".bak").write_bytes(tgt.read_bytes())
                    except Exception:
                        pass

            yield self._agent_chunk("primary", "active")
            step_prompt = _p_build_step(project_request, step, plan, built_files, fb_ctx)
            self.monitor.add_tokens(max(1, len(step_prompt) // 4), 0)

            step_output = ""
            try:
                _t0 = time.time(); _e_tok = 0
                for delta, done, p_tok, e_tok in _stream_chat(
                    self.models["primary"],
                    [{"role": "user", "content": step_prompt}],
                    PRIMARY_OPTIONS,
                ):
                    if delta:
                        step_output += delta
                        self.monitor.add_tokens(0, max(1, len(delta) // 4))
                        yield self._tok_chunk("building", delta, base_pct + 2)
                    if done:
                        if p_tok:
                            self.monitor.add_tokens(p_tok, 0)
                        _e_tok = e_tok
                        break
                from core.metrics import log_request
                log_request("primary", self.models["primary"],
                            round((time.time() - _t0) * 1000, 1), _e_tok, success=True)
            except Exception as e:
                yield self._err(f"Step {idx+1} error: {e}", base_pct)
                yield self._agent_chunk("primary", "idle")
                continue

            yield self._agent_chunk("primary", "idle")

            written = _write_files(step_output, output_dir)
            built_files.extend(written)
            if written:
                yield self._c("files", json.dumps(written), base_pct + 5)

            yield self._agent_chunk("supervisor", "active")
            sq_prompt = _p_supervisor_step(step, step_output, written)
            self.monitor.add_tokens(max(1, len(sq_prompt) // 4), 0)
            sq = {"complete": True, "score": 80, "issue": ""}
            try:
                st2 = _collect(
                    self.models["supervisor"], sq_prompt, SUPERVISOR_OPTIONS, self.monitor,
                    role="supervisor",
                )
                sq = _parse_json(st2, sq)
            except Exception:
                pass
            yield self._agent_chunk("supervisor", "idle")

            if not sq.get("complete", True) and sq.get("issue"):
                yield self._c("supervisor",
                               f"Quality issue: {sq['issue']} — retrying…", base_pct + 3)
                yield self._agent_chunk("primary", "active")
                retry_prompt = (
                    step_prompt +
                    f"\n\nFix this problem: {sq['issue']}\nRewrite affected files completely."
                )
                self.monitor.add_tokens(max(1, len(retry_prompt) // 4), 0)
                retry_out = ""
                try:
                    _t0 = time.time(); _e_tok = 0
                    for delta, done, p_tok, e_tok in _stream_chat(
                        self.models["primary"],
                        [{"role": "user", "content": retry_prompt}],
                        PRIMARY_OPTIONS,
                    ):
                        if delta:
                            retry_out += delta
                            self.monitor.add_tokens(0, max(1, len(delta) // 4))
                        if done:
                            if p_tok:
                                self.monitor.add_tokens(p_tok, 0)
                            _e_tok = e_tok
                            break
                    built_files.extend(_write_files(retry_out, output_dir))
                    from core.metrics import log_request
                    log_request("primary", self.models["primary"],
                                round((time.time() - _t0) * 1000, 1), _e_tok, success=True)
                except Exception as e:
                    yield self._err(f"Retry error: {e}", base_pct + 3)

        # Phase 3.5: Smoke-test loop
        MAX_FIX_ROUNDS = 3
        if built_files:
            for fix_round in range(MAX_FIX_ROUNDS):
                round_label = f"Round {fix_round + 1}/{MAX_FIX_ROUNDS}"
                yield self._sys(f"Smoke testing built files… ({round_label})", base_pct + 6)
                yield self._agent_chunk("supervisor", "active")
                test_results = _smoke_test(built_files, output_dir)
                for tr in test_results:
                    status = "✓" if tr["ok"] else "✗"
                    yield self._c(
                        "supervisor" if tr["ok"] else "error",
                        f"{status} [{tr['file']}] {tr['result']}",
                        base_pct + 7,
                    )
                yield self._agent_chunk("supervisor", "idle")

                failures = [t for t in test_results if not t["ok"]]
                if not failures:
                    yield self._sys(f"✓ All smoke tests passed ({round_label})", base_pct + 8)
                    break

                fail_summary = "; ".join(f"{t['file']}: {t['result']}" for t in failures)
                yield self._c("supervisor",
                    f"⚠ {len(failures)} test failure(s) — fix pass {round_label}", base_pct + 8)

                file_contexts = []
                for t in failures:
                    fp = output_dir / t["file"]
                    if fp.exists():
                        try:
                            content = fp.read_text(encoding="utf-8", errors="replace")
                            if len(content) > 3000:
                                content = content[:3000] + "\n... [truncated]"
                            file_contexts.append(f'=== {t["file"]} ===\n{content}')
                        except Exception:
                            pass

                ctx_block = "\n\n".join(file_contexts) if file_contexts else "(could not read files)"
                yield self._agent_chunk("primary", "active")
                fix_prompt = (
                    f"PROJECT: {project_request}\n"
                    f"STACK: {plan.get('tech_stack', 'auto')}\n\n"
                    f"SMOKE TEST FAILURES that MUST be fixed:\n{fail_summary}\n\n"
                    f"CURRENT CONTENT OF FAILING FILES:\n{ctx_block}\n\n"
                    f"Files written so far:\n" +
                    "\n".join(f"  - {f}" for f in built_files[-12:]) +
                    "\n\nFix ALL failures. Rewrite each failing file completely with correct content.\n"
                    "Output ONLY fixed file blocks:\n"
                    '<FILE path="relative/path/file.ext">\ncomplete fixed content\n</FILE>\n'
                    "Zero prose outside FILE tags."
                )
                self.monitor.add_tokens(max(1, len(fix_prompt) // 4), 0)
                fix_out = ""
                try:
                    _t0 = time.time(); _e_tok = 0
                    for delta, done, p_tok, e_tok in _stream_chat(
                        self.models["primary"],
                        [{"role": "user", "content": fix_prompt}],
                        PRIMARY_OPTIONS,
                    ):
                        if delta:
                            fix_out += delta
                            self.monitor.add_tokens(0, max(1, len(delta) // 4))
                            yield self._tok_chunk("building", delta, base_pct + 9)
                        if done:
                            if p_tok:
                                self.monitor.add_tokens(p_tok, 0)
                            _e_tok = e_tok
                            break
                    new_files = _write_files(fix_out, output_dir)
                    if new_files:
                        new_set = set(new_files)
                        built_files = [f for f in built_files if f not in new_set] + new_files
                        yield self._c("files", json.dumps(new_files), base_pct + 10)
                    from core.metrics import log_request
                    log_request("primary", self.models["primary"],
                                round((time.time() - _t0) * 1000, 1), _e_tok, success=True)
                except Exception as e:
                    yield self._err(f"Fix pass error: {e}", base_pct + 9)
                yield self._agent_chunk("primary", "idle")
            else:
                remaining = [t for t in test_results if not t["ok"]]
                yield self._c("supervisor",
                    f"⚠ {len(remaining)} issue(s) remain after {MAX_FIX_ROUNDS} fix rounds — "
                    "project may need manual review.", base_pct + 10)

        # Phase 4: E2E Test Suite
        yield self._sys("Phase 4: Designing & running end-to-end tests…", 82)
        yield self._agent_chunk("primary", "active")

        e2e_system = (
            "You are a QA engineer with access to tools. "
            "Design tests, write them to disk, run them, fix failures, loop until all pass. "
            "Use run_command to actually execute tests — don't just describe them."
        )
        e2e_prompt = _p_e2e_test_design(project_request, plan, built_files, str(output_dir))
        self.monitor.add_tokens(max(1, len(e2e_prompt) // 4), 0)

        e2e_ctx = [
            {"role": "system", "content": e2e_system},
            {"role": "user",   "content": e2e_prompt},
        ]

        e2e_full_output = ""
        for item, sentinel in self._run_tool_loop(e2e_ctx, 82, 90):
            if sentinel is not None:
                e2e_full_output = sentinel
            elif item is not None:
                if item.get("role") == "building":
                    item = dict(item)
                    item["role"] = "supervisor"
                yield item

        yield self._agent_chunk("primary", "idle")

        e2e_verdict = _extract_verdict(e2e_full_output)
        yield self._c(
            "supervisor" if e2e_verdict == "PASS" else "error",
            f"E2E Round 1 — {e2e_verdict}: {e2e_full_output[-400:].strip()}",
            90,
        )

        MAX_E2E_ROUNDS = 2
        for e2e_round in range(1, MAX_E2E_ROUNDS + 1):
            if e2e_verdict == "PASS":
                break
            yield self._sys(f"E2E fix round {e2e_round}/{MAX_E2E_ROUNDS}…", 90 + e2e_round)
            yield self._agent_chunk("primary", "active")

            fix_ctx = [
                {"role": "system", "content": e2e_system},
                {"role": "user",   "content": _p_e2e_fix(
                    project_request, plan, str(output_dir), e2e_full_output, e2e_round
                )},
            ]
            self.monitor.add_tokens(max(1, len(fix_ctx[-1]["content"]) // 4), 0)

            e2e_full_output = ""
            for item, sentinel in self._run_tool_loop(fix_ctx, 90 + e2e_round, 93):
                if sentinel is not None:
                    e2e_full_output = sentinel
                elif item is not None:
                    if item.get("role") == "building":
                        item = dict(item)
                        item["role"] = "supervisor"
                    yield item

            yield self._agent_chunk("primary", "idle")
            e2e_verdict = _extract_verdict(e2e_full_output)
            yield self._c(
                "supervisor" if e2e_verdict == "PASS" else "error",
                f"E2E Round {e2e_round + 1} — {e2e_verdict}: {e2e_full_output[-400:].strip()}",
                93,
            )

        yield self._sys(f"✓ E2E testing complete — Final verdict: {e2e_verdict}", 94)

        # Phase 5: Final supervisor eval
        yield self._sys("Supervisor: final evaluation…", 95)
        yield self._agent_chunk("supervisor", "active")
        final_prompt = _p_final_eval(project_request, built_files)
        self.monitor.add_tokens(max(1, len(final_prompt) // 4), 0)
        final_score = 80
        try:
            ft = _collect(
                self.models["supervisor"], final_prompt, SUPERVISOR_OPTIONS, self.monitor,
                role="supervisor",
            )
            fd = _parse_json(ft, {"score": 80, "issues": [], "summary": ""})
            final_score = fd.get("score", 80)
            if e2e_verdict == "PASS":
                final_score = max(final_score, 88)
            if fd.get("issues"):
                yield self._c("supervisor", "Issues: " + "; ".join(fd["issues"]), 96)
            if fd.get("summary"):
                yield self._c("supervisor", fd["summary"], 97)
        except Exception as e:
            yield self._err(f"Final eval error: {e}", 95)
        yield self._agent_chunk("supervisor", "idle")

        # Phase 6: Save project context to memory
        yield self._sys("Saving project to memory for chat editing…", 97)
        project_summary = (
            f"PROJECT BUILT: {project_request}\n"
            f"OUTPUT DIR: {output_dir}\n"
            f"STACK: {plan.get('tech_stack', 'auto')}\n"
            f"E2E VERDICT: {e2e_verdict}\n"
            f"FILES ({len(built_files)}):\n" +
            "\n".join(f"  {f}" for f in built_files[:20])
        )
        memory_manager.add_short_term(session_id, {
            "role":      "build",
            "user":      f"Build project: {project_request}",
            "assistant": project_summary,
            "ts":        datetime.now().isoformat(),
        })
        memory_manager.add_long_term(session_id, {
            "topic": "last_built_project",
            "value": project_summary,
            "ts":    datetime.now().isoformat(),
        })
        yield self._c("supervisor",
            f"✓ Project saved to memory. Switch to Chat mode and say "
            f"'update the project' or 'add feature X' to continue editing.", 98)

        # Phase 7: agent.html report
        yield self._sys("Writing project report…", 98)
        try:
            html = _build_agent_html(
                project_request, plan, built_files, final_score, e2e_verdict,
                models=self.models,
            )
            ap = output_dir / "agent.html"
            ap.write_text(html, encoding="utf-8")
            built_files.append(str(ap))
        except Exception as e:
            yield self._err(f"agent.html error: {e}", 98)

        self._active[session_id] = False
        yield {
            "role": "build_complete", "text": "", "progress": 100,
            "tokens": self.monitor.get_token_snapshot(),
            "final_score": final_score, "files": built_files,
            "output_path": str(output_dir), "session_id": session_id,
        }

    # ── Build control ─────────────────────────────────────────────────────────

    def inject_feedback(self, session_id: str, feedback: str):
        self._feedback.setdefault(session_id, []).append(feedback)

    def cancel_build(self, session_id: str):
        self._active[session_id] = False


# ══════════════════════════════════════════════════════════════════════════════
#  FILE WRITER  (for build mode)
# ══════════════════════════════════════════════════════════════════════════════

def _write_files(llm_output: str, output_dir: Path) -> list[str]:
    written = []
    pattern = re.compile(r'<FILE\s+path="([^"]+)">(.*?)</FILE>', re.DOTALL)
    base = output_dir.resolve()
    for rel_path, content in pattern.findall(llm_output):
        target = (output_dir / rel_path.lstrip("/\\")).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            logger.warning(f"Blocked path traversal: {rel_path}")
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content.strip(), encoding="utf-8")
            written.append(str(target))
        except Exception as e:
            logger.warning(f"Failed to write {rel_path}: {e}")
    return written


def _extract_verdict(text: str) -> str:
    m = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    text_upper = text.upper()
    passes = text_upper.count("PASSED") + text_upper.count("TESTS_PASSED")
    fails  = text_upper.count("FAILED") + text_upper.count("TESTS_FAILED")
    if passes > 0 and fails == 0:
        return "PASS"
    if fails > 0:
        return "FAIL"
    return "UNKNOWN"


def _smoke_test(files: list[str], output_dir: Path) -> list[dict]:
    import subprocess
    import shutil as _shutil

    results = []
    for fp in files:
        p = Path(fp)
        try:
            rel = str(p.relative_to(output_dir))
        except ValueError:
            rel = p.name

        if not p.exists():
            results.append({"file": rel, "ok": False, "result": "file missing after write"})
            continue
        size = p.stat().st_size
        if size == 0:
            results.append({"file": rel, "ok": False, "result": "file is empty (0 bytes)"})
            continue

        ext = p.suffix.lower()

        if ext == ".py":
            try:
                r = subprocess.run(
                    ["python", "-m", "py_compile", str(p)],
                    capture_output=True, text=True, timeout=10
                )
                if r.returncode != 0:
                    lines = (r.stderr or r.stdout).strip().splitlines()
                    err = lines[-1][:140] if lines else "(no output)"
                    results.append({"file": rel, "ok": False, "result": f"SyntaxError: {err}"})
                    continue
            except Exception as e:
                results.append({"file": rel, "ok": False, "result": f"compile check failed: {e}"})
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                issues = []
                if "pass  # TODO" in content or "raise NotImplementedError" in content:
                    issues.append("contains unimplemented stubs")
                if content.count("def ") > 0 and content.count("return") == 0 and "pass" in content:
                    issues.append("functions may be empty stubs")
                if issues:
                    results.append({"file": rel, "ok": False,
                                    "result": "syntax OK but: " + "; ".join(issues)})
                else:
                    results.append({"file": rel, "ok": True,
                                    "result": f"syntax OK, {size:,} bytes, "
                                              f"{content.count('def ')} functions"})
            except Exception:
                results.append({"file": rel, "ok": True, "result": f"syntax OK ({size:,} bytes)"})

        elif ext == ".js" and _shutil.which("node"):
            try:
                r = subprocess.run(
                    ["node", "--check", str(p)],
                    capture_output=True, text=True, timeout=10
                )
                if r.returncode == 0:
                    results.append({"file": rel, "ok": True, "result": f"syntax OK ({size:,} bytes)"})
                else:
                    lines = (r.stderr or r.stdout).strip().splitlines()
                    err = lines[-1][:140] if lines else "(no output)"
                    results.append({"file": rel, "ok": False, "result": f"JS error: {err}"})
            except Exception as e:
                results.append({"file": rel, "ok": False, "result": f"node check failed: {e}"})

        elif ext == ".json":
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                json.loads(content)
                results.append({"file": rel, "ok": True, "result": f"valid JSON ({size:,} bytes)"})
            except json.JSONDecodeError as e:
                results.append({"file": rel, "ok": False, "result": f"invalid JSON: {e}"})

        elif ext in (".html", ".htm"):
            content = p.read_text(encoding="utf-8", errors="replace").lower()
            missing = []
            if "<!doctype" not in content and "<html" not in content:
                missing.append("<!DOCTYPE> or <html>")
            if "<head" not in content:
                missing.append("<head>")
            if "<body" not in content:
                missing.append("<body>")
            raw = p.read_text(encoding="utf-8", errors="replace")
            if "<!-- TODO" in raw or "YOUR_CONTENT_HERE" in raw:
                missing.append("contains placeholder comments")
            if missing:
                results.append({"file": rel, "ok": False,
                                 "result": "missing: " + ", ".join(missing)})
            else:
                results.append({"file": rel, "ok": True,
                                 "result": f"HTML structure OK ({size:,} bytes)"})

        elif ext == ".css":
            content = p.read_text(encoding="utf-8", errors="replace")
            opens  = content.count("{")
            closes = content.count("}")
            if opens == 0:
                results.append({"file": rel, "ok": False, "result": "CSS file has no rules"})
            elif opens != closes:
                results.append({"file": rel, "ok": False,
                                 "result": f"unbalanced braces: {opens} open, {closes} close"})
            else:
                results.append({"file": rel, "ok": True,
                                 "result": f"CSS OK, {opens} rules ({size:,} bytes)"})

        elif ext in (".md", ".txt", ".rst", ".toml", ".yaml", ".yml",
                     ".ini", ".cfg", ".env"):
            results.append({"file": rel, "ok": True, "result": f"present, {size:,} bytes"})

        else:
            results.append({"file": rel, "ok": True, "result": f"present, {size:,} bytes"})

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _p_supervisor_chat(user_msg: str, response: str) -> str:
    return (
        f"User: {user_msg[:200]}\nResponse (first 600 chars): {response[:600]}\n\n"
        "Evaluate: accuracy, completeness, safety.\n"
        "Reply ONLY with JSON (no prose):\n"
        '{"approved":true,"score":85,"issues":[],"refinement":""}'
    )


def _p_memory_update(user_msg: str, response: str) -> str:
    return (
        f"Conversation:\nUser: {user_msg[:250]}\nAssistant: {response[:250]}\n\n"
        "Did the user reveal their name, a preference, or a persistent fact?\n"
        "If YES reply with ONLY this JSON (fill in real values):\n"
        '{"topic":"TOPIC","value":"VALUE"}\n\n'
        "If NO reply with ONLY: null"
    )


def _p_safety(project: str, out_path: str) -> str:
    return (
        f"Project: {project[:400]}\nOutput dir: {out_path}\n\n"
        "Check for path traversal, malware, system file ops outside output dir.\n"
        "Reply ONLY with JSON:\n"
        '{"safe":true,"warnings":[],"reason":""}'
    )


def _p_e2e_test_design(project: str, plan: dict, files: list, output_dir: str) -> str:
    file_list = "\n".join(f"  - {f}" for f in files[:30])
    stack = plan.get("tech_stack", "auto")
    return (
        f"PROJECT: {project}\n"
        f"TECH STACK: {stack}\n"
        f"OUTPUT DIR: {output_dir}\n"
        f"FILES BUILT:\n{file_list}\n\n"
        "You are a QA engineer. Design and implement a complete end-to-end test suite for this project.\n\n"
        "REQUIREMENTS:\n"
        "1. Analyze what type of project this is (web app, CLI tool, library, API, etc.)\n"
        "2. Install any missing dependencies first using run_command with pip install\n"
        "3. Write actual test files (pytest for Python, or test scripts)\n"
        "4. Run the tests using run_command — use the actual output dir as cwd\n"
        "5. For web apps: test that the server starts, all routes return correct status codes\n"
        "6. For Python libs: test all public functions with real inputs/outputs\n"
        "7. For CLIs: test with sample inputs\n"
        "8. After running tests, report: which passed, which failed, why\n\n"
        "USE TOOLS to:\n"
        "- read_file: inspect source files before writing tests\n"
        "- write_file: write test files to the project directory\n"
        "- run_command: install deps, run tests, check syntax\n"
        "- list_dir: see project structure\n\n"
        "START by listing the project dir, then reading key files, then writing+running tests.\n"
        "Report final result as:\n"
        "TESTS_PASSED: <n>\nTESTS_FAILED: <n>\nFAILURES:\n- <description of each failure>\n"
        "VERDICT: PASS or FAIL"
    )


def _p_e2e_fix(project: str, plan: dict, output_dir: str,
               test_output: str, fix_round: int) -> str:
    stack = plan.get("tech_stack", "auto")
    return (
        f"PROJECT: {project}\n"
        f"TECH STACK: {stack}\n"
        f"OUTPUT DIR: {output_dir}\n"
        f"FIX ROUND: {fix_round}\n\n"
        f"END-TO-END TEST RESULTS (failures to fix):\n{test_output[:2000]}\n\n"
        "You are a senior developer. Fix ALL failing tests.\n\n"
        "PROCESS:\n"
        "1. read_file each failing file to see current content\n"
        "2. edit_file or write_file to fix the issues\n"
        "3. run_command to re-run the specific failing tests\n"
        "4. Repeat until all tests pass\n\n"
        "RULES:\n"
        "- Fix root causes, not symptoms\n"
        "- Don't remove tests to make them pass — fix the code instead\n"
        "- After all fixes, run the FULL test suite one more time\n"
        "- Report final: TESTS_PASSED / TESTS_FAILED / VERDICT: PASS or FAIL"
    )


def _p_build_plan(project: str) -> str:
    return (
        f"Project: {project[:800]}\n\n"
        "You are a senior software architect. Create a COMPLETE, professional build plan.\n\n"
        "REQUIREMENTS:\n"
        "- Plan must produce a FULLY FUNCTIONAL project, not just one file\n"
        "- Every project needs: backend logic, frontend UI (HTML/CSS/JS or framework), "
        "config files, dependency files, and a README\n"
        "- Web apps need: server file, HTML templates or static pages, CSS styling, "
        "JS interactivity, routes for all features\n"
        "- Python projects need: requirements.txt, main entry point, modules, tests/\n"
        "- Break into 4-8 logical steps. Each step builds on the previous.\n"
        "- Typical step order: 1=project structure+config files, 2=backend/core logic, "
        "3=database models if needed, 4=API routes, 5=frontend HTML templates, "
        "6=CSS styling (full design), 7=JavaScript interactivity, 8=tests+README\n\n"
        "Reply ONLY with JSON, no other text:\n"
        '{"tech_stack":"e.g. Flask+SQLite+HTML/CSS/JS","steps":['
        '{"name":"Step name","desc":"Detailed description of exactly what this step builds and why","files":["relative/path.ext","another/file.ext"]}]}'
    )


def _p_build_step(project: str, step: dict, plan: dict, built: list, feedback: str) -> str:
    built_str = "\n".join(f"  - {f}" for f in built[-12:]) or "  (none yet — this is the first step)"
    files_str = "\n".join(f"  - {f}" for f in step.get("files", []))
    return (
        f"PROJECT: {project}\n"
        f"TECH STACK: {plan.get('tech_stack', 'auto')}\n"
        f"TOTAL STEPS IN PLAN: {len(plan.get('steps', []))}\n\n"
        f"CURRENT STEP: {step.get('name')} — {step.get('desc')}\n"
        f"FILES THIS STEP MUST CREATE:\n{files_str}\n\n"
        f"FILES ALREADY WRITTEN IN PREVIOUS STEPS:\n{built_str}"
        f"{feedback}\n\n"
        "MANDATORY RULES:\n"
        "1. Write COMPLETE, production-ready content for EVERY file listed above — no placeholders, no TODOs, no truncation\n"
        "2. Each file must be fully functional and complete — not a skeleton\n"
        "3. Frontend files (HTML) must include: proper DOCTYPE, full <head> with meta tags, "
        "embedded or linked CSS with real styling (colors, layout, responsive design), "
        "all interactive elements wired up with JavaScript\n"
        "4. CSS files must have complete styling: reset, typography, colors, layout, "
        "responsive breakpoints, hover states, animations\n"
        "5. Python files must have all imports, complete functions/classes, proper error handling\n"
        "6. If this step has a server/backend file, include ALL routes needed for the full app\n"
        "7. Each <FILE> block must contain the ENTIRE file content, not a partial\n\n"
        "Output ONLY file blocks, nothing else:\n"
        '<FILE path="relative/path/file.ext">\n'
        "complete file content here\n"
        "</FILE>\n\n"
        "Repeat for every file in this step. Zero prose outside FILE tags."
    )


def _p_supervisor_step(step: dict, output: str, written: list) -> str:
    return (
        f"Step: {step.get('name')} — {step.get('desc')}\n"
        f"Expected: {step.get('files', [])}  Written: {written}\n"
        f"Preview: {output[:400]}\n\n"
        "Reply ONLY with JSON:\n"
        '{"complete":true,"score":80,"issue":""}'
    )


def _p_final_eval(project: str, files: list) -> str:
    return (
        f"Request: {project[:400]}\nFiles: {files}\n\n"
        "Reply ONLY with JSON:\n"
        '{"score":85,"issues":[],"summary":"brief summary"}'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  agent.html GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _build_agent_html(
    project_request: str,
    plan: dict,
    files: list,
    score: int,
    e2e_verdict: str = "UNKNOWN",
    models: dict | None = None,
) -> str:
    if models is None:
        models = MODELS
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sc    = "#39d98a" if score >= 85 else "#f5c842" if score >= 65 else "#f55b5b"
    vc    = "#39d98a" if e2e_verdict == "PASS" else "#f55b5b" if e2e_verdict == "FAIL" else "#f5c842"
    frows = "\n".join(f'<tr><td class="p">{f}</td></tr>' for f in files)
    srows = "\n".join(
        f'<tr><td class="n">{s.get("name","?")}</td><td>{s.get("desc","")}</td></tr>'
        for s in plan.get("steps", [])
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Hermes Agent — Project Report</title>
<style>
:root{{--bg:#07080d;--card:#10121e;--bdr:#1e2235;--acc:#5b6ef5;--txt:#c8cfe8;--mu:#5a6080}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:'Courier New',monospace;padding:2rem;line-height:1.7}}
h1{{color:var(--acc);font-size:1.5rem;letter-spacing:3px;margin-bottom:.25rem}}
h2{{font-size:.75rem;color:var(--mu);letter-spacing:2px;text-transform:uppercase;
    margin:1.5rem 0 .45rem;border-bottom:1px solid var(--bdr);padding-bottom:.3rem}}
.meta{{font-size:.7rem;color:var(--mu);margin-bottom:1.3rem}}
.score{{font-size:3rem;font-weight:900;color:{sc};line-height:1}}
.card{{background:var(--card);border:1px solid var(--bdr);border-radius:6px;padding:1rem 1.2rem;margin-bottom:.8rem}}
.hero{{display:flex;gap:2rem;align-items:flex-start}}
table{{width:100%;border-collapse:collapse;font-size:.8rem}}
td{{padding:.38rem .6rem;border-bottom:1px solid var(--bdr)}}
td.p{{color:#2ee89a;word-break:break-all}} td.n{{color:var(--acc);font-weight:bold;white-space:nowrap;padding-right:1rem;width:180px}}
.badge{{display:inline-block;background:rgba(91,110,245,.18);color:var(--acc);padding:.1rem .55rem;border-radius:3px;font-size:.7rem;margin-right:.3rem}}
</style></head>
<body>
<h1>◈ HERMES AGENT — PROJECT REPORT</h1>
<div class="meta">Generated: {ts}</div>
<div class="card hero">
  <div><div class="score">{score}%</div><div style="font-size:.62rem;color:var(--mu);letter-spacing:2px;margin-top:3px">COMPLETION</div></div>
  <div><div class="score" style="color:{vc};font-size:1.8rem">{e2e_verdict}</div><div style="font-size:.62rem;color:var(--mu);letter-spacing:2px;margin-top:3px">E2E TESTS</div></div>
  <div style="flex:1">
    <h2 style="margin-top:0">Request</h2>
    <div>{project_request}</div>
    <div style="margin-top:.7rem">
      <span class="badge">Primary: {models.get('primary','qwen3')}</span>
      <span class="badge">Reviewer: {models.get('reviewer','qwen3')}</span>
      <span class="badge">Supervisor: {models.get('supervisor','mistral')}</span>
    </div>
  </div>
</div>
<h2>Build Steps — {len(plan.get('steps',[]))}</h2>
<div class="card"><table><tbody>{srows}</tbody></table></div>
<h2>Files Created — {len(files)}</h2>
<div class="card"><table><tbody>{frows}</tbody></table></div>
</body></html>"""
