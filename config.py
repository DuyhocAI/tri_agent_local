"""
HERMES AGENT — config.py
Single source of truth for all hardware and pipeline settings.

Hardware target: RTX 3060 12GB VRAM · 32 GB RAM · (CPU model auto-detected)

Critical constraint: only ONE model can be resident in VRAM at a time.
The orchestrator serialises all LLM calls and uses keep_alive="0" to
evict each model immediately after use, freeing VRAM for the next one.
"""

# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"

# ── Model roster ──────────────────────────────────────────────────────────────
# Each role maps to an ORDERED list of candidate names.
# _resolve_models() in orchestrator.py picks the first one that is actually
# installed, so you can rename or swap models without touching any other file.
#
# GTX 1650 budget:
#   qwen3          ~3-4 GB VRAM   ← fits easily, fast
#   mistral:v0.3   ~4-5 GB VRAM   ← tight but works with num_ctx=2048
#   llama3.3       ~8-9 GB VRAM   ← does NOT fit in 4 GB; runs on CPU = slow
#
# Therefore:
#   primary    = qwen3        (fast, fits, good quality for chat)
#   reviewer   = qwen3        (same model re-used; unloaded between uses)
#   supervisor = mistral:v0.3 (slightly heavier, for final quality gate)

MODEL_CANDIDATES = {
    "primary": [
        "llama3.1:8b",          # INT4 / Q4_K_M — 5GB VRAM, fits RTX 3060 12GB perfectly
        "llama3.1:latest",
        "llama3.1",
        "qwen3:latest", "qwen3",          # fallback nếu llama3.1 chưa install
    ],
    "reviewer": [
        "mistral:v0.3", "mistral:latest", "mistral",
        "qwen3:latest", "qwen3",
    ],
    "supervisor": [
        "qwen3:latest", "qwen3",
        "mistral:v0.3", "mistral:latest", "mistral",
    ],
}

# ── Per-role Ollama options ────────────────────────────────────────────────────
# keep_alive="0"  → unload model from VRAM immediately after the call.
# num_gpu=99      → offload every layer that fits (maximise GPU use).
# num_thread=1    → CRITICAL: 1 CPU thread = GPU handles all inference (95%+ GPU load).
#                   Higher values compete with the GPU driver and drop utilization to ~10%.
# num_ctx         → context window. Smaller = less VRAM.
# num_predict     → max output tokens. Reviewer/supervisor need few tokens.
# temperature     → lower = more deterministic for structured JSON output.
#
# RTX 3060 12GB budget:
#   qwen3          ~6.6 GB VRAM   ← primary worker, full GPU
#   mistral:v0.3   ~6.0 GB VRAM   ← reviewer/supervisor, unloaded between uses

PRIMARY_OPTIONS = {
    "num_gpu":     99,
    "num_thread":  1,          # 1 CPU thread = GPU handles all inference (95%+ GPU load)
    "num_ctx":     8192,       # llama3.1:8b hỗ trợ 128K context nhưng 8192 đủ dùng và tiết kiệm VRAM
    "num_predict": -1,         # unlimited output
    "keep_alive":  "5m",       # giữ model warm — llama3.1 mất ~10s load lại từ đầu
    "temperature": 0.7,
}

REVIEWER_OPTIONS = {
    "num_gpu":     99,
    "num_thread":  1,
    "num_ctx":     2048,       # mistral reviewer — chỉ cần output JSON nhỏ
    "num_predict": 128,
    "keep_alive":  "0",        # evict ngay sau khi dùng
    "temperature": 0.0,        # deterministic JSON
}

SUPERVISOR_OPTIONS = {
    "num_gpu":     99,
    "num_thread":  1,
    "num_ctx":     4096,       # qwen supervisor — cần context lớn hơn để đánh giá
    "num_predict": 512,        # cần output nhiều hơn để cho quality score + nhận xét
    "keep_alive":  "0",        # evict ngay sau khi dùng
    "temperature": 0.3,        # hơi creative để phân tích sâu hơn
}

# ── Resource limits ───────────────────────────────────────────────────────────

RESOURCE_LIMITS = {
    "ram_warn_pct":       80,    # log a warning when RAM > 80%
    "vram_warn_pct":      85,    # log a warning when VRAM > 85%
    "request_timeout_s":  600,   # 10 min — needed for long code generation
    "max_context_msgs":   10,    # how many prior exchanges to include
}

# ── Memory subsystem ──────────────────────────────────────────────────────────

MEMORY_CONFIG = {
    "short_term_max":     30,
    "long_term_max":      200,
    "expire_seconds":     7200,  # 2 h — short-term entries older than this are pruned
}

# ── System monitor ────────────────────────────────────────────────────────────

MONITOR_CONFIG = {
    "push_interval_s":    1.5,   # how often stats are pushed to UI via SocketIO
    "cpu_sample_s":       1.0,   # cpu_percent sampling window
    "disk_ttl_s":         60.0,  # disk stats are cached for 60 s
    "gpu_daemon_interval":2,     # nvidia-smi --loop=N  (seconds)
}

# ── Server ────────────────────────────────────────────────────────────────────

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 7799,
}

# ── Hermes paths ──────────────────────────────────────────────────────────────
# User-specific data lives under ~/.hermes/ so it survives code updates.

from pathlib import Path as _Path

HERMES_CONFIG = {
    "profiles_dir": str(_Path.home() / ".hermes" / "profiles"),
    "skills_dir":   str(_Path.home() / ".hermes" / "skills"),
    "memory_dir":   str(_Path.home() / ".hermes" / "memory"),
}

SKILL_SEARCH_TOP_K = 8

# ── HCI security ──────────────────────────────────────────────────────────────
import os as _os

HCI_CONFIG = {
    # SHA-256 hex digest of the HCI dashboard password.
    # Set via:  $env:HERMES_HCI_PASSWORD_HASH = python -c "import hashlib; print(hashlib.sha256(b'yourpass').hexdigest())"
    "password_hash": _os.environ.get("HERMES_HCI_PASSWORD_HASH", ""),
    "session_secret": _os.environ.get("HERMES_SESSION_SECRET", "change-me-in-production"),
}