"""
HERMES AGENT — server.py
Flask + SocketIO entry point.  http://localhost:7799
HCI Control Interface:         http://localhost:7799/hci/
"""

import json
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from config import SERVER_CONFIG, MODEL_CANDIDATES, HCI_CONFIG
from core.orchestrator import Orchestrator
from core.system_monitor import SystemMonitor
from memory.memory_manager import MemoryManager
from memory.hermes_memory import HermesMemory

# ── Flask + SocketIO ──────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = HCI_CONFIG.get("session_secret", "hermes-local-2025")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 7  # 7 days

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=120,
    ping_interval=25,
)

# ── Core services ─────────────────────────────────────────────────────────────

memory_manager = MemoryManager()
hermes_memory  = HermesMemory()
monitor        = SystemMonitor(socketio)
orchestrator   = Orchestrator(socketio, monitor)

# ── Optional: rate limiter (flask-limiter) ───────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(get_remote_address, app=app, default_limits=["200 per minute"])
except ImportError:
    limiter = None

# ── HCI Blueprint + services ──────────────────────────────────────────────────
from hci.routes import hci_bp
from hci.terminal import register_terminal_namespace
from subconscious.scheduler import SubconsciousScheduler

scheduler = SubconsciousScheduler(monitor, hermes_memory, socketio,
                                   resource_guard=getattr(orchestrator, '_resource_guard', None))
scheduler.start()

# Expose services via app.config for HCI API endpoints
app.config["HERMES_MONITOR"]       = monitor
app.config["HERMES_MEMORY"]        = hermes_memory
app.config["HERMES_SCHEDULER"]     = scheduler
app.config["HERMES_ORCHESTRATOR"]  = orchestrator

app.register_blueprint(hci_bp)
register_terminal_namespace(socketio)

# Apply rate limit to HCI login if limiter is available
if limiter:
    limiter.limit("10 per minute")(hci_bp)

# ── Main routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# Chat — Server-Sent Events stream

@app.route("/api/chat", methods=["POST"])
def chat():
    data       = request.get_json(force=True)
    message    = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "empty message"}), 400

    def generate():
        try:
            for chunk in orchestrator.run_chat(
                message, session_id, memory_manager,
                hermes_memory=hermes_memory,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'role':'error','text':str(exc),'progress':-1})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# Build — Server-Sent Events stream

@app.route("/api/build", methods=["POST"])
def build():
    data            = request.get_json(force=True)
    project_request = data.get("request", "").strip()
    output_path     = data.get("output_path", str(Path.home() / "hermes-projects"))
    session_id      = data.get("session_id", "build-default")

    if not project_request:
        return jsonify({"error": "empty request"}), 400

    def generate():
        try:
            for chunk in orchestrator.run_build(
                project_request, output_path, session_id, memory_manager,
                hermes_memory=hermes_memory,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'role':'error','text':str(exc),'progress':-1})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# Build control

@app.route("/api/build/feedback", methods=["POST"])
def build_feedback():
    data = request.get_json(force=True)
    if data.get("session_id") and data.get("feedback", "").strip():
        orchestrator.inject_feedback(data["session_id"], data["feedback"].strip())
    return jsonify({"status": "ok"})


@app.route("/api/build/cancel", methods=["POST"])
def build_cancel():
    data = request.get_json(force=True)
    if data.get("session_id"):
        orchestrator.cancel_build(data["session_id"])
    return jsonify({"status": "cancelled"})


# Memory

@app.route("/api/memory", methods=["GET"])
def get_memory():
    sid = request.args.get("session_id", "default")
    return jsonify({
        "short_term":          memory_manager.get_short_term(sid),
        "long_term":           memory_manager.get_long_term(sid),
        "cross_session_facts": hermes_memory.get_user_facts(),
        "recent_episodes":     hermes_memory.get_recent_episodes(limit=5),
    })


@app.route("/api/memory/clear", methods=["POST"])
def clear_memory():
    data = request.get_json(force=True)
    memory_manager.clear(
        data.get("session_id", "default"),
        data.get("scope", "short"),
    )
    return jsonify({"status": "ok"})


# System stats (HTTP fallback for non-WS clients)

@app.route("/api/system", methods=["GET"])
def system_stats():
    return jsonify(monitor.get_stats())


# Resolved model names

@app.route("/api/models", methods=["GET"])
def get_models():
    return jsonify(orchestrator.models)


# Subconscious status (public endpoint — no HCI auth required)

@app.route("/api/subconscious/status", methods=["GET"])
def subconscious_status():
    return jsonify(scheduler.get_recent_results(50))


@app.route("/api/subconscious/run", methods=["POST"])
def subconscious_run():
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return jsonify({"error": "Forbidden"}), 403
    checks = scheduler.run_now()
    return jsonify({"ok": True, "checks": checks})


# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    monitor.start_streaming()
    emit("connected", {"status": "Hermes Agent Online", "ts": time.time()})


@socketio.on("disconnect")
def on_disconnect():
    pass


@socketio.on("request_stats")
def on_request_stats():
    emit("system_stats", monitor.get_stats())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = SERVER_CONFIG["host"]
    port = SERVER_CONFIG["port"]

    hci_status = "ENABLED  →  http://{}:{}/hci/".format(host, port) if HCI_CONFIG.get("password_hash") else "DISABLED (set HERMES_HCI_PASSWORD_HASH to enable)"

    print()
    print("  ██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗")
    print("  ██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝")
    print("  ███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗")
    print("  ██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║")
    print("  ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║")
    print("  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝")
    print()
    print(f"  Primary    : {MODEL_CANDIDATES['primary'][0]}")
    print(f"  Reviewer   : {MODEL_CANDIDATES['reviewer'][0]}")
    print(f"  Supervisor : {MODEL_CANDIDATES['supervisor'][0]}")
    print()
    print(f"  Chat UI  →  http://{host}:{port}")
    print(f"  HCI      →  {hci_status}")
    print()

    socketio.run(
        app,
        host=host,
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True,
        log_output=False,
    )
