"""
HCI REST API endpoints — hci/api.py  (P5)

All routes are mounted under /hci/api/ via the hci Blueprint.
All routes require HCI authentication. Mutating routes require CSRF header.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app

from hci.auth import hci_login_required, csrf_required

api_bp = Blueprint("hci_api", __name__, url_prefix="/hci/api")


# ── System stats ──────────────────────────────────────────────────────────────

@api_bp.get("/system")
@hci_login_required
def system_stats():
    monitor = current_app.config["HERMES_MONITOR"]
    return jsonify(monitor.get_stats())


# ── File browser ──────────────────────────────────────────────────────────────

@api_bp.get("/files")
@hci_login_required
def list_files():
    raw_path = request.args.get("path", ".")
    try:
        p = Path(raw_path).expanduser().resolve()
        if not p.exists():
            return jsonify({"error": "Path not found"}), 404
        if p.is_file():
            return jsonify({"type": "file", "path": str(p), "size": p.stat().st_size})
        entries = []
        for item in sorted(p.iterdir()):
            entries.append({
                "name":  item.name,
                "type":  "dir" if item.is_dir() else "file",
                "size":  item.stat().st_size if item.is_file() else None,
            })
        return jsonify({"type": "dir", "path": str(p), "entries": entries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.get("/files/read")
@hci_login_required
def read_file():
    raw_path = request.args.get("path", "")
    if not raw_path:
        return jsonify({"error": "path required"}), 400
    try:
        p = Path(raw_path).expanduser().resolve()
        if not p.is_file():
            return jsonify({"error": "Not a file"}), 404
        if p.stat().st_size > 1_000_000:
            return jsonify({"error": "File too large (>1MB)"}), 413
        content = p.read_text(encoding="utf-8", errors="replace")
        return jsonify({"path": str(p), "content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.post("/files/write")
@hci_login_required
@csrf_required
def write_file():
    data = request.get_json() or {}
    raw_path = data.get("path", "")
    content  = data.get("content", "")
    if not raw_path:
        return jsonify({"error": "path required"}), 400
    try:
        p = Path(raw_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return jsonify({"ok": True, "path": str(p), "size": len(content)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.delete("/files/delete")
@hci_login_required
@csrf_required
def delete_file():
    raw_path = request.args.get("path", "")
    if not raw_path:
        return jsonify({"error": "path required"}), 400
    try:
        import shutil
        p = Path(raw_path).expanduser().resolve()
        if not p.exists():
            return jsonify({"error": "Not found"}), 404
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Cron / scheduler ─────────────────────────────────────────────────────────

@api_bp.get("/cron/list")
@hci_login_required
def cron_list():
    scheduler = current_app.config.get("HERMES_SCHEDULER")
    if not scheduler:
        return jsonify([])
    return jsonify(scheduler.get_jobs())


@api_bp.post("/cron/run/<job_id>")
@hci_login_required
@csrf_required
def cron_run(job_id: str):
    scheduler = current_app.config.get("HERMES_SCHEDULER")
    if not scheduler or not scheduler._scheduler:
        return jsonify({"error": "Scheduler not running"}), 503
    try:
        from datetime import datetime as _dt
        scheduler._scheduler.get_job(job_id).modify(next_run_time=_dt.now())
        return jsonify({"ok": True, "job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Token analytics ───────────────────────────────────────────────────────────

@api_bp.get("/analytics/tokens")
@hci_login_required
def analytics_tokens():
    """Parse logs/metrics.jsonl and return per-day token totals."""
    try:
        log_path = Path("logs") / "metrics.jsonl"
        if not log_path.exists():
            return jsonify([])
        daily: dict[str, dict] = {}
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    day = entry.get("timestamp", "")[:10]
                    if not day:
                        continue
                    bucket = daily.setdefault(day, {"tokens": 0, "calls": 0})
                    bucket["tokens"] += entry.get("tokens_generated", 0)
                    bucket["calls"]  += 1
                except Exception:
                    continue
        result = [
            {"day": k, **v}
            for k, v in sorted(daily.items())
        ]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.get("/analytics/skills")
@hci_login_required
def analytics_skills():
    hermes_memory = current_app.config.get("HERMES_MEMORY")
    if not hermes_memory:
        return jsonify([])
    return jsonify(hermes_memory.get_skill_stats())


@api_bp.get("/analytics/sessions")
@hci_login_required
def analytics_sessions():
    hermes_memory = current_app.config.get("HERMES_MEMORY")
    if not hermes_memory:
        return jsonify([])
    return jsonify(hermes_memory.get_recent_episodes(limit=50))


# ── Subconscious ──────────────────────────────────────────────────────────────

@api_bp.get("/subconscious/status")
@hci_login_required
def subconscious_status():
    scheduler = current_app.config.get("HERMES_SCHEDULER")
    if not scheduler:
        return jsonify([])
    return jsonify(scheduler.get_recent_results(50))


@api_bp.post("/subconscious/run")
@hci_login_required
@csrf_required
def subconscious_run():
    scheduler = current_app.config.get("HERMES_SCHEDULER")
    if not scheduler:
        return jsonify({"error": "Scheduler not initialised"}), 503
    checks = scheduler.run_now()
    return jsonify({"ok": True, "checks": checks})


# ── Skills ────────────────────────────────────────────────────────────────────

@api_bp.get("/skills/list")
@hci_login_required
def skills_list():
    orchestrator = current_app.config.get("HERMES_ORCHESTRATOR")
    if not orchestrator or not orchestrator.skill_registry:
        return jsonify([])
    skills = [
        {
            "name":        s.name,
            "description": s.description,
            "domain":      s.domain,
            "destructive": s.destructive,
            "params":      list(s.parameters.keys()),
        }
        for s in orchestrator.skill_registry.all_skills()
    ]
    return jsonify(skills)


@api_bp.post("/skills/reload")
@hci_login_required
@csrf_required
def skills_reload():
    orchestrator = current_app.config.get("HERMES_ORCHESTRATOR")
    if not orchestrator or not orchestrator.skill_registry:
        return jsonify({"error": "No skill registry"}), 503
    orchestrator.skill_registry.reload()
    return jsonify({"ok": True, "count": len(orchestrator.skill_registry)})
