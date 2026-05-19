"""
Hermes Subconscious — subconscious/scheduler.py  (P4)

Runs automated health checks 3x daily (08:00, 14:00, 22:00) and
hourly metrics snapshots via APScheduler.

IMPORTANT: Uses 'threadpool' executor (NOT gevent executor) to avoid
conflict with Flask-SocketIO's gevent monkeypatching.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

import requests

from config import OLLAMA_URL

if TYPE_CHECKING:
    from memory.hermes_memory import HermesMemory

logger = logging.getLogger("hermes.subconscious")

_MAX_RESULTS = 200  # ring buffer for recent check results


class SubconsciousScheduler:
    """
    Background scheduler for automated health checks.

    Args:
        monitor:       SystemMonitor instance (for disk/RAM/GPU stats)
        hermes_memory: HermesMemory instance (for DB size check)
        socketio:      Flask-SocketIO instance (to push results to HCI clients)
        resource_guard: ResourceGuard instance (for VRAM/audio checks)
    """

    def __init__(self, monitor, hermes_memory, socketio, resource_guard=None):
        self._monitor        = monitor
        self._hermes_memory  = hermes_memory
        self._socketio       = socketio
        self._resource_guard = resource_guard
        self._results: list[dict] = []
        self._scheduler = None

    def start(self) -> None:
        """Start the background scheduler. Safe to call multiple times."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.warning(
                "APScheduler not installed — subconscious checks disabled. "
                "Run: pip install apscheduler"
            )
            return

        self._scheduler = BackgroundScheduler(
            daemon=True,
            executors={"default": {"type": "threadpool", "max_workers": 2}},
            job_defaults={"coalesce": True, "max_instances": 1},
        )

        # 3x daily health checks
        for hour in (8, 14, 22):
            self._scheduler.add_job(
                self._health_check,
                CronTrigger(hour=hour, minute=0),
                id=f"health_{hour}",
                replace_existing=True,
            )

        # Hourly metrics snapshot
        self._scheduler.add_job(
            self._snapshot_metrics,
            CronTrigger(minute=0),
            id="metrics_snapshot",
            replace_existing=True,
        )

        self._scheduler.start()
        logger.info("Subconscious scheduler started (health checks at 08:00, 14:00, 22:00)")

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def run_now(self) -> list[dict]:
        """Trigger a manual health check and return results."""
        return self._health_check()

    # ── Health check ──────────────────────────────────────────────────────────

    def _health_check(self) -> list[dict]:
        ts     = datetime.now().isoformat()
        checks = []

        checks.append(self._check_ollama())
        checks.append(self._check_resources())
        checks.append(self._check_audio())
        checks.append(self._check_memory_db())
        checks.append(self._check_disk())

        result = {
            "ts":      ts,
            "checks":  checks,
            "all_ok":  all(c["ok"] for c in checks),
            "summary": self._summarise(checks),
        }

        self._results.append(result)
        if len(self._results) > _MAX_RESULTS:
            self._results = self._results[-_MAX_RESULTS:]

        # Push to connected HCI clients
        try:
            self._socketio.emit("subconscious_check", result, namespace="/hci")
        except Exception:
            pass

        status = "✓ ALL OK" if result["all_ok"] else "⚠ ISSUES FOUND"
        logger.info(f"Subconscious health check: {status} — {result['summary']}")
        return checks

    def _check_ollama(self) -> dict:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return {"name": "ollama", "ok": True,
                        "detail": f"Running, {len(models)} model(s) available"}
            return {"name": "ollama", "ok": False,
                    "detail": f"HTTP {r.status_code}"}
        except Exception as e:
            return {"name": "ollama", "ok": False, "detail": str(e)}

    def _check_resources(self) -> dict:
        try:
            if self._resource_guard:
                ok = self._resource_guard.check_resources_ok()
                return {"name": "resources", "ok": ok,
                        "detail": "RAM/VRAM within limits" if ok else "Resource pressure detected"}
            # Fallback: use monitor directly
            stats = self._monitor.get_stats()
            ram_pct  = stats.get("ram", {}).get("percent", 0)
            vram_pct = stats.get("gpu", {}).get("vram_pct", 0)
            ok = ram_pct < 90 and (vram_pct < 90 or vram_pct == 0)
            return {"name": "resources", "ok": ok,
                    "detail": f"RAM {ram_pct:.0f}%  VRAM {vram_pct:.0f}%"}
        except Exception as e:
            return {"name": "resources", "ok": False, "detail": str(e)}

    def _check_audio(self) -> dict:
        try:
            if self._resource_guard and hasattr(self._resource_guard, "check_audio_services"):
                ok = self._resource_guard.check_audio_services()
                if not ok:
                    # Attempt auto-restart (existing logic in resource_guard)
                    try:
                        self._resource_guard.restart_audio_services()
                        return {"name": "audio", "ok": True,
                                "detail": "Auto-restarted audio services"}
                    except Exception:
                        pass
                return {"name": "audio", "ok": ok,
                        "detail": "OK" if ok else "Audio services not running"}
            return {"name": "audio", "ok": True, "detail": "Skipped (no resource guard)"}
        except Exception as e:
            return {"name": "audio", "ok": False, "detail": str(e)}

    def _check_memory_db(self) -> dict:
        try:
            if self._hermes_memory:
                stats   = self._hermes_memory.db_stats()
                size_mb = stats.get("size_mb", 0)
                ok      = size_mb < 100
                detail  = (
                    f"{size_mb:.1f} MB  |  "
                    f"{stats.get('facts',0)} facts  "
                    f"{stats.get('episodes',0)} episodes"
                )
                if not ok:
                    detail += "  ⚠ DB > 100MB — consider pruning"
                return {"name": "memory_db", "ok": ok, "detail": detail}
            return {"name": "memory_db", "ok": True, "detail": "HermesMemory not initialised"}
        except Exception as e:
            return {"name": "memory_db", "ok": False, "detail": str(e)}

    def _check_disk(self) -> dict:
        try:
            stats    = self._monitor.get_stats()
            disk_pct = stats.get("disk", {}).get("percent", 0)
            ok       = disk_pct < 90
            return {"name": "disk", "ok": ok,
                    "detail": f"Disk {disk_pct:.0f}% used"
                              + ("  ⚠ Low disk space!" if not ok else "")}
        except Exception as e:
            return {"name": "disk", "ok": False, "detail": str(e)}

    # ── Metrics snapshot ──────────────────────────────────────────────────────

    def _snapshot_metrics(self) -> None:
        """Hourly: prune old episodes to keep DB lean."""
        try:
            if self._hermes_memory:
                deleted = self._hermes_memory.prune_old_episodes(keep_days=90)
                if deleted:
                    logger.info(f"Metrics snapshot: pruned {deleted} old episodes")
        except Exception as e:
            logger.debug(f"Metrics snapshot error: {e}")

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_recent_results(self, limit: int = 50) -> list[dict]:
        return self._results[-limit:]

    def get_jobs(self) -> list[dict]:
        """Return list of scheduled job info for HCI cron tab."""
        if not self._scheduler:
            return []
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id":       job.id,
                "name":     job.name or job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger":  str(job.trigger),
            })
        return jobs

    @staticmethod
    def _summarise(checks: list[dict]) -> str:
        failed = [c["name"] for c in checks if not c["ok"]]
        if not failed:
            return "All checks passed"
        return "Failed: " + ", ".join(failed)
