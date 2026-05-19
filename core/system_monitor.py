"""
TRINITY AGENT — core/system_monitor.py

Combines the best parts of both previous versions:

  From v2 (uploaded):
    • nvidia-smi --loop=2 daemon (no fork per poll)
    • Individual TTL cache per metric group (CPU 3s, RAM 5s, GPU 2s, Disk 60s)
    • Windows CREATE_NO_WINDOW flag
    • Graceful stop() / cleanup

  From v1 (our previous):
    • Non-blocking CPU sampler (background thread, never blocks monitor loop)
    • Token counter updated PER TOKEN for live UI display
    • start_streaming() pushes stats via SocketIO every PUSH_INTERVAL seconds
    • get_token_snapshot() callable from orchestrator mid-stream

All settings come from config.py.
"""

import logging
import platform
import shutil
import subprocess
import threading
import time
from typing import Optional

import psutil

from config import MONITOR_CONFIG

logger = logging.getLogger("trinity.monitor")

_IS_WINDOWS = platform.system() == "Windows"
_DISK_ROOT  = "C:\\" if _IS_WINDOWS else "/"
_PUSH_INTERVAL  = MONITOR_CONFIG["push_interval_s"]
_CPU_SAMPLE     = MONITOR_CONFIG["cpu_sample_s"]
_DISK_TTL       = MONITOR_CONFIG["disk_ttl_s"]
_GPU_INTERVAL   = MONITOR_CONFIG["gpu_daemon_interval"]


# ── Non-blocking CPU sampler ──────────────────────────────────────────────────

class _CpuSampler:
    """
    Runs psutil.cpu_percent(interval=1.0) in a dedicated background thread.
    The monitor loop reads a cached value instantly without ever blocking.
    """
    def __init__(self, sample_interval: float = 1.0):
        self._lock  = threading.Lock()
        self._value = 0.0
        psutil.cpu_percent(interval=None)   # discard first dummy reading
        t = threading.Thread(
            target=self._run, args=(sample_interval,),
            daemon=True, name="cpu-sampler",
        )
        t.start()

    def _run(self, interval: float):
        while True:
            v = psutil.cpu_percent(interval=interval)
            with self._lock:
                self._value = v

    def get(self) -> float:
        with self._lock:
            return self._value


# ── NVIDIA daemon ─────────────────────────────────────────────────────────────

class _NvidiaDaemon:
    """
    Runs `nvidia-smi --query-gpu=... --format=csv --loop=N` as one persistent
    process. A background reader thread updates a cached dict every N seconds.
    No subprocess spawned per poll — zero fork overhead on SSD.
    """

    _QUERY = (
        "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    )

    def __init__(self, interval: int = 2):
        self._lock    = threading.Lock()
        self._data    = _gpu_zero()
        self._proc:   Optional[subprocess.Popen] = None
        self._alive   = False
        self._interval = interval

        if shutil.which("nvidia-smi"):
            self._start()
        else:
            self._data["reason"] = "nvidia-smi not found"

    def _start(self):
        flags = {}
        if _IS_WINDOWS:
            flags["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self._proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={self._QUERY}",
                    "--format=csv,noheader,nounits",
                    f"--loop={self._interval}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                **flags,
            )
            self._alive = True
            threading.Thread(
                target=self._reader, daemon=True, name="nvidia-daemon-reader"
            ).start()
        except Exception as e:
            logger.warning(f"nvidia-smi daemon failed to start: {e}")
            self._data["reason"] = str(e)

    def _reader(self):
        try:
            while self._alive and self._proc and self._proc.poll() is None:
                if self._proc.stdout is None:
                    break
                line = self._proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                self._parse(line)
        except Exception as e:
            logger.debug(f"nvidia reader error: {e}")

    def _parse(self, line: str):
        try:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                return
            gpu_load = int(float(parts[0]))
            vram_used  = float(parts[1])   # MiB
            vram_total = float(parts[2])   # MiB
            temp       = int(float(parts[3]))
            power      = float(parts[4]) if len(parts) > 4 and parts[4] != "" else 0.0
            with self._lock:
                self._data = {
                    "available":  True,
                    "load":       gpu_load,
                    "vram_used":  round(vram_used  / 1024, 2),   # → GB
                    "vram_total": round(vram_total / 1024, 2),
                    "vram_pct":   round(vram_used / max(vram_total, 1) * 100, 1),
                    "temp":       temp,
                    "power":      round(power, 1),
                }
        except Exception as e:
            logger.debug(f"nvidia parse error on '{line}': {e}")

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    @property
    def available(self) -> bool:
        return self._alive

    def stop(self):
        self._alive = False
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None


def _gpu_zero() -> dict:
    return {
        "available":  False,
        "load":       0,
        "vram_used":  0.0,
        "vram_total": 0.0,
        "vram_pct":   0.0,
        "temp":       0,
        "power":      0.0,
        "reason":     "initializing",
    }


# ── Disk cache ────────────────────────────────────────────────────────────────

class _DiskCache:
    """Read disk_usage once then cache for TTL seconds."""

    def __init__(self, path: str = _DISK_ROOT, ttl: float = 60.0):
        self._path = path
        self._ttl  = ttl
        self._lock = threading.Lock()
        self._data = self._read()
        self._ts   = time.time()

    def _read(self) -> dict:
        try:
            d = psutil.disk_usage(self._path)
            return {
                "used_gb":  round(d.used  / 1e9, 1),
                "total_gb": round(d.total / 1e9, 1),
                "free_gb":  round(d.free  / 1e9, 1),
                "percent":  d.percent,
            }
        except Exception:
            return {"used_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0, "percent": 0.0}

    def get(self) -> dict:
        now = time.time()
        with self._lock:
            if now - self._ts >= self._ttl:
                self._data = self._read()
                self._ts   = now
            return dict(self._data)


# ── SystemMonitor ─────────────────────────────────────────────────────────────

class SystemMonitor:
    """
    Collects hardware stats and streams them to SocketIO clients.

    Usage:
        monitor = SystemMonitor(socketio)
        monitor.start_streaming()          # call once on first WS connect
        monitor.add_tokens(in_n, out_n)    # call per streaming token chunk
        stats = monitor.get_stats()        # callable from any thread
        monitor.stop()                     # clean up on shutdown
    """

    def __init__(self, socketio=None):
        self.socketio = socketio

        # Hardware samplers
        self._cpu   = _CpuSampler(sample_interval=_CPU_SAMPLE)
        self._gpu   = _NvidiaDaemon(interval=_GPU_INTERVAL)
        self._disk  = _DiskCache(ttl=_DISK_TTL)

        # Token counter — updated per-chunk by orchestrator
        self._tok_lock = threading.Lock()
        self._tok_in   = 0
        self._tok_out  = 0

        # Streaming control
        self._stream_lock    = threading.Lock()
        self._streaming      = False
        self._stream_thread: Optional[threading.Thread] = None

        # Boot diagnostic — printed to console so failures are immediately visible
        threading.Thread(target=self._boot_diagnostic, daemon=True,
                         name="monitor-diag").start()

    def _boot_diagnostic(self):
        """Runs once at startup — prints exactly what psutil can and cannot read."""
        time.sleep(2.0)   # let samplers warm up first
        print()
        print("  ── SystemMonitor diagnostic ──────────────────")
        try:
            v = psutil.cpu_percent(interval=1.0)
            print(f"  ✓  cpu_percent  : {v}%")
        except Exception as e:
            print(f"  ✗  cpu_percent  : {e}")
        try:
            f = psutil.cpu_freq()
            print(f"  ✓  cpu_freq     : {f.current:.0f} MHz" if f else "  ✗  cpu_freq: returned None")
        except Exception as e:
            print(f"  ✗  cpu_freq     : {e}")
        try:
            r = psutil.virtual_memory()
            print(f"  ✓  virtual_mem  : {r.total/1e9:.1f} GB total, {r.percent}% used")
        except Exception as e:
            print(f"  ✗  virtual_mem  : {e}")
        try:
            d = psutil.disk_usage(_DISK_ROOT)
            print(f"  ✓  disk_usage   : {d.total/1e9:.1f} GB total, {d.percent}% used")
        except Exception as e:
            print(f"  ✗  disk_usage   : {e}")
        g = self._gpu.get()
        if g.get("available"):
            print(f"  ✓  nvidia-smi   : GPU {g['load']}%  VRAM {g['vram_used']}/{g['vram_total']} GB")
        else:
            print(f"  ✗  nvidia-smi   : {g.get('reason','not available')}")
        print("  ─────────────────────────────────────────────")
        print()

    # ── Token tracking ────────────────────────────────────────────────────────

    def add_tokens(self, in_delta: int = 0, out_delta: int = 0):
        """Thread-safe. Call on every streaming chunk from orchestrator."""
        with self._tok_lock:
            self._tok_in  += in_delta
            self._tok_out += out_delta

    def get_token_snapshot(self) -> dict:
        with self._tok_lock:
            return {
                "in":    self._tok_in,
                "out":   self._tok_out,
                "total": self._tok_in + self._tok_out,
            }

    def reset_tokens(self):
        with self._tok_lock:
            self._tok_in  = 0
            self._tok_out = 0

    # ── Stats snapshot ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Cheap — all data comes from cached background workers.
        Safe to call from any thread at any frequency.
        Each metric is collected independently so one failure never zeros the rest.
        """
        # ── CPU ───────────────────────────────────────────────────────────────
        try:
            cpu_load = round(self._cpu.get(), 1)
        except Exception as e:
            logger.warning(f"get_stats CPU load error: {e}")
            cpu_load = 0.0

        try:
            freq = psutil.cpu_freq()
            freq_mhz = int(freq.current) if freq else 0
            freq_max = int(freq.max)     if freq else 0
        except Exception as e:
            logger.warning(f"get_stats cpu_freq error: {e}")
            freq_mhz, freq_max = 0, 0

        try:
            cores = psutil.cpu_count(logical=True)  or 1
            phys  = psutil.cpu_count(logical=False) or 1
        except Exception as e:
            logger.warning(f"get_stats cpu_count error: {e}")
            cores, phys = 1, 1

        # ── RAM ───────────────────────────────────────────────────────────────
        try:
            ram = psutil.virtual_memory()
            ram_stats = {
                "used_gb":  round(ram.used      / 1e9, 2),
                "total_gb": round(ram.total     / 1e9, 2),
                "free_gb":  round(ram.available / 1e9, 2),
                "percent":  ram.percent,
            }
            if ram_stats["total_gb"] < 0.1:
                logger.warning("get_stats: RAM total_gb suspiciously low — psutil permissions?")
        except Exception as e:
            logger.warning(f"get_stats RAM error: {e}")
            ram_stats = {"used_gb": 0, "total_gb": 0, "free_gb": 0, "percent": 0}

        # ── GPU / Disk / Tokens ───────────────────────────────────────────────
        try:
            gpu_stats = self._gpu.get()
        except Exception as e:
            logger.warning(f"get_stats GPU error: {e}")
            gpu_stats = _gpu_zero()

        try:
            disk_stats = self._disk.get()
        except Exception as e:
            logger.warning(f"get_stats disk error: {e}")
            disk_stats = {"used_gb": 0, "total_gb": 0, "free_gb": 0, "percent": 0}

        return {
            "gpu": gpu_stats,
            "cpu": {
                "load":     cpu_load,
                "freq_mhz": freq_mhz,
                "freq_max": freq_max,
                "cores":    cores,
                "phys":     phys,
            },
            "ram":    ram_stats,
            "disk":   disk_stats,
            "tokens": self.get_token_snapshot(),
            "ts":     time.time(),
        }

    def get_summary(self) -> str:
        """One-line string for server-side logging."""
        s  = self.get_stats()
        g  = s["gpu"]
        parts = [
            f"CPU:{s['cpu']['load']:.0f}%",
            f"RAM:{s['ram']['used_gb']}/{s['ram']['total_gb']}GB",
        ]
        if g.get("available"):
            parts += [
                f"GPU:{g['load']}%",
                f"VRAM:{g['vram_used']}/{g['vram_total']}GB",
                f"{g['temp']}°C",
            ]
        parts.append(f"Disk:{s['disk']['percent']:.0f}%")
        tok = s["tokens"]
        parts.append(f"Tokens in:{tok['in']} out:{tok['out']}")
        return " | ".join(parts)

    # ── SocketIO streaming ────────────────────────────────────────────────────

    def start_streaming(self):
        """Idempotent. Safe to call on every WebSocket connect."""
        with self._stream_lock:
            if self._streaming:
                return
            self._streaming = True
            self._stream_thread = threading.Thread(
                target=self._push_loop,
                daemon=True,
                name="monitor-push",
            )
            self._stream_thread.start()
        logger.info("SystemMonitor streaming started")

    def stop_streaming(self):
        with self._stream_lock:
            self._streaming = False

    def _push_loop(self):
        # Emit one immediate update so the UI doesn't wait for the first interval
        try:
            if self.socketio:
                self.socketio.emit("system_stats", self.get_stats(), namespace="/")
        except Exception as e:
            logger.debug(f"monitor initial push error: {e}")
        while self._streaming:
            try:
                if self.socketio:
                    self.socketio.emit("system_stats", self.get_stats(), namespace="/")
            except Exception as e:
                logger.debug(f"monitor push error: {e}")
            time.sleep(_PUSH_INTERVAL)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Alias kept for compatibility with uploaded server.py."""
        self.start_streaming()

    def stop(self):
        """Clean shutdown — terminate nvidia-smi daemon."""
        self.stop_streaming()
        self._gpu.stop()
        logger.info("SystemMonitor stopped")