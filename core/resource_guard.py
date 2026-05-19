"""
TRINITY AGENT - core/resource_guard.py

Integrated resource guard:
1. VRAM/RAM check before each LLM call
2. Audio service protection and auto-recovery
3. GPU resource throttling to prevent audio driver crash
"""
import subprocess
import psutil
import time
import logging
import threading

logger = logging.getLogger('hermes.resource_guard')

class ResourceGuard:
    def __init__(self, vram_threshold_pct=85, ram_threshold_pct=90):
        self.vram_threshold_pct = vram_threshold_pct
        self.ram_threshold_pct = ram_threshold_pct
        self._audio_check_interval = 30
        self._audio_monitor_thread = None
        self._running = False
        self._last_vram = {}

    def get_gpu_memory(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                used, free, total = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
                self._last_vram = {'used': used, 'free': free, 'total': total, 'pct': round(used/total*100, 1)}
                return self._last_vram
        except Exception as e:
            logger.warning(f'GPU memory check failed: {e}')
        return self._last_vram or {'used': 0, 'free': 12288, 'total': 12288, 'pct': 0}

    def get_ram_usage(self):
        mem = psutil.virtual_memory()
        return {'used_gb': round(mem.used/1024**3,1), 'free_gb': round(mem.available/1024**3,1), 'total_gb': round(mem.total/1024**3,1), 'pct': mem.percent}

    def check_resources_ok(self):
        gpu = self.get_gpu_memory()
        ram = self.get_ram_usage()
        issues = []
        if gpu['pct'] > self.vram_threshold_pct:
            issues.append(f'VRAM high: {gpu["pct"]}%')
        if ram['pct'] > self.ram_threshold_pct:
            issues.append(f'RAM high: {ram["pct"]}%')
        if issues:
            logger.warning(f'Resource warning: {issues}')
            return False, '; '.join(issues)
        return True, f'OK - VRAM:{gpu["pct"]}% RAM:{ram["pct"]}%'

    def check_audio_services(self):
        services = {}
        for svc in ['Audiosrv', 'AudioEndpointBuilder']:
            try:
                result = subprocess.run(['sc', 'query', svc], capture_output=True, text=True, timeout=5)
                services[svc] = 'running' if 'RUNNING' in result.stdout else 'stopped'
            except Exception:
                services[svc] = 'unknown'
        return services

    def restart_audio_services(self):
        logger.warning('Restarting audio services...')
        results = {}
        for svc in ['AudioEndpointBuilder', 'Audiosrv']:
            try:
                subprocess.run(['net', 'stop', svc], capture_output=True, timeout=10)
                time.sleep(1)
                r = subprocess.run(['net', 'start', svc], capture_output=True, text=True, timeout=10)
                results[svc] = 'restarted' if r.returncode == 0 else 'failed'
            except Exception as e:
                results[svc] = f'error: {e}'
        logger.info(f'Audio restart: {results}')
        return results

    def _audio_monitor_loop(self):
        while self._running:
            try:
                services = self.check_audio_services()
                if any(v != 'running' for v in services.values()):
                    logger.warning(f'Audio issue: {services}')
                    self.restart_audio_services()
            except Exception as e:
                logger.error(f'Audio monitor error: {e}')
            time.sleep(self._audio_check_interval)

    def start_audio_monitor(self):
        if self._audio_monitor_thread and self._audio_monitor_thread.is_alive():
            return
        self._running = True
        self._audio_monitor_thread = threading.Thread(target=self._audio_monitor_loop, daemon=True, name='AudioMonitor')
        self._audio_monitor_thread.start()
        logger.info('Audio monitor started')

    def stop_audio_monitor(self):
        self._running = False

    def pre_llm_guard(self):
        for attempt in range(3):
            ok, msg = self.check_resources_ok()
            if ok:
                return True, msg
            logger.warning(f'Resource guard attempt {attempt+1}/3: {msg}')
            time.sleep(10)
        return False, f'Resources constrained: {msg}'

_guard = None
def get_guard():
    global _guard
    if _guard is None:
        _guard = ResourceGuard()
        _guard.start_audio_monitor()
    return _guard
