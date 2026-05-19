$c = @'
import time
import logging
import requests
from core.resource_guard import ResourceGuard
from core.retry import retry_request, stream_with_retry
from core.metrics import log_request

logger = logging.getLogger("trinity.orchestrator")

class Orchestrator:
    def __init__(self, config=None):
        self.config = config or {}
        self.base_url = self.config.get("ollama_url", "http://localhost:11434")
        self.resource_guard = ResourceGuard(vram_threshold_pct=self.config.get("vram_threshold", 85), ram_threshold_pct=self.config.get("ram_threshold", 85))
        self.resource_guard.start_audio_monitor()
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    def _call_ollama(self, endpoint, payload, timeout=120):
        url = f"{self.base_url}{endpoint}"
        self.resource_guard.pre_llm_guard()
        start = time.time()
        try:
            resp = retry_request("POST", url, json=payload, timeout=timeout, max_retries=3, base_delay=1.0)
            latency_ms = (time.time() - start) * 1000
            success = resp.status_code == 200
            self._request_count += 1
            self._total_latency += latency_ms
            if not success:
                self._error_count += 1
            log_request(role=payload.get("role","chat"), model=payload.get("model","unknown"), latency_ms=latency_ms, tokens_generated=0, success=success)
            return resp
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            self._request_count += 1
            self._error_count += 1
            self._total_latency += latency_ms
            log_request(role=payload.get("role","chat"), model=payload.get("model","unknown"), latency_ms=latency_ms, tokens_generated=0, success=False, error=str(e))
            raise

    def run_chat(self, messages, model="qwen3:1.7b", role="chat"):
        payload = {"model": model, "messages": messages, "stream": False, "role": role}
        resp = self._call_ollama("/api/chat", payload)
        return resp.json().get("message", {}).get("content", "")

    def run_generate(self, prompt, model="qwen3:1.7b", role="coder"):
        payload = {"model": model, "prompt": prompt, "stream": False, "role": role}
        resp = self._call_ollama("/api/generate", payload)
        return resp.json().get("response", "")

    def run_stream(self, messages, model="qwen3:1.7b", role="chat", timeout=120):
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": True, "role": role}
        self.resource_guard.pre_llm_guard()
        start = time.time()
        try:
            resp = stream_with_retry("POST", url, json=payload, timeout=timeout)
            latency_ms = (time.time() - start) * 1000
            log_request(role=role, model=model, latency_ms=latency_ms, tokens_generated=0, success=True)
            return resp
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            log_request(role=role, model=model, latency_ms=latency_ms, tokens_generated=0, success=False, error=str(e))
            raise

    def get_metrics(self):
        avg = (self._total_latency / self._request_count) if self._request_count > 0 else 0
        return {"total_requests": self._request_count, "errors": self._error_count, "avg_latency_ms": round(avg,1), "error_rate": round(self._error_count/max(self._request_count,1)*100,1)}

    def get_resource_status(self):
        return {"gpu_memory": self.resource_guard.get_gpu_memory(), "ram_usage": self.resource_guard.get_ram_usage(), "resources_ok": self.resource_guard.check_resources_ok(), "audio_ok": self.resource_guard.check_audio_services()}

    def shutdown(self):
        self.resource_guard.stop_audio_monitor()
'@
Set-Content -Path "core/orchestrator.py" -Value $c -Encoding UTF8
Write-Host "ORCH_WRITTEN"
