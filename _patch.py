import re, os

# Read original orchestrator
with open(r'core\orchestrator.py', 'rb') as f:
    orig = f.read().decode('utf-8-sig')

# Find where imports end (after last import line before first def/class)
lines = orig.split('\n')
insert_idx = 0
for i, line in enumerate(lines):
    if line.startswith('import ') or line.startswith('from '):
        insert_idx = i + 1

# New code block to insert after imports
new_block = '''
# === TRINITY v2 ENHANCEMENTS ===
import time as _time
import os as _os
import logging as _logging
from core.resource_guard import get_guard as _get_guard

_logger = _logging.getLogger('trinity.orchestrator')
_fh = _logging.FileHandler('trinity_agent.log', encoding='utf-8')
_fh.setFormatter(_logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
_logger.addHandler(_fh)
_logger.setLevel(_logging.INFO)

class _Metrics:
    def __init__(self):
        self.calls=0; self.fails=0; self.retries=0; self.fallbacks=0
        self.total_lat=0.0; self.total_tok=0
    def record(self, role, model, lat, tok, ok):
        self.calls+=1
        if not ok: self.fails+=1
        self.total_lat+=lat; self.total_tok+=tok
        _logger.info(f'LLM role={role} model={model} lat={lat:.1f}s tok={tok} ok={ok}')
    def summary(self):
        avg=self.total_lat/max(self.calls,1)
        return dict(calls=self.calls, fails=self.fails, retries=self.retries,
                    fallbacks=self.fallbacks, avg_lat=round(avg,2), tokens=self.total_tok)
_metrics = _Metrics()

def _rag_retrieve(query, data_dir='memory/data', max_chunks=3):
    """Basic keyword RAG from memory/data folder."""
    if not _os.path.isdir(data_dir):
        return ''
    words = set(query.lower().split())
    hits = []
    for fn in _os.listdir(data_dir):
        fp = _os.path.join(data_dir, fn)
        if not _os.path.isfile(fp):
            continue
        try:
            t = open(fp, 'r', encoding='utf-8', errors='ignore').read(4096)
            sc = sum(1 for w in words if w in t.lower())
            if sc > 0:
                hits.append((sc, t[:800]))
        except Exception:
            pass
    hits.sort(key=lambda x: x[0], reverse=True)
    return '\\n'.join(h[1] for h in hits[:max_chunks])

def _guarded_post(url, payload, stream, timeout=120):
    """Ollama call with resource guard + retry/backoff + metrics."""
    import requests
    guard = _get_guard()
    ok, rmsg = guard.pre_llm_guard()
    if not ok:
        _logger.error(f'Resource guard blocked: {rmsg}')
        raise RuntimeError(f'Resources unavailable: {rmsg}')
    last_err = None
    for attempt in range(3):
        try:
            t0 = _time.time()
            resp = requests.post(url, json=payload, stream=stream, timeout=timeout)
            resp.raise_for_status()
            lat = _time.time() - t0
            _metrics.record(payload.get('model','?'), payload.get('model','?'), lat, 0, True)
            return resp
        except Exception as e:
            last_err = e
            _metrics.retries += 1
            delay = 2 * (2 ** attempt)
            _logger.warning(f'Ollama retry {attempt+1}/3 in {delay}s: {e}')
            _time.sleep(delay)
    _metrics.record(payload.get('model','?'), payload.get('model','?'), 0, 0, False)
    _logger.error(f'All retries failed: {last_err}')
    raise RuntimeError(f'Ollama failed after 3 retries: {last_err}')

def get_metrics():
    """Public metrics endpoint."""
    g = _get_guard()
    return dict(llm=_metrics.summary(), gpu=g.get_gpu_memory(),
                ram=g.get_ram_usage(), audio=g.check_audio_services())
# === END ENHANCEMENTS ===
'''

# Insert new block after imports
lines.insert(insert_idx, new_block)

# Now replace requests.post calls with _guarded_post
joined = '\n'.join(lines)

# Replace: requests.post(OLLAMA_URL + "/api/chat", ...) or similar
# Pattern: resp/response = requests.post(...)
joined = re.sub(
    r'(\w+)\s*=\s*requests\.post\(\s*(OLLAMA_URL\s*\+\s*["\'][^"\']+["\']|f["\'][^"\']+["\']|["\']http[^"\']+["\'])\s*,\s*json\s*=\s*(\w+)\s*,\s*stream\s*=\s*(\w+)(?:\s*,\s*timeout\s*=\s*(\w+))?\s*\)',
    lambda m: f'{m.group(1)} = _guarded_post({m.group(2)}, {m.group(3)}, {m.group(4)}, timeout={m.group(5) or "120"})',
    joined
)

# Inject RAG into run_chat: find where user_message is added to messages
# Add RAG context as system message
rag_inject = '''
        # RAG context injection
        _ctx = _rag_retrieve(user_message)
        if _ctx:
            messages.insert(0, {"role": "system", "content": "Relevant context:\\n" + _ctx})
'''
# Insert before the streaming call inside run_chat
chat_match = re.search(r'(def run_chat\b.*?)(for\s+\w+\s+in\s+)', joined, re.DOTALL)
if chat_match:
    insert_pos = chat_match.start(2)
    joined = joined[:insert_pos] + rag_inject + joined[insert_pos:]

# Write patched file
with open(r'core\orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(joined)

print(f'PATCHED OK: {len(joined)} chars written to core/orchestrator.py')