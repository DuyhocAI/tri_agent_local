import sys
sys.path.insert(0,".")
try:
    from core.orchestrator import KEEP_ALIVE, OLLAMA_OPTIONS, MODEL_CONFIG
    print("IMPORT: OK")
    print("KEEP_ALIVE:", KEEP_ALIVE)
    print("OPTIONS:", OLLAMA_OPTIONS)
    print("MODELS:", {k: v["model"] for k, v in MODEL_CONFIG.items()})
except Exception as e:
    print("IMPORT FAILED:", e)
    import traceback
    traceback.print_exc()
