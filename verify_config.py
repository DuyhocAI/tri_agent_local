import sys
sys.path.insert(0,".")
try:
    from config import PRIMARY_OPTIONS, MODEL_CANDIDATES
    print("IMPORT: OK")
    print("keep_alive:", PRIMARY_OPTIONS.get("keep_alive"))
    print("OPTIONS:", PRIMARY_OPTIONS)
    print("MODELS:", {role: candidates[0] for role, candidates in MODEL_CANDIDATES.items()})
except Exception as e:
    print("IMPORT FAILED:", e)
    import traceback
    traceback.print_exc()
