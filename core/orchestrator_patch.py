import re

with open(r'D:\agent_local\core\orchestrator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Patch keep_alive from 5m to 2m
content = content.replace("KEEP_ALIVE = '5m'", "KEEP_ALIVE = '2m'")
content = content.replace("keep_alive=300 (5 min)", "keep_alive=120 (2 min)")

with open(r'D:\agent_local\core\orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("PATCHED orchestrator.py: keep_alive 5m -> 2m")