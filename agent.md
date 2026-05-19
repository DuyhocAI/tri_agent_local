# TRINITY AGENT - Configuration and Optimization 
 
## Models (3 roles, 2 unique models) 
| Role | Model | Size | GPU | Context | 
| chat | qwen3:latest | 6.6 GB | 100%% GPU | 8192 | 
| coder | qwen3:latest | 6.6 GB | 100%% GPU | 8192 | 
| reviewer | mistral:v0.3 | 6.0 GB | 100%% GPU | 8192 | 
 
## Ollama Options 
- num_ctx: 8192 (balanced for 12GB VRAM) 
- num_predict: -1 (unlimited output) 
- num_gpu: 99 (full GPU offload) 
- keep_alive: 2m (auto-unload idle models) 
 
## Hardware 
- GPU: NVIDIA RTX 3060 12GB VRAM 
- RAM: 32GB (22GB free typical) 
- Single model loaded at a time (6-6.6 GB VRAM) 
 
## Excluded: llama3.3 (42GB, does not fit 12GB VRAM) 
 
## Test Results 2026-05-13 
- qwen3:latest: PASS 100%% GPU ctx=8192 
- mistral:v0.3: PASS 100%% GPU ctx=8192 
- Auto-unload 2m: CONFIRMED 
- System lag: NONE 
 
## Quality Strategy 
- Unlimited output tokens = complete answers 
- Large projects: chunk/RAG, not full codebase in context 
- keep_alive=2m frees VRAM when idle 
