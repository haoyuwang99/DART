#!/usr/bin/env python3
"""Re-run AgentDojo workspace/banking/travel for one model (with the MLX cache-limit fix).
Prints MLX active/cache memory per suite so we can verify the cache stays capped (no unbounded growth).
Usage: run_agentdojo_suites.py <model_id>"""
import sys
import mlx.core as mx
from dart.agent import HiddenLM
from dart.datasets import run_agentdojo
mid = sys.argv[1]; name = mid.split("/")[-1]
lm = HiddenLM(mid)
print(f"[mem] {name} loaded: active={mx.get_active_memory()/1e9:.1f}GB cache={mx.get_cache_memory()/1e9:.1f}GB", flush=True)
for s in ("workspace", "banking", "travel"):
    print(f"[suites] {name}: agentdojo {s}", flush=True)
    run_agentdojo(lm, name, 8, suite_name=s)
    print(f"[mem] {name} after {s}: active={mx.get_active_memory()/1e9:.1f}GB cache={mx.get_cache_memory()/1e9:.1f}GB", flush=True)
print(f"{name}: suites done", flush=True)
