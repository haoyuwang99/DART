#!/usr/bin/env python3
"""Tier-1 deepening for one model: InjecAgent-enhanced (stronger attack) + 3 unused AgentDojo suites.
One model load, reused across both. Usage: run_tier1.py <model_id>"""
import sys
from dart.agent import HiddenLM
from dart.injecagent import run_injecagent
from dart.datasets import run_agentdojo
mid = sys.argv[1]; name = mid.split("/")[-1]
lm = HiddenLM(mid)
print(f"[tier1] {name}: injecagent ENHANCED", flush=True)
run_injecagent(lm, name, n=10, variant="enhanced")
for s in ("workspace", "banking", "travel"):
    print(f"[tier1] {name}: agentdojo {s}", flush=True)
    run_agentdojo(lm, name, 8, suite_name=s)
print(f"{name}: tier1 done", flush=True)
