#!/usr/bin/env python3
"""CLI: run InjecAgent for one model. Usage: run_injecagent.py <model_id> [n=10] [variant=base|enhanced]"""
import sys
from dart.agent import HiddenLM
from dart.injecagent import run_injecagent
mid = sys.argv[1]; name = mid.split("/")[-1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
variant = sys.argv[3] if len(sys.argv) > 3 else "base"
print(f"[injecagent-run] {name} n={n} variant={variant}", flush=True)
run_injecagent(HiddenLM(mid), name, n, variant=variant)
print(f"{name}: injecagent done", flush=True)
