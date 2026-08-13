#!/usr/bin/env python3
"""CLI: run the Agent-SafetyBench dataset adapter for one model (fit u on ASB, then test
{none,reminder,redact,steer} via the real ASB executable environments). Usage: run_asb.py <model_id> [n=15]"""
import sys
from dart.agent import HiddenLM
from dart.asb import run_asb
model_id = sys.argv[1]; name = model_id.split("/")[-1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
print(f"[asb-run] {name} n={n}", flush=True)
run_asb(HiddenLM(model_id), name, n)
print(f"{name}: asb done", flush=True)
