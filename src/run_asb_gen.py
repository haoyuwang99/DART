#!/usr/bin/env python3
"""CLI: generate + STORE ASB trajectories for one agent model (judge pipeline, phase 1).
Usage: run_asb_gen.py <model_id> [n=10]. Judge them offline with judge_asb.py."""
import sys
from dart.agent import HiddenLM
from dart.asb import asb_generate
model_id = sys.argv[1]; name = model_id.split("/")[-1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
print(f"[asb-gen] {name} n={n}", flush=True)
asb_generate(HiddenLM(model_id), name, n)
print(f"{name}: asb gen done", flush=True)
