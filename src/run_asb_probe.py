#!/usr/bin/env python3
"""Step 2 generation CLI: probe one model, storing per-transition hidden reps + trajectories so the
reading direction can be re-fit on the JUDGE label offline. Usage: run_asb_probe.py <model_id> [n=120]"""
import sys
from dart.agent import HiddenLM
from dart.asb import asb_probe_run
mid = sys.argv[1]; name = mid.split("/")[-1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
print(f"[asb-probe] {name} n_cases={n}", flush=True)
asb_probe_run(HiddenLM(mid), name, n)
print(f"{name}: asb probe done", flush=True)
