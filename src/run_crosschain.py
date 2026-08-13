#!/usr/bin/env python3
"""CLI: run the self-authored cross-integration chained-injection eval. Usage: run_crosschain.py <model_id>"""
import sys
from dart.agent import HiddenLM
from dart.crosschain import run_crosschain
mid = sys.argv[1]; name = mid.split("/")[-1]
print(f"[crosschain-run] {name}", flush=True)
run_crosschain(HiddenLM(mid), name)
print(f"{name}: crosschain done", flush=True)
