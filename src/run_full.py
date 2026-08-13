#!/usr/bin/env python3
"""Full InjecAgent driver: n=255, base+enhanced, all models. Appends to rdeval_records.jsonl.
(AgentDojo now runs via src/run_online.py on the online enforcement core.)

Usage:
  run_full.py                 # full run (n=255, all models)
  run_full.py <IA_N> <LIMIT>  # smoke, e.g. `run_full.py 2 1` = n=2, first model only
"""
import sys, time, gc, traceback
from dart.agent import HiddenLM
from dart.injecagent import run_injecagent

IA_N   = int(sys.argv[1]) if len(sys.argv) > 1 else 255
LIMIT  = int(sys.argv[2]) if len(sys.argv) > 2 else 99

MODELS = [
    "mlx-community/Qwen3-8B-8bit",
    "mlx-community/Qwen3-14B-8bit",
    "mlx-community/Qwen3-30B-A3B-8bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    "mlx-community/Mistral-Small-24B-Instruct-2501-8bit",
    "mlx-community/Qwen3-32B-8bit",
][:LIMIT]

def log(m): print(f"[full {time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def free(lm):
    try: lm.clear_cache()
    except Exception: pass
    try:
        import mlx.core as mx; mx.clear_cache()
    except Exception: pass
    gc.collect()

# ---- Phase A: InjecAgent full (all models) -- the main deliverable, mock-tools ----
log(f"=== PHASE A: InjecAgent n={IA_N} (base+enhanced), {len(MODELS)} models ===")
for mid in MODELS:
    name = mid.split("/")[-1]; t0 = time.time()
    log(f"{name}: loading")
    try:
        lm = HiddenLM(mid)
    except Exception as e:
        log(f"{name}: LOAD FAILED {e!r}"); continue
    for variant in ("base", "enhanced"):
        try:
            log(f"{name}: injecagent {variant} n={IA_N}")
            run_injecagent(lm, name, IA_N, variant=variant)
            log(f"{name}: injecagent {variant} DONE (+{time.time()-t0:.0f}s)")
        except Exception:
            log(f"{name}: injecagent {variant} FAILED\n{traceback.format_exc()}")
    free(lm); del lm; gc.collect()
    log(f"{name}: PhaseA done (+{time.time()-t0:.0f}s)")

log("=== ALL DONE ===")
