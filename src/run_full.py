#!/usr/bin/env python3
"""Full-benchmark driver: InjecAgent (n=255, base+enhanced) for all models,
then AgentDojo all-suites for the faster models (large dense models crawl on live
AgentDojo, so 32B is skipped and Mistral runs last). Appends to rdeval_records.jsonl.

Usage:
  run_full.py                 # full run (n=255, all models)
  run_full.py <IA_N> <LIMIT>  # smoke, e.g. `run_full.py 2 1` = n=2, first model only
"""
import sys, time, gc, traceback
from dart.agent import HiddenLM
from dart.injecagent import run_injecagent
from dart.datasets import run_agentdojo

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

# AgentDojo suites with a safe n (<= user_tasks//2): slack21 wksp40 bank16 trav20
AD_SUITES = {"slack": 10, "workspace": 20, "banking": 8, "travel": 10}
AD_SKIP   = {"Qwen3-32B-8bit"}          # 32B stalls on live AgentDojo rollouts
AD_LAST   = "Mistral-Small-24B-Instruct-2501-8bit"   # large dense; slow on live AD

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

# ---- Phase B: AgentDojo all-suites (faster models; 32B skipped, Mistral last) ----
ad_models = [m for m in MODELS if m.split("/")[-1] not in AD_SKIP]
ad_models.sort(key=lambda m: m.split("/")[-1] == AD_LAST)   # Mistral last
log(f"=== PHASE B: AgentDojo suites {list(AD_SUITES)}, {len(ad_models)} models ===")
for mid in ad_models:
    name = mid.split("/")[-1]; t0 = time.time()
    log(f"{name}: loading (AgentDojo)")
    try:
        lm = HiddenLM(mid)
    except Exception as e:
        log(f"{name}: LOAD FAILED {e!r}"); continue
    for suite, n in AD_SUITES.items():
        try:
            log(f"{name}: agentdojo {suite} n={n}")
            run_agentdojo(lm, name, n, suite_name=suite)
            log(f"{name}: agentdojo {suite} DONE (+{time.time()-t0:.0f}s)")
        except Exception:
            log(f"{name}: agentdojo {suite} FAILED\n{traceback.format_exc()}")
    free(lm); del lm; gc.collect()
    log(f"{name}: PhaseB done (+{time.time()-t0:.0f}s)")

log("=== ALL DONE ===")
