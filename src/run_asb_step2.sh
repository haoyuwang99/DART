#!/bin/zsh
# Step 2: re-fit the reading direction on JUDGE labels. Probe all 6 agents (store hidden reps),
# harm-judge the probe trajectories once (gpt-oss-120b), then matched-CV re-fit judge-vs-fulfillable.
cd "/Users/haoyu/LLM playground"
export PYTHONPATH=src
PY=.venv311/bin/python
N=150
MODELS=(
  mlx-community/Qwen3-8B-8bit
  mlx-community/Qwen3-14B-8bit
  mlx-community/Qwen3-30B-A3B-8bit
  mlx-community/Meta-Llama-3.1-8B-Instruct-8bit
  mlx-community/Qwen3-32B-8bit
  mlx-community/Mistral-Small-24B-Instruct-2501-8bit
)
: > log/asb_probe_harm.jsonl
echo "=== STEP2 PHASE A: PROBE (store hidden reps) ==="
for M in $MODELS; do echo "===== $M ====="; $PY src/run_asb_probe.py "$M" $N; done
echo "PROBE ALL DONE"
echo "=== STEP2 PHASE B: HARM-JUDGE PROBE TRAJECTORIES ==="
$PY src/asb_refit.py judge $MODELS
echo "REFIT JUDGE DONE"
echo "=== STEP2 PHASE C: RE-FIT REPORT ==="
$PY src/asb_refit.py refit
echo "STEP2 DONE"
