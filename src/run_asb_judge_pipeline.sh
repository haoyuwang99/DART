#!/bin/zsh
# Full ASB semantic-judge pipeline: generate trajectories (6 agents) -> judge (gpt-oss-120b) -> report.
cd "/Users/haoyu/LLM playground"
export PYTHONPATH=src
PY=.venv311/bin/python
: > log/asb_traj.jsonl                                   # start clean (fresh trajectories)
: > log/asb_verdicts.jsonl
echo "=== PHASE 1: GENERATE ==="
for M in \
  mlx-community/Qwen3-8B-8bit \
  mlx-community/Qwen3-14B-8bit \
  mlx-community/Qwen3-30B-A3B-8bit \
  mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  mlx-community/Qwen3-32B-8bit \
  mlx-community/Mistral-Small-24B-Instruct-2501-8bit ; do
  echo "===== $M ====="
  $PY src/run_asb_gen.py "$M" 10
done
echo "GEN DONE"
echo "=== PHASE 2: JUDGE (gpt-oss-120b) ==="
$PY src/judge_asb.py judge
echo "JUDGE DONE"
echo "=== PHASE 3: REPORT ==="
$PY src/judge_asb.py report
echo "PIPELINE DONE"
