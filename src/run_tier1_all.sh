#!/bin/zsh
# Tier-1 deepening across 6 models: InjecAgent-enhanced (dh+ds) + AgentDojo workspace/banking/travel.
cd "/Users/haoyu/LLM playground"
export PYTHONPATH=src
PY=.venv311/bin/python
for M in \
  mlx-community/Qwen3-8B-8bit \
  mlx-community/Qwen3-14B-8bit \
  mlx-community/Qwen3-30B-A3B-8bit \
  mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  mlx-community/Qwen3-32B-8bit \
  mlx-community/Mistral-Small-24B-Instruct-2501-8bit ; do
  echo "===== $M ====="
  $PY src/run_tier1.py "$M"
done
echo "ALL TIER1 DONE"
