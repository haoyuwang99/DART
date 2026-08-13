#!/bin/bash
# Corrected (length-controlled) MT for all 6 models, + Mistral's AgentDojo (the one gap).
# AgentDojo (5 models) and R-Judge (all 6) are already complete elsewhere.
cd "/Users/haoyu/LLM playground"
source .venv311/bin/activate 2>/dev/null
MODELS=(
  mlx-community/Qwen3-8B-8bit
  mlx-community/Qwen3-14B-8bit
  mlx-community/Qwen3-32B-8bit
  mlx-community/Qwen3-30B-A3B-8bit
  mlx-community/Meta-Llama-3.1-8B-Instruct-8bit
)
for m in "${MODELS[@]}"; do
  echo "@@@@ ${m##*/} @@@@"
  python detect.py "$m" mt 60 2>&1 | grep -v -E "Fetching|it/s|â|%\|"
done
# Mistral: needs AgentDojo (crashed pre-fallback) + MT, both with the _ids fallback now in place.
echo "@@@@ Mistral-Small-24B-Instruct-2501-8bit @@@@"
python detect.py mlx-community/Mistral-Small-24B-Instruct-2501-8bit agentdojo,mt 25 2>&1 | grep -v -E "Fetching|it/s|â|%\|"
echo "MT GRID DONE"
