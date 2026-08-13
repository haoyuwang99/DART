#!/bin/zsh
# Regenerate AgentDojo records (with redact enforcement) for the 5 remaining models.
# Highest off-ASR first so the redaction mechanism can be validated early.
cd "/Users/haoyu/LLM playground"
PY=.venv311/bin/python
for M in \
  mlx-community/Mistral-Small-24B-Instruct-2501-8bit \
  mlx-community/Qwen3-32B-8bit \
  mlx-community/Qwen3-30B-A3B-8bit \
  mlx-community/Qwen3-14B-8bit \
  mlx-community/Meta-Llama-3.1-8B-Instruct-8bit ; do
  echo "===== $M ====="
  $PY src/rdeval.py "$M" agentdojo 10
done
echo "ALL REDACT-GRID MODELS DONE"
