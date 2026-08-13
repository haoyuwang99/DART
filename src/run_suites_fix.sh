#!/bin/zsh
cd "/Users/haoyu/LLM playground"; export PYTHONPATH=src; PY=.venv311/bin/python
for M in mlx-community/Qwen3-32B-8bit mlx-community/Mistral-Small-24B-Instruct-2501-8bit; do
  echo "===== $M ====="; $PY src/run_agentdojo_suites.py "$M"
done
echo "ALL SUITES FIX DONE"
