#!/bin/zsh
# InjecAgent grid: 6 agent models x {dh single-shot, ds chained} -> rdeval_records.jsonl (programmatic ASR).
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
  $PY src/run_injecagent.py "$M" 10
done
echo "ALL INJECAGENT DONE"
