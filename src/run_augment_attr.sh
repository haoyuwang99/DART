#!/bin/zsh
cd "/Users/haoyu/LLM playground"
PY=.venv311/bin/python
for M in \
  mlx-community/Qwen3-30B-A3B-8bit \
  mlx-community/Qwen3-14B-8bit \
  mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  mlx-community/Mistral-Small-24B-Instruct-2501-8bit \
  mlx-community/Qwen3-32B-8bit ; do
  echo "===== $M ====="
  $PY src/augment_attributed.py "$M" 10
done
echo "ALL ATTR AUGMENT DONE"
