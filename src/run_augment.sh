#!/bin/zsh
# Lean redact augmentation for the 5 models not yet done (8B already has full redact records).
# Highest off-ASR first for early validation.
cd "/Users/haoyu/LLM playground"
PY=.venv311/bin/python
for M in \
  mlx-community/Mistral-Small-24B-Instruct-2501-8bit \
  mlx-community/Qwen3-32B-8bit \
  mlx-community/Qwen3-30B-A3B-8bit \
  mlx-community/Qwen3-14B-8bit \
  mlx-community/Meta-Llama-3.1-8B-Instruct-8bit ; do
  echo "===== $M ====="
  $PY src/augment_redact.py "$M" 10
done
echo "ALL AUGMENT MODELS DONE"
