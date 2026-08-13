#!/bin/zsh
# Re-run redact augmentation for the 4 remaining models (8B + Mistral already complete),
# memory-bounded (fresh process per model + MLX cache cap + clear_cache between cases).
# Smaller/faster models first to de-risk; 32B last.
cd "/Users/haoyu/LLM playground"
PY=.venv311/bin/python
for M in \
  mlx-community/Qwen3-30B-A3B-8bit \
  mlx-community/Qwen3-14B-8bit \
  mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  mlx-community/Qwen3-32B-8bit ; do
  echo "===== $M ====="
  $PY src/augment_redact.py "$M" 10
done
echo "ALL AUGMENT2 MODELS DONE"
