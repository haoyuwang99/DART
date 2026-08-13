#!/bin/bash
set -u
cd "/Users/haoyu/LLM playground"; PY=./.venv311/bin/python
for M in "mlx-community/Qwen3-14B-8bit" "mlx-community/Qwen3-32B-8bit" "mlx-community/Qwen3-30B-A3B-8bit"; do
  echo "######## $M ########"
  $PY enforce_gen.py "$M" 2>&1 | grep -vE "NotOpenSSLWarning|warnings.warn|Fetching|tokenizer_config|Some kwargs|special_tokens|it/s"
done
echo "GEN COMPLETE"
