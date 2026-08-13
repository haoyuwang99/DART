#!/bin/bash
set -u
cd "/Users/haoyu/LLM playground"
PY=./.venv311/bin/python
FILT='NotOpenSSLWarning|warnings.warn|Fetching|tokenizer_config|Some kwargs|special_tokens|it/s'
for M in \
  "mlx-community/Qwen3-8B-8bit" \
  "mlx-community/Qwen3-14B-8bit" \
  "mlx-community/Qwen3-30B-A3B-8bit" \
  "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"; do
  echo "======== $M ========"
  $PY score_ablation.py "$M" traces_campaign.jsonl 2>&1 | grep -vE "$FILT"
done
echo "ABLATION COMPLETE"
