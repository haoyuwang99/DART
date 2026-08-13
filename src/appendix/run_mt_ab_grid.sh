#!/bin/bash
# MT reminder A/B: generic vs consolidate vs derived vs directive, on the key models.
# 8B & 32B are the models the generic reminder left most on the table (1.0->0.75, 1.0->0.67).
cd "/Users/haoyu/LLM playground"
source .venv311/bin/activate 2>/dev/null
: > mitigate_mt_ab_results.jsonl
NFIT=${1:-30}; NTEST=${2:-12}
for m in Qwen3-8B-8bit Qwen3-14B-8bit Qwen3-32B-8bit; do
  echo "@@@@ $m @@@@"
  python mitigate_mt.py "mlx-community/$m" "$NFIT" "$NTEST" 2>&1 | grep -vE "Fetching|it/s|â|%\|"
done
echo "MT AB GRID DONE"
