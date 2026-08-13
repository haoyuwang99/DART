#!/bin/bash
# Closed-loop mitigation across the attackable cells (from campaign-trace compliance analysis).
# slack is the most attackable suite; 32B>14B>30B-A3B by unguarded compliance. n_it=1 = injection_task_1.
cd "/Users/haoyu/LLM playground"
source .venv311/bin/activate 2>/dev/null
# 14B/slack already in mitigate_results.jsonl; append the remaining cells (do NOT wipe).
NUT=${1:-12}
for cell in "Qwen3-32B-8bit slack" "Qwen3-30B-A3B-8bit slack" "Qwen3-32B-8bit banking"; do
  set -- $cell; m=$1; s=$2
  echo "@@@@ $m / $s @@@@"
  python mitigate.py "mlx-community/$m" "$s" "$NUT" 1 2>&1 | grep -vE "Fetching|it/s|â|%\|"
done
echo "MITIGATE GRID DONE"
