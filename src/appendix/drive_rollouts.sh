#!/bin/bash
# Sequentially roll out the remaining fleet models across all 4 AgentDojo suites.
# Each: start mlx_lm.server -> run_rollout.py (resumable) -> stop server.
set -u
cd "/Users/haoyu/LLM playground"
PY=./.venv311/bin/python
MODELS=(
  "mlx-community/Qwen3-14B-8bit"
  "mlx-community/Qwen3-32B-8bit"
  "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
  "mlx-community/Mistral-Small-24B-Instruct-2501-8bit"
)
for M in "${MODELS[@]}"; do
  echo "=========== SERVER: $M ==========="
  $PY -m mlx_lm server --model "$M" --port 8082 > /tmp/srv_drive.log 2>&1 &
  SRV=$!
  for i in $(seq 1 60); do curl -s http://localhost:8082/v1/models >/dev/null 2>&1 && break; sleep 4; done
  echo "=========== ROLLOUT: $M ==========="
  $PY run_rollout.py "$M" 8082 traces_campaign.jsonl workspace,banking,slack,travel 2>&1 | grep -vE "NotOpenSSLWarning|warnings.warn"
  kill $SRV 2>/dev/null; sleep 4
  echo "=========== DONE: $M ==========="
done
echo "DRIVER COMPLETE"
