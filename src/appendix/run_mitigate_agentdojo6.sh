#!/bin/bash
# Phase 1: closed-loop mitigation, all 6 models x {slack, banking}. Appends to mitigate_results.jsonl.
# Already done (kept): 14B/slack, 32B/slack, 30B-A3B/slack, 32B/banking. This runs the remaining 8 cells.
# Injection-robust models (Mistral, 8B) will show ASR_off~=0 -- that is the honest finding, not a bug.
cd "/Users/haoyu/LLM playground"
source .venv311/bin/activate 2>/dev/null
NUT=${1:-12}
CELLS=(
  "Qwen3-8B-8bit slack"
  "Meta-Llama-3.1-8B-Instruct-8bit slack"
  "Mistral-Small-24B-Instruct-2501-8bit slack"
  "Qwen3-8B-8bit banking"
  "Qwen3-14B-8bit banking"
  "Qwen3-30B-A3B-8bit banking"
  "Meta-Llama-3.1-8B-Instruct-8bit banking"
  "Mistral-Small-24B-Instruct-2501-8bit banking"
)
for cell in "${CELLS[@]}"; do
  set -- $cell; m=$1; s=$2
  echo "@@@@ $m / $s @@@@"
  python mitigate.py "mlx-community/$m" "$s" "$NUT" 1 2>&1 | grep -vE "Fetching|it/s|â|%\|"
done
echo "AGENTDOJO6 MITIGATE DONE"
