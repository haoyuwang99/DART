#!/bin/bash
# Score every campaign model on AgentDojo (AUROC + leave-one-suite-out CV),
# then R-Judge (non-injection) on two scales. In-process; GPU-serial.
set -u
cd "/Users/haoyu/LLM playground"
PY=./.venv311/bin/python
FILT='NotOpenSSLWarning|warnings.warn|Fetching|tokenizer_config|Some kwargs|special_tokens|it/s'
MODELS=(
  "mlx-community/Qwen3-8B-8bit"
  "mlx-community/Qwen3-14B-8bit"
  "mlx-community/Qwen3-32B-8bit"
  "mlx-community/Qwen3-30B-A3B-8bit"
  "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
  "mlx-community/Mistral-Small-24B-Instruct-2501-8bit"
)
echo "######## AGENTDOJO (per-model AUROC + LOSO-CV) ########"
for M in "${MODELS[@]}"; do
  echo "======== $M ========"
  $PY score_campaign.py "$M" traces_campaign.jsonl 2>&1 | grep -vE "$FILT"
done
echo
echo "######## R-JUDGE (non-injection, leave-one-domain-out) ########"
for M in "mlx-community/Qwen3-8B-8bit" "mlx-community/Qwen3-32B-8bit"; do
  echo "======== R-Judge: $M ========"
  $PY rjudge_eval.py "$M" 2>&1 | grep -vE "$FILT"
done
echo "SCORING COMPLETE"
