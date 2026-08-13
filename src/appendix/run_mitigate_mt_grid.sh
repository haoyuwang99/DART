#!/bin/bash
# Phase 2: MT-AgentRisk closed-loop mitigation, all 6 models. Fresh mitigate_mt_results.jsonl.
# Decomposition is a jailbreak: all models are candidates (ASR_off = unguarded compliance).
cd "/Users/haoyu/LLM playground"
source .venv311/bin/activate 2>/dev/null
: > mitigate_mt_results.jsonl
NFIT=${1:-40}; NTEST=${2:-12}
for m in Qwen3-8B-8bit Qwen3-14B-8bit Qwen3-30B-A3B-8bit Meta-Llama-3.1-8B-Instruct-8bit Mistral-Small-24B-Instruct-2501-8bit Qwen3-32B-8bit; do
  echo "@@@@ $m @@@@"
  python mitigate_mt.py "mlx-community/$m" "$NFIT" "$NTEST" 2>&1 | grep -vE "Fetching|it/s|â|%\|"
done
echo "MT MITIGATE GRID DONE"
