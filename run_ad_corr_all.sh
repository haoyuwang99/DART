cd "/Users/haoyu/LLM playground"
export PYTHONPATH=src
MODELS="Qwen3-8B-8bit Qwen3-14B-8bit Qwen3-30B-A3B-8bit Meta-Llama-3.1-8B-Instruct-8bit Mistral-Small-24B-Instruct-2501-8bit Qwen3-32B-8bit"
echo "[driver $(date '+%m-%d %H:%M')] === PHASE 1: slack (fast) all models ==="
for M in $MODELS; do
  echo "[driver $(date '+%m-%d %H:%M')] P1 $M slack"
  .venv311/bin/python src/run_ad_corrected.py mlx-community/$M slack || echo "[driver] $M slack FAILED"
done
echo "[driver $(date '+%m-%d %H:%M')] === PHASE 2: banking,travel,workspace (slow) all models ==="
for M in $MODELS; do
  echo "[driver $(date '+%m-%d %H:%M')] P2 $M banking,travel,workspace"
  .venv311/bin/python src/run_ad_corrected.py mlx-community/$M banking,travel,workspace || echo "[driver] $M rest FAILED"
done
echo "[driver $(date '+%m-%d %H:%M')] === ALL DONE ==="
