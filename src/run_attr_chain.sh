#!/bin/zsh
cd "/Users/haoyu/LLM playground"
# wait for the already-running AgentDojo validation to finish
until grep -q "^ASR:" attr_val_ad.log 2>/dev/null; do sleep 15; done
echo "=== AgentDojo attr-val complete; starting MT (Qwen3-8B, fast) ==="
.venv311/bin/python attr_val_mt.py mlx-community/Qwen3-8B-8bit 10 > attr_val_mt.log 2>&1
echo "=== MT attr-val complete ==="
