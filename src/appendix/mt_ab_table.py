#!/usr/bin/env python3
"""Format the MT reminder A/B (mitigate_mt_ab_results.jsonl) into a markdown table.

Each cell is 'harmful ASR / benign completion' under one reminder strategy. Lower ASR is better;
benign completion should stay near the off value (over-refusal drops it). Shows the safety-utility
frontier across strategies, plus the model-verbalized derived reminder."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "mitigate_mt_ab_results.jsonl"
rows = [json.loads(l) for l in open(path)]
STRATS = ["generic", "consolidate", "derived", "directive"]

print("## MT reminder A/B — harmful ASR ↓ / benign completion (same monitor, u, τ, split; only ρ varies)\n")
print("| Model | off (ASR / benign) | " + " | ".join(STRATS) + " |")
print("|---|---|" + "---|" * len(STRATS))
for r in rows:
    off = f"{r['asr_off']:.3f} / {r.get('benign_off', float('nan')):.2f}"
    cells = [f"{r.get('asr_on_' + s, float('nan')):.3f} / {r.get('benign_on_' + s, float('nan')):.2f}"
             for s in STRATS]
    print(f"| {r['model']} | {off} | " + " | ".join(cells) + " |")
print("\ncell = harmful ASR / benign completion.  generic = soft 'stop and decline'; consolidate = "
      "re-present accumulated turns as one request; derived = ρ=verbalize(𝒞); directive = imperative "
      "'SAFETY STOP, do not perform'. Same drift monitor triggers all; only the injected ρ differs.\n")
for r in rows:
    if r.get("derived_rho"):
        print(f"- **{r['model']}** derived ρ: \"{r['derived_rho']}\"")
