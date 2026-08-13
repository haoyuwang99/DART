#!/usr/bin/env python3
"""Summarize the redact column for the paper: per model (AgentDojo), ASR under each enforcement at the
deployable point tau=0, and the availability contrast at tau=1 (forced firing) where stop collapses.
Usage: redact_summary.py [records.jsonl]"""
import sys, json, collections, numpy as np
from dart.mitigation import mit_cell
recs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "rdeval_records.jsonl")]
ORDER = ["Qwen3-8B-8bit", "Qwen3-14B-8bit", "Qwen3-32B-8bit", "Qwen3-30B-A3B-8bit",
         "Meta-Llama-3.1-8B-Instruct-8bit", "Mistral-Small-24B-Instruct-2501-8bit"]
G = collections.defaultdict(list)
for r in recs:
    if r["dataset"] == "agentdojo" and "redact" in r:
        G[r["model"]].append(r)

print("AgentDojo, tau=0 (deployable): ASR by enforcement + Utility; and tau=1 Utility (forced firing)")
print(f"{'model':34}{'none':>7}{'stop':>7}{'direc':>7}{'verb':>7}{'redact':>8}{'Util0':>8}"
      f"{'Ustop@1':>9}{'Ured@1':>9}")
for m in ORDER:
    rows = G.get(m)
    if not rows:
        print(f"{m:34}  (pending)"); continue
    a_none = mit_cell(rows, "none", 0.0)[0]
    a_stop, u0, _, _ = mit_cell(rows, "stop", 0.0)
    a_dir = mit_cell(rows, "directive", 0.0)[0]
    a_verb = mit_cell(rows, "verbalize", 0.0)[0]
    a_red = mit_cell(rows, "redact", 0.0)[0]
    u_stop1 = mit_cell(rows, "stop", 1.0)[1]
    u_red1 = mit_cell(rows, "redact", 1.0)[1]
    print(f"{m:34}{a_none:>7.2f}{a_stop:>7.2f}{a_dir:>7.2f}{a_verb:>7.2f}{a_red:>8.2f}{u0:>8.2f}"
          f"{u_stop1:>9.2f}{u_red1:>9.2f}")
print("\nUtil0 = benign utility at tau=0 (no-intervention level for a precise gate).")
print("Ustop@1 / Ured@1 = benign utility when forced to fire on everything (tau=1): stop's DoS vs redact.")
