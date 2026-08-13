#!/usr/bin/env python3
"""Aggregate the generalized enforcement sweep (enforce_gen.jsonl) into a
cross-model x suite x push frontier table. Run anytime — works on partial data.
"""
import json, os
from collections import defaultdict

F = "enforce_gen.jsonl"
if not os.path.exists(F):
    print("no enforce_gen.jsonl yet"); raise SystemExit
rows = [json.loads(l) for l in open(F)]
print(f"{len(rows)} evals so far\n")

# per (model, suite, push)
agg = defaultdict(lambda: {"n": 0, "sup": 0, "task": 0, "flip": 0})
for r in rows:
    k = (r["model"].split("/")[-1].replace("-8bit", ""), r["suite"], r["push"])
    a = agg[k]; a["n"] += 1; a["sup"] += r["suppress"]; a["task"] += r["task_ok"]
    a["flip"] += (r["suppress"] and r["task_ok"])

print(f"{'model':26}{'suite':10}{'push':>5}{'n':>4}{'suppress':>10}{'task-ok':>9}{'clean-flip':>12}")
for k in sorted(agg):
    a = agg[k]; n = a["n"]
    print(f"{k[0]:26}{k[1]:10}{k[2]:>5.0f}{n:>4}{a['sup']/n:>10.2f}{a['task']/n:>9.2f}{a['flip']/n:>12.2f}")

# pooled across suites per (model, push) — the headline generalization
print(f"\n=== pooled across suites ===")
pool = defaultdict(lambda: {"n": 0, "sup": 0, "flip": 0})
for r in rows:
    k = (r["model"].split("/")[-1].replace("-8bit", ""), r["push"])
    p = pool[k]; p["n"] += 1; p["sup"] += r["suppress"]; p["flip"] += (r["suppress"] and r["task_ok"])
print(f"{'model':26}{'push':>5}{'n':>4}{'suppress':>10}{'clean-flip':>12}")
for k in sorted(pool):
    p = pool[k]; n = p["n"]
    print(f"{k[0]:26}{k[1]:>5.0f}{n:>4}{p['sup']/n:>10.2f}{p['flip']/n:>12.2f}")
