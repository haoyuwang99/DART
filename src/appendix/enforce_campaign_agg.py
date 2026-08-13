#!/usr/bin/env python3
"""Aggregate enforce_campaign.jsonl. Works on partial data. Three views:
  1. LOSO frontier   per (model,suite,push): suppress / task-ok / clean-flip
  2. LOSO pooled     per (model,push) across suites: clean-flip (the headline generalization)
  3. Controls @PUSH=1 per model: clean-flip for loso vs indomain vs random vs reminder
"""
import json, os
from collections import defaultdict

F = "enforce_campaign.jsonl"
if not os.path.exists(F):
    print("no enforce_campaign.jsonl yet"); raise SystemExit
rows = [json.loads(l) for l in open(F)]
short = lambda m: m.split("/")[-1].replace("-8bit", "")
print(f"{len(rows)} evals so far\n")

# 1. LOSO frontier per (model,suite,push)
print("=== LOSO frontier: per suite ===")
agg = defaultdict(lambda: {"n": 0, "sup": 0, "task": 0, "flip": 0})
for r in rows:
    if r["method"] != "loso":
        continue
    a = agg[(short(r["model"]), r["suite"], r["push"])]
    a["n"] += 1; a["sup"] += r["suppress"]; a["task"] += r["task_ok"]; a["flip"] += r["flip"]
print(f"{'model':16}{'suite':10}{'push':>5}{'n':>4}{'suppress':>10}{'task-ok':>9}{'clean-flip':>12}")
for k in sorted(agg):
    a = agg[k]; n = a["n"]
    print(f"{k[0]:16}{k[1]:10}{k[2]:>5.0f}{n:>4}{a['sup']/n:>10.2f}{a['task']/n:>9.2f}{a['flip']/n:>12.2f}")

# 2. LOSO pooled across suites per (model,push)
print(f"\n=== LOSO frontier: pooled across suites (headline) ===")
pool = defaultdict(lambda: {"n": 0, "sup": 0, "flip": 0})
for r in rows:
    if r["method"] != "loso":
        continue
    p = pool[(short(r["model"]), r["push"])]
    p["n"] += 1; p["sup"] += r["suppress"]; p["flip"] += r["flip"]
print(f"{'model':16}{'push':>5}{'n':>4}{'suppress':>10}{'clean-flip':>12}")
for k in sorted(pool):
    p = pool[k]; n = p["n"]
    print(f"{k[0]:16}{k[1]:>5.0f}{n:>4}{p['sup']/n:>10.2f}{p['flip']/n:>12.2f}")

# 3. Controls at PUSH=1 per model: clean-flip loso vs indomain vs random vs reminder
print(f"\n=== operating point (PUSH=1): method comparison, clean-flip ===")
ctrl = defaultdict(lambda: {"n": 0, "flip": 0, "sup": 0})
for r in rows:
    if r["push"] != 1.0:
        continue
    c = ctrl[(short(r["model"]), r["method"])]
    c["n"] += 1; c["flip"] += r["flip"]; c["sup"] += r["suppress"]
models = sorted({k[0] for k in ctrl})
methods = ["loso", "indomain", "random", "reminder"]
print(f"{'model':16}" + "".join(f"{m:>11}" for m in methods))
for mdl in models:
    cells = []
    for meth in methods:
        c = ctrl.get((mdl, meth))
        cells.append(f"{c['flip']/c['n']:.2f}" if c and c["n"] else "  - ")
    print(f"{mdl:16}" + "".join(f"{x:>11}" for x in cells))
print("(cell = clean-flip rate = suppress AND still-on-task; n per method ~= complying cases/model)")
