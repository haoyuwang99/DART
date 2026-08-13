#!/usr/bin/env python3
"""Build mt_agentrisk.jsonl from downloaded MT-AgentRisk prompt files (robust to naming variants).
harmful_single = any task.md under a tool; harmful_multi = any dir with task-turn-*.md; benign = benign_tasks/.
Excludes the Subset_100 duplicate layout. Joins Format/Method from multi_dataset.csv by normalized name.
"""
import os, glob, csv, json, re
from collections import Counter

R = "mt_agentrisk_data/workspaces"
TOOLS = ["filesystem", "notion", "playwright", "postgres", "terminal"]
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
cmeta = {norm(r["Task Name"]): r for r in csv.DictReader(open("mt_agentrisk_data/multi_dataset.csv"))}
turn_num = lambda f: int(re.search(r"task-turn-(\d+)", f).group(1)) if re.search(r"task-turn-(\d+)", f) else 999
tool_of = lambda p: p.split(R + "/")[1].split("/")[0]

recs, seen = [], set()
def add(r):
    k = (r["type"], r["tool"], r["task"])
    if k not in seen: seen.add(k); recs.append(r)

# harmful_single: any task.md under a tool dir
for f in glob.glob(f"{R}/**/task.md", recursive=True):
    if "Subset_100" in f or "benign_tasks" in f: continue
    if tool_of(f) not in TOOLS: continue
    txt = open(f).read().strip()
    if txt: add({"type": "harmful_single", "tool": tool_of(f), "task": os.path.basename(os.path.dirname(f)),
                 "format": "single", "method": "single", "turns": [txt]})

# harmful_multi: any dir with task-turn-1.md
for f1 in glob.glob(f"{R}/**/task-turn-1.md", recursive=True):
    if "Subset_100" in f1: continue
    d = os.path.dirname(f1); name = os.path.basename(d)
    turns = [open(x).read().strip() for x in sorted(glob.glob(f"{d}/task-turn-*.md"), key=turn_num) if open(x).read().strip()]
    if not turns: continue
    m = cmeta.get(norm(name), {})
    add({"type": "harmful_multi", "tool": tool_of(f1), "task": name,
         "format": m.get("Format", "?"), "method": m.get("Method", "?"), "turns": turns})

# benign
for f in glob.glob(f"{R}/benign_tasks/**/task.md", recursive=True):
    tool = f.split("benign_tasks_")[1].split("/")[0] if "benign_tasks_" in f else "?"
    txt = open(f).read().strip()
    if txt: add({"type": "benign", "tool": tool, "task": os.path.basename(os.path.dirname(f)),
                 "format": "benign", "method": "benign", "turns": [txt]})

with open("mt_agentrisk.jsonl", "w") as out:
    for r in recs: out.write(json.dumps(r) + "\n")
print("records:", len(recs))
print("by type:", dict(Counter(r["type"] for r in recs)))
print("multi by format:", dict(Counter(r["format"] for r in recs if r["type"] == "harmful_multi")))
print("multi turn-count dist:", dict(sorted(Counter(len(r["turns"]) for r in recs if r["type"] == "harmful_multi").items())))
print("by tool:", dict(Counter(r["tool"] for r in recs)))
print("format '?' (unmatched to CSV):", sum(1 for r in recs if r.get("format") == "?"))
