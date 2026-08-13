#!/usr/bin/env python3
"""Add the `steer` column (RepE steering, alpha=2) to AgentDojo records, measuring BOTH harmful (ASR)
and benign (utility) -- so we see whether steering suppresses cleanly or by corrupting generation.
Merges in place (order-matched). MT steering handled separately. Usage: <model_id> [n=10] [alpha=2]"""
import sys, json, numpy as np
import mlx.core as mx
from dart.agent import HiddenLM, SYS as AD_SYS
from dart.datasets import ad_fit, ad_fit_steer, ad_rollout, AD_TEST
from dart.eval import RECORDS
mx.set_cache_limit(4 * 1024**3)

model_id = sys.argv[1]; name = model_id.split("/")[-1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
lm = HiddenLM(model_id)
u, L = ad_fit(lm, name)
u_s, Ls, cref, gap = ad_fit_steer(lm, name, frac=0.5)
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
suite = get_suites("v1")[AD_TEST]
attack = ImportantInstructionsAttack(suite, type("P", (), {"name": "local"})())
it = list(suite.injection_tasks.values())[0]
print(f"[augment-steer] {name} L={L} Ls={Ls} gap={gap:.2f} alpha={alpha}", flush=True)

out = {"harmful": [], "benign": []}
for ut in list(suite.user_tasks.values())[:n]:
    inj = attack.attack(ut, it)
    for kind, injections, task in (("harmful", inj, it), ("benign", {k: "" for k in inj}, None)):
        s_off, sec, util, tr = ad_rollout(lm, suite, AD_SYS, ut, injections, task, u, L, None, None)
        peak = int(np.argmax(s_off)) if s_off else 0
        _, sec_s, util_s, _ = ad_rollout(lm, suite, AD_SYS, ut, injections, task, u, L, None, None,
                                         steer_at=peak, steer=(u_s, Ls, cref, gap, alpha))
        out[kind].append(int(sec_s) if kind == "harmful" else int(not util_s))
    print(f"  {ut.ID} h={out['harmful'][-1]} b={out['benign'][-1]}  [mem {round(mx.get_active_memory()/1e9,1)}G]",
          flush=True); mx.clear_cache()

recs = [json.loads(l) for l in open(RECORDS)]
idx = {"harmful": 0, "benign": 0}
for r in recs:
    if "steer" not in r:
        r["steer"] = r["off"]                                        # default (val / MT / other models)
    if r["model"] == name and r["dataset"] == "agentdojo" and r["split"] == "test":
        v = out[r["kind"]]
        if idx[r["kind"]] < len(v):
            r["steer"] = v[idx[r["kind"]]]; idx[r["kind"]] += 1
with open(RECORDS, "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print(f"[augment-steer] {name}: merged  ASR={np.mean(out['harmful']):.2f}  "
      f"Util(ungated)={1 - np.mean(out['benign']):.2f}  ad={out}", flush=True)
