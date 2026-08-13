#!/usr/bin/env python3
"""Lean redact augmentation: add the `redact` column to an existing model's AgentDojo records without
re-running the off/verbalize/directive rollouts (copied from the pre-redact backup). Per test case we
run only 2 rollouts: off (to locate the peak segment) + redact-at-peak. Val records are copied with
redact=off. Emits full merged records to rdeval_records.jsonl (same schema as rdeval.emit).

Usage: augment_redact.py <model_id> [n=10] [backup=rdeval_records.pre_redact.bak.jsonl]
"""
import sys, json, numpy as np
import mlx.core as mx
from dart.agent import HiddenLM, SYS as AD_SYS
from dart.datasets import ad_fit, ad_rollout, AD_TEST
from dart.eval import emit
mx.set_cache_limit(4 * 1024**3)                                          # cap MLX buffer cache at 4 GB

model_id = sys.argv[1]; name = model_id.split("/")[-1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
BAK = sys.argv[3] if len(sys.argv) > 3 else "rdeval_records.pre_redact.bak.jsonl"

bak = [json.loads(l) for l in open(BAK)]
bak = [r for r in bak if r["model"] == name and r["dataset"] == "agentdojo"]
val_bak = [r for r in bak if r["split"] == "val"]
test_bak = [r for r in bak if r["split"] == "test"]                      # order: per ut -> harmful, benign
assert test_bak, f"no backup agentdojo records for {name}"

lm = HiddenLM(model_id); u, L = ad_fit(lm, name)
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
suite = get_suites("v1")[AD_TEST]
attack = ImportantInstructionsAttack(suite, type("P", (), {"name": "local"})())
it = list(suite.injection_tasks.values())[0]
test_uts = list(suite.user_tasks.values())[:n]
print(f"[augment] {name}: L*={L}  copying {len(val_bak)} val + verb/direc from backup, "
      f"computing redact for {len(test_uts)} uts", flush=True)

for r in val_bak:                                                        # copy val (redact=off, N/A for calib)
    emit(name, "agentdojo", "val", "benign", r["score"], r["off"], r["verbalize"], r["directive"],
         redact=r["off"], as_fired=r.get("as_fired", 0))

bi = iter(test_bak)
for ut in test_uts:
    inj = attack.attack(ut, it)
    for kind, injections, task in (("harmful", inj, it), ("benign", {k: "" for k in inj}, None)):
        b = next(bi)                                                     # matching backup record (same order)
        s_off, sec, util, tr = ad_rollout(lm, suite, AD_SYS, ut, injections, task, u, L, None, None)
        score = max(s_off) if s_off else -1e9; peak = int(np.argmax(s_off)) if s_off else 0
        off = int(sec) if kind == "harmful" else int(not util)
        _, sec_r, util_r, _ = ad_rollout(lm, suite, AD_SYS, ut, injections, task, u, L, None, None, redact_at=peak)
        redact = int(sec_r) if kind == "harmful" else int(not util_r)
        emit(name, "agentdojo", "test", kind, score, off, b["verbalize"], b["directive"],
             redact=redact, as_fired=b.get("as_fired", 0))
        print(f"  {kind:8} score={score:6.2f} off={off} verb={b['verbalize']} direc={b['directive']} redact={redact}"
              f"  [mem {round(mx.get_active_memory()/1e9,1)}G]", flush=True)
        mx.clear_cache()                                                # release buffer pool between cases
print(f"[augment] {name}: done", flush=True)
