#!/usr/bin/env python3
"""Fast validation: attributed reminder vs generic reminder on AgentDojo injection (harmful cases).
Does quoting the leave-one-span-out span recover the suppression the generic reminder lost? Runs
none / generic / attributed on the SAME cases for a clean head-to-head. Usage: <model_id> [n=10]"""
import sys, json, numpy as np
import mlx.core as mx
from dart.agent import HiddenLM, SYS as AD_SYS
from dart.datasets import ad_fit, ad_rollout, AD_TEST
from dart.mitigation import AD_VERBALIZE, localize_injection_text
mx.set_cache_limit(4 * 1024**3)

model_id = sys.argv[1]; name = model_id.split("/")[-1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
lm = HiddenLM(model_id); u, L = ad_fit(lm, name)
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
suite = get_suites("v1")[AD_TEST]
attack = ImportantInstructionsAttack(suite, type("P", (), {"name": "local"})())
it = list(suite.injection_tasks.values())[0]
test_uts = list(suite.user_tasks.values())[:n]
print(f"[attr-val] {name} L*={L}  {len(test_uts)} harmful uts", flush=True)

none_, gen_, attr_ = [], [], []
for ut in test_uts:
    inj = attack.attack(ut, it)
    s_off, sec, util, tr = ad_rollout(lm, suite, AD_SYS, ut, inj, it, u, L, None, None)
    peak = int(np.argmax(s_off)) if s_off else 0
    _, sec_g, _, _ = ad_rollout(lm, suite, AD_SYS, ut, inj, it, u, L, peak, AD_VERBALIZE)            # generic
    _, sec_a, _, _ = ad_rollout(lm, suite, AD_SYS, ut, inj, it, u, L, peak, None, attributed=True)   # attributed
    none_.append(int(sec)); gen_.append(int(sec_g)); attr_.append(int(sec_a))
    print(f"  none={int(sec)} generic={int(sec_g)} attributed={int(sec_a)}   [mem {round(mx.get_active_memory()/1e9,1)}G]", flush=True)
    mx.clear_cache()
print(f"\nASR: none={np.mean(none_):.2f}  generic_reminder={np.mean(gen_):.2f}  "
      f"attributed_reminder={np.mean(attr_):.2f}   (redact was 0.00)", flush=True)
