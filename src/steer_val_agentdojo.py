#!/usr/bin/env python3
"""Fast validation: RepE steering vs redact/attributed on AgentDojo injection (harmful cases). Steering
adds -alpha*(p-clean_ref)*u at the mid-layer during generation (RQ5's best-case config). Sweeps a couple
of alphas so we see steering's ceiling. Usage: <model_id> [n=8] [alphas=1,2,4]"""
import sys, numpy as np
import mlx.core as mx
from dart.agent import HiddenLM, SYS as AD_SYS
from dart.datasets import ad_fit, ad_fit_steer, ad_rollout, AD_TEST
from dart.mitigation import AD_VERBALIZE
mx.set_cache_limit(4 * 1024**3)

model_id = sys.argv[1]; name = model_id.split("/")[-1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
alphas = [float(a) for a in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["1", "2", "4"])]
lm = HiddenLM(model_id)
u, L = ad_fit(lm, name)
u_s, Ls, cref, gap = ad_fit_steer(lm, name, frac=0.5)
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
suite = get_suites("v1")[AD_TEST]
attack = ImportantInstructionsAttack(suite, type("P", (), {"name": "local"})())
it = list(suite.injection_tasks.values())[0]
test_uts = list(suite.user_tasks.values())[:n]
print(f"[steer-val] {name}  monitor L={L}  steer Ls={Ls}(frac0.5) clean_ref={cref:.2f} gap={gap:.2f}  alphas={alphas}", flush=True)

none_ = []; steer_ = {a: [] for a in alphas}
for ut in test_uts:
    inj = attack.attack(ut, it)
    s_off, sec, util, tr = ad_rollout(lm, suite, AD_SYS, ut, inj, it, u, L, None, None)
    peak = int(np.argmax(s_off)) if s_off else 0
    none_.append(int(sec))
    line = f"  none={int(sec)}"
    for a in alphas:
        _, sec_s, _, _ = ad_rollout(lm, suite, AD_SYS, ut, inj, it, u, L, None, None,
                                    steer_at=peak, steer=(u_s, Ls, cref, gap, a))
        steer_[a].append(int(sec_s)); line += f" steer@a{a}={int(sec_s)}"
    print(line + f"   [mem {round(mx.get_active_memory()/1e9,1)}G]", flush=True); mx.clear_cache()
print(f"\nASR: none={np.mean(none_):.2f}  " +
      "  ".join(f"steer(a{a})={np.mean(steer_[a]):.2f}" for a in alphas) +
      "   (redact 0.00, attributed 0.20 from validation)", flush=True)
