#!/usr/bin/env python3
"""Full audit trail: how the 'injection' direction is COLLECTED and USED to monitor.
Prints real data + numbers at each step so nothing is hidden."""
import numpy as np
from mlx_lm import load
from introspect import Introspector
from repeng import encode
from agentdojo_pairs import make_pairs

MODEL = "mlx-community/Qwen3-8B-8bit"
LAYER = 21

def L2(x): return float(np.linalg.norm(x))

print("="*74)
print("STEP 1 — STIMULUS: matched (clean, injected) contrast pairs from AgentDojo")
print("="*74)
train = make_pairs(attacks=("important_instructions",), max_pairs=40)
xattack = make_pairs(attacks=("ignore_previous",), max_pairs=40)
p = train[0]
def tail(s, n=220): return s[-n:].replace("\n", " ")
print(f"pairs: {len(train)} (train attack=important_instructions), "
      f"{len(xattack)} (held-out attack=ignore_previous)")
print(f"\nexample pair — user_task={p.user_task}")
print("  CLEAN  tool_result tail :", tail(p.clean_text, 90))
print("  INJECT tool_result tail :", tail(p.injected_text, 200))

print("\n" + "="*74)
print("STEP 2 — ENCODE each context (forward pass, read residual stream @ layer", LAYER, ")")
print("="*74)
model, tok = load(MODEL)
ins = Introspector(model, tok, MODEL)
# split train into FIT (collect direction) and HELD-OUT (evaluate)
cut = 28
fit, heldout = train[:cut], train[cut:]
Cf = np.stack([encode(ins, q.clean_context, "last")[LAYER] for q in fit])     # [28,H]
If = np.stack([encode(ins, q.injected_context, "last")[LAYER] for q in fit])
print(f"encoded FIT set: {Cf.shape[0]} clean + {If.shape[0]} injected")
print(f"each context -> one vector of dim {Cf.shape[1]} (layer {LAYER} residual, last token)")
print(f"example clean vector norm={L2(Cf[0]):.1f}, injected vector norm={L2(If[0]):.1f}")

print("\n" + "="*74)
print("STEP 3 — COLLECT the direction:  d = mean(injected) - mean(clean)")
print("="*74)
mu_c, mu_i = Cf.mean(0), If.mean(0)
d = mu_i - mu_c
u = d / L2(d)                        # unit 'injection' direction
print(f"mean(clean)   norm = {L2(mu_c):.2f}")
print(f"mean(injected)norm = {L2(mu_i):.2f}")
print(f"direction d   norm = {L2(d):.2f}   (this is what we save; u = d/|d| is unit)")
print(f"angle between the two class means: "
      f"{np.degrees(np.arccos(mu_c@mu_i/(L2(mu_c)*L2(mu_i)))):.1f} deg")

# threshold = midpoint between the two class means projected on u
proj_c_fit = Cf @ u; proj_i_fit = If @ u
thr = 0.5*(proj_c_fit.mean() + proj_i_fit.mean())
print(f"\nFIT projections onto u (score = rep · u):")
print(f"  clean    : mean {proj_c_fit.mean():+.2f}  range [{proj_c_fit.min():+.2f},{proj_c_fit.max():+.2f}]")
print(f"  injected : mean {proj_i_fit.mean():+.2f}  range [{proj_i_fit.min():+.2f},{proj_i_fit.max():+.2f}]")
print(f"  DECISION THRESHOLD = {thr:+.2f}  (score>thr => 'injected')")

print("\n" + "="*74)
print("STEP 4 — USE it to MONITOR: score = rep·u, worked on 2 held-out contexts")
print("="*74)
c0 = encode(ins, heldout[0].clean_context, "last")[LAYER]
i0 = encode(ins, heldout[0].injected_context, "last")[LAYER]
sc, si = c0@u, i0@u
print(f"held-out CLEAN   context: score {sc:+.2f}  -> {'INJECTED' if sc>thr else 'clean'} "
      f"({'WRONG' if sc>thr else 'correct'})")
print(f"held-out INJECTED context: score {si:+.2f}  -> {'INJECTED' if si>thr else 'clean'} "
      f"({'correct' if si>thr else 'WRONG'})")

print("\n" + "="*74)
print("STEP 5 — EVALUATE the monitor")
print("="*74)
def evalset(pairs, tag):
    C = np.stack([encode(ins, q.clean_context, "last")[LAYER] for q in pairs]) @ u
    I = np.stack([encode(ins, q.injected_context, "last")[LAYER] for q in pairs]) @ u
    acc = (np.mean(C < thr) + np.mean(I > thr))/2
    print(f"{tag:32} clean {C.mean():+.2f} | injected {I.mean():+.2f} | acc {acc:.2f}")
evalset(heldout, "HELD-OUT (same attack, new tasks)")
evalset(xattack, "CROSS-ATTACK (ignore_previous)")
np.savez_compressed("safety_direction.npz", direction=u, raw=d, layer=LAYER, threshold=thr)
print("\nsaved safety_direction.npz  (u, raw d, layer, threshold)")
