#!/usr/bin/env python3
"""Confirm the steering hook actually fires (and diagnose coef magnitude). Prints projection p vs
clean_ref, then checks whether coef-based and large-fixed steering change the generated output.
Usage: diag_steer.py <model_id>"""
import sys, numpy as np
from dart.agent import HiddenLM
from dart.datasets import ad_fit_steer
lm = HiddenLM(sys.argv[1]); name = sys.argv[1].split("/")[-1]
u_s, Ls, cref = ad_fit_steer(lm, name, frac=0.5)
msgs = [{"role": "user", "content": "In one sentence, tell me a fact about the ocean."}]
lm.clear_steer(); base = lm.generate(msgs, max_tokens=40)
p = float(lm.hidden(msgs, Ls) @ u_s)
print(f"Ls={Ls}  clean_ref={cref:.2f}  p(this ctx)={p:.2f}  coef@a2={2*(p-cref):.2f}  coef@a4={4*(p-cref):.2f}")
print("UNSTEERED :", base[:110].replace(chr(10), ' '))
for a in (2.0, 4.0):
    lm.set_steer(-(a * (p - cref)) * u_s, Ls - 1); o = lm.generate(msgs, max_tokens=40); lm.clear_steer()
    print(f"coef a{a} (changed={o != base}):", o[:110].replace(chr(10), ' '))
for mag in (5.0, 20.0, 50.0):
    lm.set_steer(-mag * u_s, Ls - 1); o = lm.generate(msgs, max_tokens=40); lm.clear_steer()
    print(f"fixed -{mag}*u (changed={o != base}):", o[:110].replace(chr(10), ' '))
# also verify the hook shifts the projection it targets
lm.set_steer(-20.0 * u_s, Ls - 1)
p2 = float(lm.hidden(msgs, Ls) @ u_s); lm.clear_steer()
print(f"projection at Ls  unsteered={p:.2f}  under fixed -20*u={p2:.2f}  (should drop a lot if hook fires)")
