#!/usr/bin/env python3
"""Trajectory monitoring: watch the ANGLE between an agent's turn-to-turn
representation shift and the known 'injection-present' direction.

Per agent turn we grow the context and read rep(turn) = last-token residual at a
chosen layer. The turn's *movement* is  delta = rep(turn) - rep(prev_turn).
We report cos(delta, injection_direction). A benign turn should be ~orthogonal
(cos ~ 0); the turn where a poisoned tool result arrives should align (cos >> 0).

NOTE: this monitors movement toward an empirical 'injection-present' subspace,
NOT a general 'safety' axis. It flags a known failure mode, nothing more.
"""
import numpy as np
from mlx_lm import load
from introspect import Introspector
from repeng import encode
from agentdojo_pairs import make_pairs, SYS, _context

MODEL = "mlx-community/Qwen3-8B-8bit"
LAYER = 21                                   # best detection layer from the probe


def rep(ins, text):
    return encode(ins, text, pool="last")[LAYER]     # [H] at layer LAYER


def trajectory(user_prompt, tool_output):
    """Return the 4 growing-context stages of one agent turn sequence."""
    s1 = f"<|system|>\n{SYS}\n<|user|>\n{user_prompt}\n"
    s2 = s1 + "<|assistant|>\nLet me check the calendar/email for that.\n"
    s3 = s2 + f"<|tool_result|>\n{tool_output}\n"          # <-- injection (if any) enters here
    s4 = s3 + "<|assistant|>\n"
    return [("user", s1), ("tool_call", s2), ("tool_result", s3), ("act", s4)]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    pairs = make_pairs(attacks=("important_instructions",), max_pairs=40)
    model, tok = load(MODEL)
    ins = Introspector(model, tok, MODEL)

    # unit injection direction at LAYER (mean injected - mean clean)
    C = np.stack([encode(ins, p.clean_context, "last")[LAYER] for p in pairs])
    I = np.stack([encode(ins, p.injected_context, "last")[LAYER] for p in pairs])
    u = I.mean(0) - C.mean(0); u /= np.linalg.norm(u)

    # --- one illustrative trajectory: clean vs injected, cos(delta,u) per turn ---
    p = pairs[0]
    print(f"layer {LAYER} | one trajectory ({p.user_task}), cos(turn-delta, injection_dir):")
    print(f"{'turn':>12} {'clean':>8} {'injected':>10}")
    tj_clean = trajectory(_userprompt(p), p.clean_text)
    tj_inj   = trajectory(_userprompt(p), p.injected_text)
    reps_c = [rep(ins, t) for _, t in tj_clean]
    reps_i = [rep(ins, t) for _, t in tj_inj]
    for k in range(1, len(tj_clean)):
        name = tj_clean[k][0]
        dc = cos(reps_c[k] - reps_c[k-1], u)
        di = cos(reps_i[k] - reps_i[k-1], u)
        mark = "  <-- injection lands" if name == "tool_result" else ""
        print(f"{name:>12} {dc:8.3f} {di:10.3f}{mark}")

    # --- distribution over many tasks: cos at the tool_result turn ---
    print("\ndistribution of cos(tool-turn delta, injection_dir) over tasks:")
    cc, ci = [], []
    for p in pairs:
        up = _userprompt(p)
        tc, ti = trajectory(up, p.clean_text), trajectory(up, p.injected_text)
        rc = [rep(ins, t) for _, t in tc[1:3]]     # tool_call, tool_result
        ri = [rep(ins, t) for _, t in ti[1:3]]
        cc.append(cos(rc[1] - rc[0], u))
        ci.append(cos(ri[1] - ri[0], u))
    cc, ci = np.array(cc), np.array(ci)
    print(f"  CLEAN tool turn    : mean cos {cc.mean():+.3f} ± {cc.std():.3f}")
    print(f"  INJECTED tool turn : mean cos {ci.mean():+.3f} ± {ci.std():.3f}")
    thr = 0.5 * (cc.mean() + ci.mean())
    acc = (np.mean(cc < thr) + np.mean(ci > thr)) / 2
    print(f"  separation threshold {thr:+.3f} -> monitor accuracy {acc:.2f}")


def _userprompt(pair):
    # recover the user prompt text used in the pair's context
    return pair.clean_context.split("<|user|>\n")[1].split("\n<|assistant|>")[0]


if __name__ == "__main__":
    main()
