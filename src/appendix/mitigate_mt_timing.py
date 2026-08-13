#!/usr/bin/env python3
"""Does firing the reminder EARLIER help multi-turn defense -- especially the soft/verbalized one?

Isolates TIMING from flagging: for each test trajectory we inject a FIXED reminder at a fixed
relative turn position (early=first turn, mid, late=committing turn) and measure refusal, vs no
intervention. If ASR drops when the reminder is delivered earlier, the hypothesis holds -- an early
reminder (achieved online by lowering the accumulated-risk threshold tau) primes caution before the
agent commits, so a softer/verbalized reminder can work where a late one needs a hard stop.
Reminders: verbalized (rho=verbalize(C)) and generic.  Usage: mitigate_mt_timing.py <model> [n_fit] [n_test]
"""
import sys, numpy as np
from rd_agent import HiddenLM
from mitigate_mt import load_groups, fit_direction, is_refusal, r_generic, derive_mt_reminder, SYS


def rollout_at(lm, turns, fire_i, reminder):
    gen = [{"role": "system", "content": SYS}]; final = ""
    for i, t in enumerate(turns):
        gen = gen + [{"role": "user", "content": t}]
        if fire_i is not None and i == fire_i:
            gen = gen + [{"role": "user", "content": reminder}]
        final = lm.generate(gen, max_tokens=256)
        gen = gen + [{"role": "assistant", "content": final}]
    return final


def main():
    model_id = sys.argv[1]
    n_fit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    lm = HiddenLM(model_id)
    pairs = load_groups()
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    fit_pairs, test_pairs = pairs[:n_fit], pairs[n_fit:n_fit + n_test]
    u, L = fit_direction(lm, fit_pairs)                            # (only to keep the split identical)
    verbalized = derive_mt_reminder(lm, fit_pairs)
    print(f"model: {model_id.split('/')[-1]}  n_test={n_test}\nverbalized rho: {verbalized}\n", flush=True)
    REM = {"verbalized": verbalized, "generic": r_generic(None)}
    POS = [("early", 0.0), ("mid", 0.5), ("late", 1.0)]

    offh = [int(is_refusal(rollout_at(lm, h["turns"], None, ""))) for h, b in test_pairs]
    offb = [int(is_refusal(rollout_at(lm, b["turns"], None, ""))) for h, b in test_pairs]
    print(f"{'reminder':12}{'pos':7}{'ASR':>8}{'benign':>9}")
    print(f"{'(off)':12}{'-':7}{1 - np.mean(offh):>8.3f}{1 - np.mean(offb):>9.3f}", flush=True)
    for rname, rtext in REM.items():
        for pname, frac in POS:
            hh, bb = [], []
            for h, b in test_pairs:
                for turns, dst in ((h["turns"], hh), (b["turns"], bb)):
                    fi = min(len(turns) - 1, int(round(frac * (len(turns) - 1))))
                    dst.append(int(is_refusal(rollout_at(lm, turns, fi, rtext))))
            print(f"{rname:12}{pname:7}{1 - np.mean(hh):>8.3f}{1 - np.mean(bb):>9.3f}", flush=True)
    print("\nASR=harmful compliance (lower better).  benign=completion (preserve).  "
          "early=first turn, late=committing turn.  If verbalized ASR falls early->late, timing rescues it.")


if __name__ == "__main__":
    main()
