#!/usr/bin/env python3
"""Read an 'injection-present' direction from a model's activations on AgentDojo
contexts, and test whether it detects injections — including HELD-OUT attack types.

Pipeline: build matched (clean,injected) pairs -> encode each context (forward-only)
-> per-layer diff-of-means direction from TRAIN attack -> evaluate detection on
held-out tasks (in-distribution) and a DIFFERENT attack style (generalization).
"""
import numpy as np
from mlx_lm import load
from introspect import Introspector
from repeng import encode
from agentdojo_pairs import make_pairs

MODEL = "mlx-community/Qwen3-8B-8bit"
TRAIN_ATTACK = "important_instructions"
TEST_ATTACK  = "ignore_previous"          # held-out attack style


def encode_pairs(ins, pairs):
    clean = np.stack([encode(ins, p.clean_context, pool="last") for p in pairs])     # [n,L+1,H]
    inj   = np.stack([encode(ins, p.injected_context, pool="last") for p in pairs])
    return clean, inj


def main():
    print("building pairs...")
    train_pairs = make_pairs(attacks=(TRAIN_ATTACK,), max_pairs=40)
    test_pairs  = make_pairs(attacks=(TEST_ATTACK,),  max_pairs=40)
    print(f"train ({TRAIN_ATTACK}): {len(train_pairs)} pairs | "
          f"test ({TEST_ATTACK}): {len(test_pairs)} pairs")

    print(f"loading {MODEL} as encoder...")
    model, tok = load(MODEL)
    ins = Introspector(model, tok, MODEL)
    nL = ins.n_layers

    # split train attack into fit / in-distribution-test by task index
    ntr = len(train_pairs); cut = int(ntr * 0.7)
    fit, indist = train_pairs[:cut], train_pairs[cut:]

    print("encoding (forward-only)...")
    fit_clean, fit_inj   = encode_pairs(ins, fit)
    ind_clean, ind_inj   = encode_pairs(ins, indist)
    xat_clean, xat_inj   = encode_pairs(ins, test_pairs)

    def eval_layer(L, dirn, thr, clean, inj):
        pc = clean[:, L] @ dirn        # clean should score LOW, injected HIGH
        pi = inj[:, L] @ dirn
        acc = (np.mean(pc < thr) + np.mean(pi > thr)) / 2
        return acc

    print(f"\n{'layer':>5} {'fit-sep':>9} {'in-dist':>8} {'x-attack':>9}")
    best = {"indist": (0, 0), "xattack": (0, 0)}
    rows = []
    for L in range(nL + 1):
        d = fit_inj[:, L].mean(0) - fit_clean[:, L].mean(0)
        d /= (np.linalg.norm(d) + 1e-8)
        thr = 0.5 * (fit_inj[:, L] @ d).mean() + 0.5 * (fit_clean[:, L] @ d).mean()
        sep = (fit_inj[:, L] @ d).mean() - (fit_clean[:, L] @ d).mean()
        a_ind = eval_layer(L, d, thr, ind_clean, ind_inj)
        a_xat = eval_layer(L, d, thr, xat_clean, xat_inj)
        rows.append((L, a_ind, a_xat, d))
        if a_ind > best["indist"][1]: best["indist"] = (L, a_ind)
        if a_xat > best["xattack"][1]: best["xattack"] = (L, a_xat)
        if L % 3 == 0:
            print(f"{L:5d} {sep:9.1f} {a_ind:8.2f} {a_xat:9.2f}")

    print(f"\nBest IN-DISTRIBUTION : layer {best['indist'][0]} acc {best['indist'][1]:.2f}")
    print(f"Best CROSS-ATTACK    : layer {best['xattack'][0]} acc {best['xattack'][1]:.2f}  "
          f"(trained on {TRAIN_ATTACK}, tested on {TEST_ATTACK})")
    # save the best in-dist direction
    bestL = best["indist"][0]
    d = [r[3] for r in rows if r[0] == bestL][0]
    np.savez_compressed("injection_direction.npz", direction=d, layer=bestL)
    print(f"saved injection_direction.npz (layer {bestL}, dim {d.shape[0]})")


if __name__ == "__main__":
    main()
