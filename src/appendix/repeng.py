#!/usr/bin/env python3
"""Representation extraction + concept-direction reading (the 'reading' half of
Representation Engineering, a la Zou et al. 2023).

The LLM is used purely as an encoder: one forward pass per prompt, grab the
residual stream, NO generation / no lm_head sampling.

  encode(ins, text, pool)       -> [n_layers+1, hidden]  per-layer representation
  concept_direction(pos, neg)   -> [n_layers+1, hidden]  unit direction per layer
                                    (difference-of-means of the two stimulus sets)

Demo (run as script): learns a 'positive vs negative sentiment' direction from a
handful of contrast sentences, then shows it linearly separates HELD-OUT sentences
— i.e. the model carries a linear sentiment representation, and we found it.
"""
from __future__ import annotations
import numpy as np
import mlx.core as mx
from introspect import Introspector


def encode(ins: Introspector, text: str, pool: str = "last",
           chat_template: bool = False) -> np.ndarray:
    """Forward-only. Returns per-layer representation [n_layers+1, hidden]."""
    if chat_template:
        ids = ins.tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True)
    else:
        ids = ins.tok.encode(text)
    ids = mx.array(ids)[None]
    store = {"hidden": {}, "route": {}}
    with ins._capture(store):
        out = ins.model(ids)                       # one forward pass; fills store
        mx.eval(out, *store["hidden"].values())
    hs = mx.concatenate([store["hidden"][i][None]
                         for i in range(ins.n_layers + 1)], axis=0)[:, 0]  # [L+1,T,H]
    if pool == "last":
        rep = hs[:, -1, :]                         # last-token rep per layer
    elif pool == "mean":
        rep = hs.mean(axis=1)
    else:
        raise ValueError(pool)
    return np.array(rep.astype(mx.float32))        # [L+1, H]


def concept_direction(ins, pos, neg, pool="last"):
    """Diff-of-means concept direction per layer (unit norm)."""
    P = np.stack([encode(ins, t, pool) for t in pos])   # [nP, L+1, H]
    N = np.stack([encode(ins, t, pool) for t in neg])   # [nN, L+1, H]
    d = P.mean(0) - N.mean(0)                            # [L+1, H]
    d /= (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)
    return d, P, N


# ---------------------------------------------------------------- demo --------
if __name__ == "__main__":
    from mlx_lm import load

    POS_TRAIN = [
        "I absolutely loved this, it was wonderful.",
        "What a fantastic and delightful experience!",
        "This made me so happy and grateful.",
        "An amazing, brilliant, joyful day.",
        "I feel great — everything is going perfectly.",
        "The kindest, most beautiful gift ever.",
        "Pure bliss, I couldn't be more pleased.",
        "Such a triumphant and uplifting success.",
    ]
    NEG_TRAIN = [
        "I hated this, it was awful.",
        "What a terrible and miserable experience.",
        "This made me so sad and hopeless.",
        "A dreadful, bleak, depressing day.",
        "I feel horrible — everything is falling apart.",
        "The cruelest, most disgusting insult ever.",
        "Pure agony, I couldn't be more upset.",
        "Such a crushing and humiliating failure.",
    ]
    POS_TEST = [
        "This is the best news I've heard all year.",
        "I'm thrilled and overjoyed right now.",
        "A gorgeous, heart-warming little story.",
        "Everything about today felt lucky and bright.",
    ]
    NEG_TEST = [
        "This is the worst news I've heard all year.",
        "I'm devastated and heartbroken right now.",
        "A grim, gut-wrenching little story.",
        "Everything about today felt cursed and dark.",
    ]

    name = "mlx-community/Qwen3-8B-8bit"
    print(f"Loading {name} (encoder-only use)...")
    model, tok = load(name)
    ins = Introspector(model, tok, name)

    d, P, N = concept_direction(ins, POS_TRAIN, NEG_TRAIN, pool="last")
    n_layers = ins.n_layers

    # project held-out reps onto each layer's direction; threshold at train midpoint
    test_pos = np.stack([encode(ins, t, "last") for t in POS_TEST])  # [n,L+1,H]
    test_neg = np.stack([encode(ins, t, "last") for t in NEG_TEST])

    print(f"\n{'layer':>5} {'train-sep':>10} {'test-acc':>9}")
    best = (-1, 0.0)
    for L in range(n_layers + 1):
        dl = d[L]
        # midpoint threshold from training projections
        thr = 0.5 * (P[:, L] @ dl).mean() + 0.5 * (N[:, L] @ dl).mean()
        pp = test_pos[:, L] @ dl
        nn = test_neg[:, L] @ dl
        acc = (np.mean(pp > thr) + np.mean(nn < thr)) / 2
        sep = (P[:, L] @ dl).mean() - (N[:, L] @ dl).mean()   # train margin
        if L % 4 == 0 or acc == 1.0:
            print(f"{L:5d} {sep:10.2f} {acc:9.2f}")
        if acc > best[1]:
            best = (L, acc)
    print(f"\nBest layer for linear sentiment readout: layer {best[0]} "
          f"(held-out acc {best[1]:.2f}) out of {n_layers} layers")
    np.savez_compressed("sentiment_direction.npz", direction=d)
    print("Saved per-layer concept directions -> sentiment_direction.npz  "
          f"shape {d.shape}")
