#!/usr/bin/env python3
"""Activation steering — the CONTROL half of representation engineering.

We read a concept direction from contrast pairs (as in repeng.py), then ADD it
back into the residual stream during generation to steer the model's behaviour.
Same direction, now used as a control knob instead of a readout.

Mechanism: temporarily wrap the target decoder layers' __call__ so their output
gets `coef * direction` added at every position, then generate normally.
"""
from __future__ import annotations
import contextlib
import numpy as np
import mlx.core as mx
from mlx_lm import load, generate
from introspect import Introspector
from repeng import encode


def raw_direction(ins, pos, neg, pool="last"):
    """Un-normalised diff-of-means per layer [n_layers+1, H] (keeps natural scale
    so a single coefficient steers sensibly)."""
    P = np.stack([encode(ins, t, pool) for t in pos])   # [n, L+1, H]
    N = np.stack([encode(ins, t, pool) for t in neg])
    return P.mean(0) - N.mean(0)                         # raw magnitude retained


@contextlib.contextmanager
def steering(ins, layer_outputs, vecs, coef):
    """Add coef*vec to the output of each residual-stream position `L`
    (L indexes hidden states: hidden[L] = output of ins.layers[L-1])."""
    layer_cls = type(ins.layers[0])
    orig = layer_cls.__call__
    # map layer-instance index -> steering vector (mx array)
    steer_map = {}
    for L, v in zip(layer_outputs, vecs):
        steer_map[L - 1] = mx.array(v)            # ins.layers[L-1] produces hidden[L]
    for i, lyr in enumerate(ins.layers):
        lyr._steer_idx = i

    def patched(self_layer, x, *a, **k):
        out = orig(self_layer, x, *a, **k)
        v = steer_map.get(self_layer._steer_idx)
        if v is not None:
            out = out + coef * v.astype(out.dtype)
        return out

    layer_cls.__call__ = patched
    try:
        yield
    finally:
        layer_cls.__call__ = orig


def gen(ins, prompt, max_tokens=60):
    msgs = [{"role": "user", "content": prompt}]
    text = ins.tok.apply_chat_template(msgs, add_generation_prompt=True)
    return generate(ins.model, ins.tok, prompt=text, max_tokens=max_tokens,
                    verbose=False).strip().replace("\n", " ")


if __name__ == "__main__":
    POS = ["I absolutely loved this, it was wonderful.",
           "What a fantastic and delightful experience!",
           "This made me so happy and grateful.",
           "An amazing, brilliant, joyful day.",
           "I feel great, everything is perfect.",
           "The kindest, most beautiful gift ever."]
    NEG = ["I hated this, it was awful.",
           "What a terrible and miserable experience.",
           "This made me so sad and hopeless.",
           "A dreadful, bleak, depressing day.",
           "I feel horrible, everything is falling apart.",
           "The cruelest, most disgusting insult ever."]

    name = "mlx-community/Qwen3-8B-8bit"
    print(f"loading {name} ...")
    model, tok = load(name)
    ins = Introspector(model, tok, name)

    d = raw_direction(ins, POS, NEG)                       # [L+1, H]
    BAND = [14, 15, 16]                                    # small mid-layer band
    vecs = [d[L] for L in BAND]
    print(f"steering layers {BAND}  (concept: sentiment)\n")

    PROMPT = "Tell me about your day at the office in one short paragraph. /no_think"
    for coef in (-1.5, -0.8, 0.0, 0.8, 1.5):
        with steering(ins, BAND, vecs, coef):
            out = gen(ins, PROMPT, max_tokens=70)
        tag = "NEUTRAL (no steering)" if coef == 0 else (
              f"coef {coef:+.1f}  (toward {'POSITIVE' if coef>0 else 'NEGATIVE'})")
        print(f"[{tag}]\n  {out}\n")
