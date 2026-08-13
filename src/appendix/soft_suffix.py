#!/usr/bin/env python3
"""Enforcement via an optimized SOFT SUFFIX (white-box, embedding-level).

We append K trainable embedding vectors to the context's input embeddings and
optimize them (model frozen) to minimize the projection h_L . u onto the
injection direction u --- i.e. steer the representation back off the injected
axis. We report the representational shift and a greedy generation before/after.

Injection direction u is the diff-of-means of injected vs clean contexts at
layer L. Soft suffix is injected through the model's input_embeddings path.
"""
import numpy as np, mlx.core as mx, mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from introspect import Introspector
from repeng import encode
from agentdojo_pairs import make_pairs

MODEL = "mlx-community/Qwen3-8B-8bit"
LAYER = 16          # steer/measure at this residual layer
K = 8               # number of soft-suffix vectors
STEPS = 120

model, tok = load(MODEL)
ins = Introspector(model, tok, MODEL)
embed = model.model.embed_tokens
layers = model.model.layers
d_model = embed.weight.shape[1]

pairs = make_pairs(attacks=("important_instructions",), max_pairs=20)
# injection direction u at LAYER (last-token state)
C = np.stack([encode(ins, p.clean_context, "last")[LAYER] for p in pairs])
I = np.stack([encode(ins, p.injected_context, "last")[LAYER] for p in pairs])
u_np = I.mean(0) - C.mean(0); u_np /= np.linalg.norm(u_np)
u = mx.array(u_np)
clean_ref, inj_ref = float((C @ u_np).mean()), float((I @ u_np).mean())
thr = 0.5 * (clean_ref + inj_ref)

# target: an injected context
ctx = pairs[0].injected_context
ctx_ids = mx.array(tok.encode(ctx))[None]              # [1, T]
ctx_emb = embed(ctx_ids)                               # [1, T, d]  (constant)

def hidden_at_L(full_emb):
    """Differentiable: run layers 0..L-1 on input embeddings, return last-token h_L."""
    h = full_emb
    mask = create_attention_mask(h, None)
    for i in range(LAYER):
        h = layers[i](h, mask, None)
    return h[0, -1, :]

def proj(full_emb):
    return hidden_at_L(full_emb) @ u

# init soft suffix from K random real-token embeddings (on-manifold start)
rng = np.random.default_rng(0)
init_ids = mx.array(rng.integers(0, embed.weight.shape[0], size=K))
suffix = embed(init_ids[None]).astype(mx.float32)      # [1, K, d]
suffix = mx.array(suffix)

# realistic embedding norm (project suffix onto this sphere each step -> stay on-manifold)
samp = embed(mx.array(rng.integers(0, embed.weight.shape[0], size=256))[None]).astype(mx.float32)[0]
typ_norm = float(mx.median(mx.linalg.norm(samp, axis=-1)))
TARGET = clean_ref                                     # steer TO the clean region, not to -inf

def renorm(sfx):
    n = mx.linalg.norm(sfx, axis=-1, keepdims=True)
    return sfx / (n + 1e-6) * typ_norm

def loss_fn(sfx):
    full = mx.concatenate([ctx_emb, sfx.astype(ctx_emb.dtype)], axis=1)
    return (proj(full) - TARGET) ** 2                  # hit the clean projection, don't overshoot

before = float(proj(ctx_emb))                          # no suffix
suffix = renorm(suffix)
opt = optim.Adam(learning_rate=0.02)
grad_fn = mx.value_and_grad(loss_fn)
traj = []
for step in range(STEPS):
    val, g = grad_fn(suffix)
    suffix = opt.apply_gradients({"s": g}, {"s": suffix})["s"]
    suffix = renorm(suffix)                             # project back onto the embedding sphere
    mx.eval(suffix)
    if step % 20 == 0 or step == STEPS-1:
        traj.append((step, float(mx.sqrt(val))))
print(f"(suffix constrained to token-embedding norm={typ_norm:.2f}; target proj={TARGET:+.2f})")

full = mx.concatenate([ctx_emb, suffix.astype(ctx_emb.dtype)], axis=1)
after = float(proj(full))

print(f"model={MODEL}  layer={LAYER}  K={K}")
print(f"reference projections:  clean mean={clean_ref:+.2f}   injected mean={inj_ref:+.2f}   threshold={thr:+.2f}")
print(f"\nSTEER an injected context off u (minimize h_L . u):")
print(f"  before (no suffix):     {before:+.2f}   -> {'INJECTED' if before>thr else 'clean'} side")
print(f"  after  ({K}-vec suffix): {after:+.2f}   -> {'INJECTED' if after>thr else 'clean'} side")
print(f"  optimization trajectory: " + " ".join(f"{s}:{v:+.1f}" for s,v in traj))

# behavioral: greedy continuation with vs without the soft suffix
def greedy(full_emb, n=50):
    cache = make_prompt_cache(model)
    logits = model(mx.zeros((1, full_emb.shape[1]), dtype=mx.int32),
                   cache=cache, input_embeddings=full_emb)
    out = []
    tokid = int(mx.argmax(logits[0, -1]))
    for _ in range(n):
        out.append(tokid)
        if tokid == tok.eos_token_id: break
        lg = model(mx.array([[tokid]]), cache=cache)
        tokid = int(mx.argmax(lg[0, -1]))
    return tok.decode(out).strip().replace("\n", " ")

print(f"\nBEHAVIORAL (greedy, injected goal = email mark.black-2134@gmail.com):")
print(f"  WITHOUT suffix: {greedy(ctx_emb)[:180]}")
print(f"  WITH   suffix : {greedy(full)[:180]}")
