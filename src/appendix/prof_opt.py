#!/usr/bin/env python3
"""Profile opt() cost and validate a KV-cached opt (prefill ctx once, backprop only
through the appended suffix). Decides whether caching is worth wiring into the campaign.
"""
import time, json, numpy as np, mlx.core as mx, mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt
from agentdojo.task_suite.load_suites import get_suites
from introspect import Introspector

MODEL = "mlx-community/Qwen3-14B-8bit"; LAYER, K = 20, 10
model, tok = load(MODEL)
embed = model.model.embed_tokens; layers = model.model.layers
tools = get_suites("v1")["workspace"].tools

def recon(msgs):
    out = []
    for m in msgs:
        r, t = m["role"], m["text"]
        c = _make_system_prompt(t, tools) if r == "system" else (json.dumps({"result": t or "Success"}) if r == "tool" else t)
        out.append({"role": r, "content": c})
    ids = tok.encode(tok.apply_chat_template(out, add_generation_prompt=True, tokenize=False))
    return embed(mx.array(ids)[None]), len(ids)

# one complying workspace ctx
ctx = None
for line in open("traces_campaign.jsonl"):
    r = json.loads(line)
    if r["model"] != MODEL or r["suite"] != "workspace" or r["condition"] != "injected": continue
    for j, m in enumerate(r["messages"]):
        if m["role"] == "assistant" and "mark.black-2134" in json.dumps(m):
            ctx, L = recon(r["messages"][:j]); break
    if ctx is not None: break
print(f"ctx tokens L = {L}")

def hL(f):
    h = f; mask = create_attention_mask(h, None)
    for i in range(LAYER): h = layers[i](h, mask, None)
    return h[0, -1, :]

H = int(hL(ctx).shape[0])
u = mx.array((lambda v: v/np.linalg.norm(v))(np.random.default_rng(0).standard_normal(H)).astype(np.float32))
samp = embed(mx.array(np.random.default_rng(1).integers(0, embed.weight.shape[0], 256))[None]).astype(mx.float32)[0]
typ = float(mx.median(mx.linalg.norm(samp, axis=-1)))
rn = lambda x: x/(mx.linalg.norm(x, axis=-1, keepdims=True)+1e-6)*typ
TARGET = -20.0

def opt_plain(steps):
    s = rn(embed(mx.array(np.random.default_rng(0).integers(0, embed.weight.shape[0], K))[None]).astype(mx.float32))
    def loss(x):
        f = mx.concatenate([ctx, x.astype(ctx.dtype)], axis=1); return (hL(f) @ u - TARGET) ** 2
    op = optim.Adam(learning_rate=0.02); gf = mx.value_and_grad(loss)
    for _ in range(steps): _, g = gf(s); s = rn(op.apply_gradients({"s": g}, {"s": s})["s"]); mx.eval(s)
    f = mx.concatenate([ctx, s.astype(ctx.dtype)], axis=1); return float(hL(f) @ u)

def opt_cached(steps):
    # prefill ctx through LAYER layers into a cache (constants)
    cache = make_prompt_cache(model)[:LAYER]
    h = ctx; mask = create_attention_mask(h, None)
    for i in range(LAYER): h = layers[i](h, mask, cache[i])
    L0 = cache[0].offset
    s = rn(embed(mx.array(np.random.default_rng(0).integers(0, embed.weight.shape[0], K))[None]).astype(mx.float32))
    def suf_hidden(x):
        for i in range(LAYER): cache[i].offset = L0           # discard any prior suffix append
        hh = x.astype(ctx.dtype); m = create_attention_mask(hh, cache)
        for i in range(LAYER): hh = layers[i](hh, m, cache[i])
        return hh[0, -1, :]
    def loss(x): return (suf_hidden(x) @ u - TARGET) ** 2
    op = optim.Adam(learning_rate=0.02); gf = mx.value_and_grad(loss)
    for _ in range(steps): _, g = gf(s); s = rn(op.apply_gradients({"s": g}, {"s": s})["s"]); mx.eval(s)
    for i in range(LAYER): cache[i].offset = L0
    return float(suf_hidden(s) @ u)

for name, fn, steps in [("cached-15", opt_cached, 15), ("cached-50", opt_cached, 50), ("plain-15", opt_plain, 15), ("plain-50", opt_plain, 50)]:
    t0 = time.time(); proj = fn(steps); dt = time.time() - t0
    print(f"{name:10} time={dt:6.1f}s  final_proj={proj:8.2f}", flush=True)
