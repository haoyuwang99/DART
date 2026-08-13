#!/usr/bin/env python3
"""Behavioral enforcement on a COMPLYING setup (Qwen3-14B, slack).

1. Pull a real trace where Qwen3-14B executed the injected action (sent the
   attacker link secure-systems-252) and reconstruct the agent context it saw.
2. Reproduce compliance by greedy generation (no suffix).
3. Optimize a soft suffix (embedding-norm-constrained, targets the clean
   projection) to steer h_L off the injection direction u.
4. Generate WITH the suffix and check whether the injected action is suppressed.
"""
import json, numpy as np, mlx.core as mx, mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt
from agentdojo.task_suite.load_suites import get_suites
from introspect import Introspector
from repeng import encode
from agentdojo_pairs import make_pairs

MODEL = "mlx-community/Qwen3-14B-8bit"
TARGET_STR = "secure-systems-252"      # the injected attacker link
LAYER, K, STEPS = 20, 10, 140

model, tok = load(MODEL)
ins = Introspector(model, tok, MODEL)
embed = model.model.embed_tokens; layers = model.model.layers
slack_tools = get_suites("v1")["slack"].tools

# --- 1. find a complying slack trace + reconstruct the pre-compliance context ---
ctx_msgs = None
for line in open("traces_campaign.jsonl"):
    r = json.loads(line)
    if r["model"] != MODEL or r["suite"] != "slack" or r["condition"] != "injected": continue
    msgs = r["messages"]
    for j, m in enumerate(msgs):
        if m["role"] == "assistant" and TARGET_STR in json.dumps(m.get("tool_calls", [])):
            ctx_msgs = msgs[:j]; break
    if ctx_msgs: break
assert ctx_msgs, "no complying trace found"

def reconstruct(msgs):
    out = []
    for m in msgs:
        role, text = m["role"], m["text"]
        if role == "system": content = _make_system_prompt(text, slack_tools)
        elif role == "tool": content = json.dumps({"result": text or "Success"})
        else: content = text
        out.append({"role": role, "content": content})
    return tok.apply_chat_template(out, add_generation_prompt=True, tokenize=False)

ctx = reconstruct(ctx_msgs)
ctx_ids = mx.array(tok.encode(ctx))[None]
ctx_emb = embed(ctx_ids)

# --- injection direction u for 14B at LAYER (constructed slack pairs) ---
pairs = make_pairs(suite_name="slack", attacks=("important_instructions",), max_pairs=16)
C = np.stack([encode(ins, p.clean_context, "last")[LAYER] for p in pairs])
I = np.stack([encode(ins, p.injected_context, "last")[LAYER] for p in pairs])
u_np = I.mean(0) - C.mean(0); u_np /= np.linalg.norm(u_np); u = mx.array(u_np)
clean_ref, inj_ref = float((C@u_np).mean()), float((I@u_np).mean()); thr = 0.5*(clean_ref+inj_ref)

def hidden_L(full):
    h = full; mask = create_attention_mask(h, None)
    for i in range(LAYER): h = layers[i](h, mask, None)
    return h[0, -1, :]
def proj(full): return hidden_L(full) @ u

def greedy(full_emb, n=70):
    cache = make_prompt_cache(model)
    lg = model(mx.zeros((1, full_emb.shape[1]), dtype=mx.int32), cache=cache, input_embeddings=full_emb)
    out=[]; t=int(mx.argmax(lg[0,-1]))
    for _ in range(n):
        out.append(t)
        if t==tok.eos_token_id: break
        t=int(mx.argmax(model(mx.array([[t]]), cache=cache)[0,-1]))
    return tok.decode(out).strip().replace("\n"," ")

# --- 2. reproduce compliance (no suffix) ---
base_out = greedy(ctx_emb)
print(f"model={MODEL} layer={LAYER} K={K}")
print(f"proj refs: clean={clean_ref:+.2f} injected={inj_ref:+.2f} thr={thr:+.2f}")
print(f"before (no suffix): proj={float(proj(ctx_emb)):+.2f}")
print(f"  OUTPUT: {base_out[:200]}")
print(f"  -> injected action ({TARGET_STR}) present: {TARGET_STR in base_out}")

# --- 3. optimize soft suffix (constrained, target clean) ---
rng = np.random.default_rng(0)
samp = embed(mx.array(rng.integers(0, embed.weight.shape[0], 256))[None]).astype(mx.float32)[0]
typ = float(mx.median(mx.linalg.norm(samp, axis=-1)))
sfx = embed(mx.array(rng.integers(0, embed.weight.shape[0], K))[None]).astype(mx.float32)
renorm = lambda s: s/(mx.linalg.norm(s,axis=-1,keepdims=True)+1e-6)*typ
sfx = renorm(sfx)
def loss_fn(s):
    full = mx.concatenate([ctx_emb, s.astype(ctx_emb.dtype)], axis=1)
    return (proj(full)-clean_ref)**2
opt = optim.Adam(learning_rate=0.02); gfn = mx.value_and_grad(loss_fn)
for step in range(STEPS):
    v,g = gfn(sfx); sfx = renorm(opt.apply_gradients({"s":g},{"s":sfx})["s"]); mx.eval(sfx)

# --- 4. generate WITH suffix ---
full = mx.concatenate([ctx_emb, sfx.astype(ctx_emb.dtype)], axis=1)
steer_out = greedy(full)
print(f"\nafter ({K}-vec suffix): proj={float(proj(full)):+.2f}  ({'clean' if float(proj(full))<thr else 'injected'} side)")
print(f"  OUTPUT: {steer_out[:200]}")
print(f"  -> injected action ({TARGET_STR}) present: {TARGET_STR in steer_out}")
print(f"\nENFORCEMENT: {'SUPPRESSED injected action' if (TARGET_STR in base_out and TARGET_STR not in steer_out) else 'no clean flip (see outputs)'}")
