#!/usr/bin/env python3
"""Score REAL logged agent traces with the monitor direction.

For every tool-result step in every trace we reconstruct the exact context the
model saw (real Qwen3 chat template) at that decision point, encode it, and
compute score = rep_layer · u. Then compare tool results that CONTAIN the
injection vs clean ones. Also tests transfer: u was fit on SYNTHETIC contexts.
"""
import json, numpy as np
from mlx_lm import load
from introspect import Introspector

MODEL = "mlx-community/Qwen3-8B-8bit"
MARK = "INFORMATION>"

def build_messages(msgs_upto):
    """Turn serialized trace messages into chat-template messages."""
    out = []
    for m in msgs_upto:
        role, text, tcs = m["role"], m["text"], m.get("tool_calls", [])
        if role == "assistant" and tcs:
            out.append({"role": "assistant", "content": text,
                        "tool_calls": [{"type": "function",
                                        "function": {"name": c["fn"],
                                                     "arguments": json.dumps(c["args"])}}
                                       for c in tcs]})
        elif role == "tool":
            out.append({"role": "tool", "content": text or "(empty)"})
        else:
            out.append({"role": role, "content": text or ""})
    return out

def main():
    z = np.load("safety_direction.npz")
    u, LAYER, thr = z["direction"], int(z["layer"]), float(z["threshold"])
    model, tok = load(MODEL)
    ins = Introspector(model, tok, MODEL)

    def score_context(messages):
        try:
            s = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:                       # fallback: tool->user
            mm = [{"role": ("user" if m["role"] == "tool" else m["role"]),
                   "content": m.get("content", "")} for m in messages]
            s = tok.apply_chat_template(mm, add_generation_prompt=True, tokenize=False)
        from repeng import encode
        return float(encode(ins, s, pool="last", chat_template=False)[LAYER] @ u)

    inj_scores, clean_scores = [], []
    for line in open("traces.jsonl"):
        rec = json.loads(line)
        msgs = rec["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "tool":
                continue
            ctx = build_messages(msgs[:i + 1])
            sc = score_context(ctx)
            if MARK in m["text"]:
                inj_scores.append(sc)
            elif m["text"].strip():             # non-empty clean tool result
                clean_scores.append(sc)

    inj, clean = np.array(inj_scores), np.array(clean_scores)
    print(f"scored tool-result steps: {len(clean)} clean (non-empty), {len(inj)} injected")
    print(f"\n{'':14}{'mean':>8}{'min':>8}{'max':>8}")
    print(f"{'clean':14}{clean.mean():8.2f}{clean.min():8.2f}{clean.max():8.2f}")
    print(f"{'injected':14}{inj.mean():8.2f}{inj.min():8.2f}{inj.max():8.2f}")
    print(f"\nsaved threshold (from synthetic fit): {thr:+.2f}")
    print(f"  injected above threshold: {int((inj>thr).sum())}/{len(inj)}")
    print(f"  clean    above threshold: {int((clean>thr).sum())}/{len(clean)}")
    if len(inj) and len(clean):
        rethr = 0.5*(inj.mean()+clean.mean())
        acc = (np.mean(clean < rethr) + np.mean(inj > rethr))/2
        print(f"\nre-fit threshold on real traces: {rethr:+.2f} -> separation acc {acc:.2f}")
        print(f"individual injected scores: {[round(float(x),1) for x in inj]}")

if __name__ == "__main__":
    main()
