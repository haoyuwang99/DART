#!/usr/bin/env python3
"""Fit + evaluate the injection monitor on REAL 30B agent traces.

- reconstruct each tool-result context (real chat template)
- get per-layer reps with last-token AND mean pooling
- run-level split (test contexts come from unseen user tasks -> no leakage)
- per layer/pooling: fit u = mean(inj)-mean(clean) on train, eval on test
"""
import json, numpy as np, mlx.core as mx
from mlx_lm import load
from introspect import Introspector

MODEL = "mlx-community/Qwen3-30B-A3B-8bit"
MARK = "INFORMATION>"

def build_messages(msgs_upto):
    out = []
    for m in msgs_upto:
        role, text, tcs = m["role"], m["text"], m.get("tool_calls", [])
        if role == "assistant" and tcs:
            out.append({"role": "assistant", "content": text,
                        "tool_calls": [{"type": "function", "function": {"name": c["fn"],
                                        "arguments": json.dumps(c["args"])}} for c in tcs]})
        elif role == "tool":
            out.append({"role": "tool", "content": text or "(empty)"})
        else:
            out.append({"role": role, "content": text or ""})
    return out

def main():
    model, tok = load(MODEL)
    ins = Introspector(model, tok, MODEL)
    nL = ins.n_layers

    def pooled(messages):
        try:
            s = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            s = tok.apply_chat_template([{"role": ("user" if m["role"]=="tool" else m["role"]),
                 "content": m.get("content","")} for m in messages],
                 add_generation_prompt=True, tokenize=False)
        ids = mx.array(tok.encode(s))[None]
        store = {"hidden": {}, "route": {}}
        with ins._capture(store):
            out = ins.model(ids); mx.eval(out, *store["hidden"].values())
        hs = mx.concatenate([store["hidden"][i][None] for i in range(nL+1)], axis=0)[:, 0]  # [L+1,T,H]
        last = np.array(hs[:, -1, :].astype(mx.float32))
        mean = np.array(hs.mean(axis=1).astype(mx.float32))
        return last, mean

    rows = []   # (task, label, last[L+1,H], mean[L+1,H])
    for line in open("traces_30b.jsonl"):
        rec = json.loads(line)
        for i, m in enumerate(rec["messages"]):
            if m["role"] != "tool" or not m["text"].strip():
                continue
            label = 1 if MARK in m["text"] else 0
            last, mean = pooled(build_messages(rec["messages"][:i+1]))
            rows.append((rec["user_task"], label, last, mean))
    print(f"scored {len(rows)} tool-result contexts "
          f"({sum(r[1] for r in rows)} injected, {sum(1-r[1] for r in rows)} clean)")

    # run-level split by task id
    tasks = sorted({r[0] for r in rows})
    k = int(len(tasks) * 0.6)
    train_tasks = set(tasks[:k])
    tr = [r for r in rows if r[0] in train_tasks]
    te = [r for r in rows if r[0] not in train_tasks]
    print(f"train contexts {len(tr)} / test contexts {len(te)} (split by task)")

    for pool_idx, pool_name in [(2, "last-token"), (3, "mean-pool")]:
        print(f"\n=== pooling: {pool_name} ===\n{'layer':>5} {'train':>7} {'TEST':>7}")
        best = (0, 0.0)
        for L in range(nL + 1):
            Xtr = np.array([r[pool_idx][L] for r in tr]); ytr = np.array([r[1] for r in tr])
            Xte = np.array([r[pool_idx][L] for r in te]); yte = np.array([r[1] for r in te])
            d = Xtr[ytr==1].mean(0) - Xtr[ytr==0].mean(0); d /= (np.linalg.norm(d)+1e-9)
            thr = 0.5*((Xtr[ytr==1]@d).mean() + (Xtr[ytr==0]@d).mean())
            tra = (np.mean(Xtr[ytr==1]@d > thr) + np.mean(Xtr[ytr==0]@d < thr))/2
            tea = (np.mean(Xte[yte==1]@d > thr) + np.mean(Xte[yte==0]@d < thr))/2
            if tea > best[1]: best = (L, tea)
            if L % 4 == 0: print(f"{L:5d} {tra:7.2f} {tea:7.2f}")
        print(f"BEST test: layer {best[0]} acc {best[1]:.2f}")

if __name__ == "__main__":
    main()
