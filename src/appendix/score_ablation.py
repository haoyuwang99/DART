#!/usr/bin/env python3
"""State vs transition ablation. For each tool-result context we compute BOTH
  state:      h_i          (last-token hidden of the prefix incl. the tool result)
  transition: h_i - h_{i-1}(what the tool result ADDED; prev prefix ends at the
                            assistant tool-call, before the result arrived)
and evaluate each with the identical leave-one-suite-out CV (layers 1..N, layer
selected on a train inner split). Reports mean LOSO AUROC for both forms.

Usage: score_ablation.py <model_id> <traces.jsonl>
"""
import sys, json, numpy as np, mlx.core as mx
from mlx_lm import load
from introspect import Introspector

MODEL, TRACES = sys.argv[1], sys.argv[2]
MARK = "INFORMATION>"

def auroc(s, y):
    s = np.asarray(s); y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    return (ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)

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
    model, tok = load(MODEL); ins = Introspector(model, tok, MODEL); nL = ins.n_layers
    def hidden(messages):
        try:
            s = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        except Exception:
            s = tok.apply_chat_template([{"role": ("user" if m["role"]=="tool" else m["role"]),
                 "content": m.get("content","")} for m in messages], add_generation_prompt=False, tokenize=False)
        ids = mx.array(tok.encode(s))[None]
        store = {"hidden": {}, "route": {}}
        with ins._capture(store):
            out = ins.model(ids); mx.eval(out, *store["hidden"].values())
        hs = mx.concatenate([store["hidden"][i][None] for i in range(nL+1)], axis=0)[:, 0]
        return np.array(hs[:, -1, :].astype(mx.float32))

    data = []  # (suite, label, h_i, delta)
    for line in open(TRACES):
        r = json.loads(line)
        if r["model"] != MODEL: continue
        msgs = r["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "tool" or not m["text"].strip() or i == 0: continue
            lab = 1 if MARK in m["text"] else 0
            h_i = hidden(build_messages(msgs[:i+1]))
            h_prev = hidden(build_messages(msgs[:i]))
            data.append((r["suite"], lab, h_i, h_i - h_prev))
    suites = sorted({d[0] for d in data})
    print(f"model={MODEL}  contexts={len(data)} inj={sum(d[1] for d in data)} clean={sum(1-d[1] for d in data)}")

    def loso(repidx):
        aur = []
        for ho in suites:
            tr = [i for i in range(len(data)) if data[i][0] != ho]
            te = [i for i in range(len(data)) if data[i][0] == ho]
            if len({data[i][1] for i in te}) < 2: continue
            cut = int(len(tr)*0.8); itr, iv = tr[:cut], tr[cut:]
            bestL, bv = 1, -1
            for L in range(1, nL+1):                       # skip layer 0 (embeddings)
                y = np.array([data[i][1] for i in itr])
                if len(set(y)) < 2 or len({data[i][1] for i in iv}) < 2: continue
                X = np.array([data[i][repidx][L] for i in itr])
                d = X[y==1].mean(0) - X[y==0].mean(0); d /= np.linalg.norm(d)+1e-9
                v = auroc([data[i][repidx][L]@d for i in iv], [data[i][1] for i in iv])
                if v == v and v > bv: bv, bestL = v, L
            y = np.array([data[i][1] for i in tr]); X = np.array([data[i][repidx][bestL] for i in tr])
            d = X[y==1].mean(0) - X[y==0].mean(0); d /= np.linalg.norm(d)+1e-9
            aur.append(auroc([data[i][repidx][bestL]@d for i in te], [data[i][1] for i in te]))
        return np.nanmean(aur)

    print(f"  STATE      (h_i)         mean LOSO AUROC = {loso(2):.3f}")
    print(f"  TRANSITION (h_i - h_i-1) mean LOSO AUROC = {loso(3):.3f}")

if __name__ == "__main__":
    main()
