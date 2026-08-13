#!/usr/bin/env python3
"""Analysis harness for campaign traces. For a given model, encode every
tool-result context from its real traces, fit the RepE transition direction,
and evaluate with AUROC + leave-one-suite-out CV (train-selected layer -> no
test leakage). Reports per-suite cross-domain generalization.

Usage: score_campaign.py <model_id> <traces.jsonl>
"""
import sys, json, numpy as np, mlx.core as mx
from mlx_lm import load
from introspect import Introspector

MODEL, TRACES = sys.argv[1], sys.argv[2]
MARK = "INFORMATION>"

def auroc(scores, labels):
    """Rank-based AUROC; labels in {0,1}."""
    s = np.asarray(scores); y = np.asarray(labels)
    n1 = y.sum(); n0 = len(y) - n1
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
    def rep_all_layers(messages):
        try:
            s = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            s = tok.apply_chat_template([{"role": ("user" if m["role"]=="tool" else m["role"]),
                 "content": m.get("content","")} for m in messages], add_generation_prompt=True, tokenize=False)
        ids = mx.array(tok.encode(s))[None]
        store = {"hidden": {}, "route": {}}
        with ins._capture(store):
            out = ins.model(ids); mx.eval(out, *store["hidden"].values())
        hs = mx.concatenate([store["hidden"][i][None] for i in range(nL+1)], axis=0)[:, 0]
        return np.array(hs[:, -1, :].astype(mx.float32))       # [L+1, H] last-token

    # collect contexts: (suite, task, label, reps[L+1,H])
    data = []
    for line in open(TRACES):
        r = json.loads(line)
        if r["model"] != MODEL: continue
        for i, m in enumerate(r["messages"]):
            if m["role"] != "tool" or not m["text"].strip(): continue
            lab = 1 if MARK in m["text"] else 0
            data.append((r["suite"], r["user_task"], lab, rep_all_layers(build_messages(r["messages"][:i+1]))))
    suites = sorted({d[0] for d in data})
    print(f"model={MODEL}\ncontexts={len(data)} inj={sum(d[2] for d in data)} clean={sum(1-d[2] for d in data)}")
    print("per-suite:", {s: sum(1 for d in data if d[0]==s) for s in suites})

    def fit_dir(idx, L):
        X = np.array([data[i][3][L] for i in idx]); y = np.array([data[i][2] for i in idx])
        d = X[y==1].mean(0) - X[y==0].mean(0); d /= (np.linalg.norm(d)+1e-9)
        thr = 0.5*((X[y==1]@d).mean() + (X[y==0]@d).mean())
        return d, thr

    # Leave-one-suite-out CV; select layer on TRAIN suites only
    print(f"\n{'held-out suite':16}{'layer*':>7}{'AUROC':>8}{'acc':>7}")
    aurocs=[]
    for ho in suites:
        tr = [i for i in range(len(data)) if data[i][0]!=ho]
        te = [i for i in range(len(data)) if data[i][0]==ho]
        if not te or len({data[i][2] for i in te})<2:
            print(f"{ho:16}  (skip: single-class test)"); continue
        # inner split of train to pick layer
        cut=int(len(tr)*0.8); itr,ival=tr[:cut],tr[cut:]
        bestL,bestv=0,-1
        for L in range(nL+1):
            d,thr=fit_dir(itr,L)
            sv=[data[i][3][L]@d for i in ival]; yv=[data[i][2] for i in ival]
            v=auroc(sv,yv)
            if v==v and v>bestv: bestv,bestL=v,L
        d,thr=fit_dir(tr,bestL)
        st=[data[i][3][bestL]@d for i in te]; yt=[data[i][2] for i in te]
        a=auroc(st,yt); acc=(np.mean([s>thr for s,y in zip(st,yt) if y==1])+np.mean([s<thr for s,y in zip(st,yt) if y==0]))/2
        aurocs.append(a)
        print(f"{ho:16}{bestL:7d}{a:8.2f}{acc:7.2f}")
    print(f"\nmean leave-one-suite-out AUROC: {np.nanmean(aurocs):.3f}")

if __name__ == "__main__":
    main()
