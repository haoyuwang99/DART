#!/usr/bin/env python3
"""R-Judge adapter + RepE monitor eval (non-injection, labeled trajectories).

R-Judge provides multi-turn agent trajectories across 5 domains with a
trajectory-level safety label (0 safe / 1 unsafe). We re-encode each trajectory
with our model, form transition representations, fit the RepE direction with
weak (trajectory) labels, and evaluate trajectory-level AUROC with
leave-one-domain-out CV. This tests the monitor OUTSIDE prompt injection.

Usage:
  rjudge_eval.py --download        # cache the data (non-GPU)
  rjudge_eval.py <model_id>        # run the eval (needs GPU)
"""
import os, sys, json, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "rjudge_data")
DOMAINS = ["Application", "Finance", "IoT", "Program", "Web"]

def gh(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=30).read()

def download():
    os.makedirs(DATA, exist_ok=True)
    for dom in DOMAINS:
        listing = json.loads(gh(f"https://api.github.com/repos/Lordog/R-Judge/contents/data/{dom}"))
        for e in listing:
            if not e["name"].endswith(".json"):
                continue
            dst = os.path.join(DATA, f"{dom}__{e['name']}")
            if os.path.exists(dst):
                continue
            open(dst, "wb").write(gh(e["download_url"]))
            print(f"  saved {dom}/{e['name']}")
    print("R-Judge download complete ->", DATA)

def trajectories():
    """Yield (domain, messages, label). Map user->user, agent->assistant,
    environment->user('[Environment]...'); profile->system."""
    for fn in sorted(os.listdir(DATA)):
        dom = fn.split("__")[0]
        for rec in json.load(open(os.path.join(DATA, fn))):
            msgs = [{"role": "system", "content": rec.get("profile", "") or ""}]
            for rnd in rec["contents"]:
                for m in (rnd if isinstance(rnd, list) else [rnd]):
                    if not isinstance(m, dict): continue
                    c = m.get("content");
                    if not c or c == "None": continue
                    role = m.get("role")
                    if role == "agent": msgs.append({"role": "assistant", "content": c})
                    elif role == "environment": msgs.append({"role": "user", "content": f"[Environment]\n{c}"})
                    else: msgs.append({"role": "user", "content": c})
            if len(msgs) >= 3:
                yield dom, msgs, int(rec["label"])

def auroc(s, y):
    s = np.asarray(s); y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    return (ranks[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)

def main(model_id):
    import mlx.core as mx
    from mlx_lm import load
    from introspect import Introspector
    model, tok = load(model_id); ins = Introspector(model, tok, model_id); nL = ins.n_layers

    def traj_transitions(messages):
        """Return per-layer max-drift summary [L+1,H] over the trajectory."""
        prev = None; maxd = None
        for k in range(2, len(messages)+1):
            s = tok.apply_chat_template(messages[:k], add_generation_prompt=False, tokenize=False)
            ids = mx.array(tok.encode(s))[None]
            store = {"hidden": {}, "route": {}}
            with ins._capture(store):
                out = ins.model(ids); mx.eval(out, *store["hidden"].values())
            hs = mx.concatenate([store["hidden"][i][None] for i in range(nL+1)], axis=0)[:, 0]
            h = np.array(hs[:, -1, :].astype(mx.float32))     # [L+1,H]
            if prev is not None:
                d = np.abs(h - prev)
                maxd = d if maxd is None else np.maximum(maxd, d)
            prev = h
        return maxd

    data = []
    for dom, msgs, lab in trajectories():
        s = traj_transitions(msgs)
        if s is not None: data.append((dom, lab, s))
    doms = sorted({d[0] for d in data})
    print(f"model={model_id}\ntrajectories={len(data)} unsafe={sum(d[1] for d in data)} "
          f"safe={sum(1-d[1] for d in data)}")
    print("per-domain:", {dm: sum(1 for d in data if d[0]==dm) for dm in doms})

    print(f"\n{'held-out domain':16}{'layer*':>7}{'AUROC':>8}")
    aur=[]
    for ho in doms:
        tr=[i for i in range(len(data)) if data[i][0]!=ho]
        te=[i for i in range(len(data)) if data[i][0]==ho]
        if len({data[i][1] for i in te})<2: print(f"{ho:16}  (skip single-class)"); continue
        cut=int(len(tr)*0.8); itr,iv=tr[:cut],tr[cut:]
        bestL,bv=0,-1
        for L in range(nL+1):
            X=np.array([data[i][2][L] for i in itr]); y=np.array([data[i][1] for i in itr])
            d=X[y==1].mean(0)-X[y==0].mean(0); d/=np.linalg.norm(d)+1e-9
            v=auroc([data[i][2][L]@d for i in iv],[data[i][1] for i in iv])
            if v==v and v>bv: bv,bestL=v,L
        X=np.array([data[i][2][bestL] for i in tr]); y=np.array([data[i][1] for i in tr])
        d=X[y==1].mean(0)-X[y==0].mean(0); d/=np.linalg.norm(d)+1e-9
        a=auroc([data[i][2][bestL]@d for i in te],[data[i][1] for i in te]); aur.append(a)
        print(f"{ho:16}{bestL:7d}{a:8.2f}")
    print(f"\nmean leave-one-domain-out AUROC: {np.nanmean(aur):.3f}")

if __name__ == "__main__":
    if "--download" in sys.argv: download()
    else: main(sys.argv[1])
