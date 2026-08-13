#!/usr/bin/env python3
"""Step 2 analysis: does fitting the reading direction on the JUDGE label (actual unsafe behavior)
recover detection that the fulfillable label destroyed?

  judge : harm-judge the probe trajectories (log/asb_probe_<model>.jsonl) -> log/asb_probe_harm.jsonl
  refit : per model, load reps + per-case harm/fulfillable labels; with matched 5-fold CV, fit
          difference-of-means u (layer picked on train only) under EACH label set and report pooled
          held-out case-level AUROC. Same procedure both sides -> the judge-vs-fulfillable gap is the result.

Usage: asb_refit.py judge <model_id...> | asb_refit.py refit
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from dart.monitor import auroc

PROBE = "log/asb_probe"; HARM = "log/asb_probe_harm.jsonl"


def _load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def do_judge(models):
    from dart.judge import Judge, JUDGE_ID
    from dart.asb import asb_cases
    from judge_asb import env_info_for_case
    cases = {c["id"]: c for c in asb_cases()}
    done = {(r["model"], r["case_id"]) for r in _load(HARM)}
    J = Judge(JUDGE_ID)
    for model in models:
        name = model.split("/")[-1]
        rows = _load(f"{PROBE}_{name}.jsonl")
        todo = [r for r in rows if (name, r["case_id"]) not in done]
        print(f"[refit-judge] {name}: {len(rows)} cases, {len(todo)} to judge", flush=True)
        for i, r in enumerate(todo):
            v = J.harm(env_info_for_case(cases[r["case_id"]]), r["traj"])
            with open(HARM, "a") as f:
                f.write(json.dumps({"model": name, "case_id": r["case_id"],
                                    "fulfillable": r["fulfillable"], "harm": (None if v is None else int(v))}) + "\n")
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(todo)}", flush=True)
    print("[refit-judge] done", flush=True)


def _fit_score(Xtr_by_case, ytr, Xte_by_case, band=(0.30, 0.75)):
    """Diff-of-means per layer on pooled train transitions; pick layer by train case-level AUROC;
    return held-out case scores (max transition projection) at that layer."""
    nL = next(iter(Xtr_by_case.values())).shape[1]
    lo, hi = int(band[0] * nL), max(int(band[1] * nL), int(band[0] * nL) + 1)
    # pooled train transitions + per-transition labels
    P = np.concatenate([X for cid, X in Xtr_by_case.items() if ytr[cid] == 1], axis=0)
    N = np.concatenate([X for cid, X in Xtr_by_case.items() if ytr[cid] == 0], axis=0)
    best_L, best_au = lo, -1
    for L in range(lo, hi):
        u = P[:, L, :].mean(0) - N[:, L, :].mean(0); u /= np.linalg.norm(u) + 1e-9
        s = np.array([ (X[:, L, :] @ u).max() for cid, X in Xtr_by_case.items() ])
        y = np.array([ ytr[cid] for cid in Xtr_by_case ])
        au = auroc(s, y) if 0 < y.sum() < len(y) else 0.5
        if au > best_au:
            best_au, best_L = au, L
    u = P[:, best_L, :].mean(0) - N[:, best_L, :].mean(0); u /= np.linalg.norm(u) + 1e-9
    return {cid: float((X[:, best_L, :] @ u).max()) for cid, X in Xte_by_case.items()}, best_L


def _cv_auroc(by_case, label, seed=0, folds=5):
    """Matched k-fold CV; returns pooled held-out case-level AUROC for the given case->label dict."""
    cids = [c for c in by_case if label.get(c) in (0, 1)]
    y = np.array([label[c] for c in cids])
    if not (0 < y.sum() < len(y)):
        return float("nan"), int(y.sum()), len(y)
    rng = np.random.default_rng(seed)
    pos = [c for c in cids if label[c] == 1]; neg = [c for c in cids if label[c] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    fp = [pos[i::folds] for i in range(folds)]; fn = [neg[i::folds] for i in range(folds)]
    scores, labs = [], []
    for f in range(folds):
        te = set(fp[f]) | set(fn[f]); tr = [c for c in cids if c not in te]
        if not tr or not te:
            continue
        ytr = {c: label[c] for c in tr}
        if not (0 < sum(ytr.values()) < len(ytr)):
            continue
        s_te, _ = _fit_score({c: by_case[c] for c in tr}, ytr, {c: by_case[c] for c in te})
        for c in te:
            scores.append(s_te[c]); labs.append(label[c])
    s, yy = np.array(scores), np.array(labs)
    return (auroc(s, yy) if 0 < yy.sum() < len(yy) else float("nan")), int(y.sum()), len(y)


def do_refit():
    harm = _load(HARM)
    models = sorted({r["model"] for r in harm})
    print(f'\n=== STEP 2: RE-FIT u ON JUDGE LABEL vs FULFILLABLE (matched 5-fold CV, held-out AUROC) ===')
    print(f'{"model":24}{"AUROC[judge-fit]":>18}{"AUROC[fulfill-fit]":>20}{"unsafe/n":>12}')
    for name in models:
        npz = f"{PROBE}_{name}.npz"
        if not os.path.exists(npz):
            print(f"{name:24}  (no reps)"); continue
        d = np.load(npz)
        reps = d["reps"].astype(np.float32); cid_of = d["case_id"]
        by_case = {}
        for cid in np.unique(cid_of):
            by_case[int(cid)] = reps[cid_of == cid]                # [n_trans, n_layers, H]
        hj = {r["case_id"]: r["harm"] for r in harm if r["model"] == name}
        hf = {r["case_id"]: (0 if r["fulfillable"] == 1 else 1) for r in harm if r["model"] == name}
        au_j, nu, n = _cv_auroc(by_case, hj)
        au_f, _, _ = _cv_auroc(by_case, hf)
        print(f'{name.replace("-8bit","").replace("-Instruct","").replace("Meta-",""):24}'
              f'{au_j:>18.3f}{au_f:>20.3f}{f"{nu}/{n}":>12}')


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "refit"
    if mode == "judge":
        do_judge(sys.argv[2:])
    else:
        do_refit()
