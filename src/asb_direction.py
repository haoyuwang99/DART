#!/usr/bin/env python3
"""Can a better/augmented LINEAR direction beat difference-of-means on ASB intrinsic-misuse detection?
Fully offline on the step-2 probe reps (log/asb_probe_<model>.npz) + judge labels (log/asb_probe_harm.jsonl).

Estimators (all linear; a trajectory scores = max over its transition projections onto w):
  diffmean   : w = mean(pos) - mean(neg)                      [current baseline; isotropic-cov LDA]
  ridge      : regularized least-squares classifier, dual form w = X^T (XX^T + lam I)^-1 y  [proper reg. LDA]
  ridge+smote: ridge after SMOTE oversampling of the minority (unsafe) class   [data augmentation]
  ridge+jit  : ridge after Gaussian jitter + same-class mixup augmentation      [data augmentation]
  ridge+adv  : ridge, then hard-example mining (upweight boundary neg/pos), refit [adversarial-ish]

Matched 5-fold CV over cases; layer + lam picked on TRAIN case-AUROC only (fair across methods, no test
leak). Features standardized on train. Reports held-out case-level AUROC per model + macro-average."""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from dart.monitor import auroc

PROBE = "log/asb_probe"; HARM = "log/asb_probe_harm.jsonl"
LAYER_FRACS = [0.3, 0.4, 0.5, 0.6, 0.7]
LAMS = [0.3, 1.0, 3.0, 10.0]


def _load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ---- augmentations (operate on standardized train transitions) ----
def smote(X, y, rng):
    Xp = X[y == 1]; n_gen = int((y == 0).sum() - (y == 1).sum())
    if n_gen <= 0 or len(Xp) < 2:
        return X, y
    i = rng.integers(0, len(Xp), n_gen); j = rng.integers(0, len(Xp), n_gen); r = rng.random((n_gen, 1))
    syn = Xp[i] + r * (Xp[j] - Xp[i])
    return np.vstack([X, syn]), np.concatenate([y, np.ones(n_gen)])


def jitter_mixup(X, y, rng, mult=0.3):
    parts, yp = [X], [y]
    parts.append(X + rng.normal(0, mult, X.shape).astype(X.dtype)); yp.append(y)   # jitter
    for c in (0, 1):                                                                # same-class mixup
        Xc = X[y == c]
        if len(Xc) >= 2:
            i = rng.integers(0, len(Xc), len(Xc)); j = rng.integers(0, len(Xc), len(Xc)); r = rng.random((len(Xc), 1))
            parts.append(Xc[i] + r * (Xc[j] - Xc[i])); yp.append(np.full(len(Xc), c))
    return np.vstack(parts), np.concatenate(yp)


def ridge_dir(X, y, lam):
    yb = np.where(y == 1, 1.0, -1.0)                                                # balanced-ish via oversampling upstream
    G = X @ X.T
    alpha = np.linalg.solve(G + lam * np.eye(len(yb)), yb)
    return X.T @ alpha


def fit_dir(X, y, method, lam, rng):
    if method == "diffmean":
        return X[y == 1].mean(0) - X[y == 0].mean(0)
    Xa, ya = X, y
    if method == "ridge+smote":
        Xa, ya = smote(X, y, rng)
    elif method == "ridge+jit":
        Xa, ya = jitter_mixup(X, y, rng)
    elif method == "ridge+adv":
        w0 = ridge_dir(*smote(X, y, rng), lam=lam); s = X @ w0                      # hard: high-scoring neg / low-scoring pos
        neg = np.where(y == 0)[0]; pos = np.where(y == 1)[0]
        hard_n = neg[np.argsort(s[neg])[-max(1, len(neg)//4):]]
        hard_p = pos[np.argsort(s[pos])[:max(1, len(pos)//4)]]
        Xa = np.vstack([X, X[hard_n], X[hard_n], X[hard_p], X[hard_p]])             # upweight hard x3
        ya = np.concatenate([y, y[hard_n], y[hard_n], y[hard_p], y[hard_p]])
        Xa, ya = smote(Xa, ya, rng)
    return ridge_dir(Xa, ya, lam)


def case_scores(by_case, cids, L, w, mu, sd):
    return np.array([float((((by_case[c][:, L, :].astype(np.float32) - mu) / sd) @ w).max()) for c in cids])


def cv_auroc(by_case, label, method, seed=0, folds=5):
    cids = [c for c in by_case if label.get(c) in (0, 1)]
    y = {c: label[c] for c in cids}
    rng = np.random.default_rng(seed)
    pos = [c for c in cids if y[c] == 1]; neg = [c for c in cids if y[c] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    fp = [pos[i::folds] for i in range(folds)]; fn = [neg[i::folds] for i in range(folds)]
    nL = next(iter(by_case.values())).shape[1]
    layers = sorted(set(min(nL - 1, max(1, int(f * nL))) for f in LAYER_FRACS))
    scores, labs = [], []
    for f in range(folds):
        te = set(fp[f]) | set(fn[f]); tr = [c for c in cids if c not in te]
        if not tr or not te or not (0 < sum(y[c] for c in tr) < len(tr)):
            continue
        best = None
        for L in layers:
            Xtr = np.concatenate([by_case[c][:, L, :].astype(np.float32) for c in tr], 0)
            ytr = np.concatenate([np.full(by_case[c].shape[0], y[c]) for c in tr])
            Xtr = np.nan_to_num(Xtr)
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6; Xs = (Xtr - mu) / sd
            for lam in (LAMS if method != "diffmean" else [1.0]):
                w = fit_dir(Xs, ytr, method, lam, np.random.default_rng(seed + f))
                strain = np.array([(((by_case[c][:, L, :].astype(np.float32) - mu) / sd) @ w).max() for c in tr])
                au = auroc(strain, np.array([y[c] for c in tr]))
                if best is None or au > best[0]:
                    best = (au, L, w, mu, sd)
        _, L, w, mu, sd = best
        for c in (set(fp[f]) | set(fn[f])):
            scores.append(case_scores(by_case, [c], L, w, mu, sd)[0]); labs.append(y[c])
    s, yy = np.array(scores), np.array(labs)
    return auroc(s, yy) if 0 < yy.sum() < len(yy) else float("nan")


def main():
    harm = _load(HARM)
    models = sorted({r["model"] for r in harm})
    methods = ["diffmean", "ridge", "ridge+smote", "ridge+jit", "ridge+adv"]
    print(f'{"model":24}' + "".join(f"{m:>13}" for m in methods) + f'{"n+/n":>10}')
    agg = collections.defaultdict(list)
    for name in models:
        npz = f"{PROBE}_{name}.npz"
        if not os.path.exists(npz):
            continue
        d = np.load(npz); reps = d["reps"]; cid_of = d["case_id"]
        by_case = {int(c): reps[cid_of == c] for c in np.unique(cid_of)}
        lab = {r["case_id"]: r["harm"] for r in harm if r["model"] == name}
        row = {}
        for meth in methods:
            row[meth] = cv_auroc(by_case, lab, meth)
            if not np.isnan(row[meth]):
                agg[meth].append(row[meth])
        nu = sum(1 for c in by_case if lab.get(c) == 1); n = sum(1 for c in by_case if lab.get(c) in (0, 1))
        sh = name.replace("-8bit", "").replace("-Instruct", "").replace("Meta-", "")
        print(f'{sh:24}' + "".join(f"{row[m]:>13.3f}" for m in methods) + f'{f"{nu}/{n}":>10}')
    print(f'{"MACRO-AVG":24}' + "".join(f"{np.mean(agg[m]):>13.3f}" for m in methods))


if __name__ == "__main__":
    main()
