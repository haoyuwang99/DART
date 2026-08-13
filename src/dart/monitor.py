#!/usr/bin/env python3
"""The monitor: detection only. Fit the reading direction u = read(C) (difference-of-means over
injected/clean transitions), score each representation transition r_i = (h_i - h_{i-1})·u, and
evaluate held-out AUROC. `Monitor` is the online scorer; `fit_direction` is the reference read(C)."""
import json, numpy as np
from dart.agent import _ctx_to_msgs, MARK


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(s); rank = np.empty(len(s)); rank[order] = np.arange(1, len(s) + 1)
    return (rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def read(pos, neg):
    """Difference-of-means reading direction, unit norm, per layer.
    pos,neg: [n, L+1, H] per-layer representations. Returns [L+1, H]."""
    d = pos.mean(0) - neg.mean(0)
    return d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8)


def pick_layer(P, N, lo, hi):
    """Inner-validation-best layer for the difference-of-means direction, in depth band [lo,hi]."""
    nL = P.shape[1]; rng = np.random.default_rng(0)
    def half(a):
        i = rng.permutation(len(a)); c = max(2, len(a) // 2); return a[i[:c]], a[i[c:]]
    Ptr, Pva = half(P); Ntr, Nva = half(N); U = read(Ptr, Ntr)
    def vauc(L):
        return auroc([p[L] @ U[L] for p in Pva] + [n[L] @ U[L] for n in Nva], [1] * len(Pva) + [0] * len(Nva))
    return max(range(max(1, int(lo * nL)), int(hi * nL) + 1), key=lambda L: (vauc(L) if vauc(L) == vauc(L) else -1))


def fit_um(P, N, lo, hi):
    """difference-of-means reading direction u at the inner-validation-best layer in band [lo,hi]."""
    L = pick_layer(P, N, lo, hi)
    u = P[:, L].mean(0) - N[:, L].mean(0); u /= np.linalg.norm(u) + 1e-9
    return u, L


def detection_auroc(rows):
    """Held-out trajectory-level detection AUROC from emitted records (harmful vs benign test scores)."""
    te = [r for r in rows if r["split"] == "test"]
    pos = [r["score"] for r in te if r["kind"] == "harmful"]; neg = [r["score"] for r in te if r["kind"] == "benign"]
    return auroc(pos + neg, [1] * len(pos) + [0] * len(neg)) if pos and neg else float("nan")


class Monitor:
    """Online transition-level monitor: scores r_i = (h_i - h_{i-1}) . u and flags r_i > tau."""

    def __init__(self, u, layer, tau):
        self.u = np.asarray(u, float); self.layer = layer; self.tau = tau; self.h_prev = None

    def reset(self, h0):
        self.h_prev = np.asarray(h0, float)

    def observe(self, h_i):
        h_i = np.asarray(h_i, float)
        r = float((h_i - self.h_prev) @ self.u)
        self.h_prev = h_i
        return r, (r > self.tau)


def fit_direction(lm, model_id, suite_name, n_pairs=30, traces="traces_campaign.jsonl"):
    """read(C): matched injected/clean tool-result contexts -> u, layer, tau (controlled perturbation).
    Layer L* is chosen by VALIDATION AUROC on an inner split (as in the method), not train AUROC,
    and tau at 5% validation FPR. Fitting and monitoring use the identical hidden() so u transfers."""
    pos, neg = [], []
    for line in open(traces):
        r = json.loads(line)
        if r["model"] != model_id or r["suite"] != suite_name:
            continue
        def delta(msgs, j):   # transition induced by segment j: h(<=j) - h(<=j-1)
            return lm.hidden(_ctx_to_msgs(msgs, j)) - lm.hidden(_ctx_to_msgs(msgs, j - 1))
        j = next((k for k, m in enumerate(r["messages"]) if m["role"] == "tool" and MARK in (m["text"] or "")), None)
        if r["condition"] == "injected" and r["injection_surfaced"] and j and len(pos) < n_pairs:
            pos.append(delta(r["messages"], j))                                # [L+1, H] transition
        elif r["condition"] == "clean" and len(neg) < n_pairs:
            k = next((k for k, m in enumerate(r["messages"]) if m["role"] == "tool" and k > 0), None)
            if k:
                neg.append(delta(r["messages"], k))
    pos, neg = np.stack(pos), np.stack(neg)                                    # [n, L+1, H]
    nL = pos.shape[1]
    rng = np.random.default_rng(0)
    def half(a):
        i = rng.permutation(len(a)); c = max(2, len(a) // 2); return a[i[:c]], a[i[c:]]
    ptr, pva = half(pos); ntr, nva = half(neg)
    Utr = read(ptr, ntr)                                                       # fit on inner-train
    def val_auc(L):
        return auroc([p[L] @ Utr[L] for p in pva] + [n[L] @ Utr[L] for n in nva],
                     [1] * len(pva) + [0] * len(nva))
    # search the later half of depth, where the concept is represented (early layers separate
    # only on surface features like length); pick argmax validation AUROC there.
    scored = [(L, val_auc(L)) for L in range(nL // 2, nL)]
    layer = max(scored, key=lambda t: (t[1] if t[1] == t[1] else -1))[0]
    u = read(pos, neg)[layer]                                                  # refit on all data at L*
    tau = float(np.percentile([n[layer] @ u for n in neg], 95))               # 5% FPR on negatives
    a = auroc([p[layer] @ u for p in pos] + [n[layer] @ u for n in neg], [1] * len(pos) + [0] * len(neg))
    print(f"read(C): L*={layer}/{nL - 1} by val-AUROC; full-set AUROC={a:.3f}  tau={tau:.2f}  "
          f"(|T+|={len(pos)}, |T-|={len(neg)})")
    return u, layer, tau
