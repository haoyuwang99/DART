#!/usr/bin/env python3
"""Aggregate enforce_layer.jsonl into layer x alpha grids (flip, suppress) + per-layer signal."""
import json, os
from collections import defaultdict
F = "enforce_layer.jsonl"
if not os.path.exists(F): print("no enforce_layer.jsonl yet"); raise SystemExit
rows = [json.loads(l) for l in open(F)]
print(f"{len(rows)} evals\n")
fracs = sorted(set(r["frac"] for r in rows)); alphas = sorted(set(r["alpha"] for r in rows))
Ls = sorted(set((r["layer"], r["frac"]) for r in rows))

def grid(metric, title):
    print(f"=== {title}: rows=layer(frac), cols=alpha ===")
    print(f"{'layer':>12}" + "".join(f"{('a='+str(a)):>10}" for a in alphas))
    for L, fr in Ls:
        cells = []
        for a in alphas:
            v = [r[metric] for r in rows if r["layer"] == L and r["alpha"] == a]
            cells.append(f"{sum(v)/len(v):.2f}({len(v)})" if v else "  -  ")
        print(f"{('L%d f%.2f' % (L, fr)):>12}" + "".join(f"{c:>10}" for c in cells))
    print()

grid("flip", "CLEAN-FLIP")
grid("suppress", "SUPPRESS")

# per-layer signal: mean separation (p_inj - clean_ref) — where is the direction strongest?
print("=== per-layer signal: mean(p_inj - clean_ref) and |.| (injected-vs-clean separation) ===")
sep = defaultdict(list)
for r in rows: sep[(r["layer"], r["frac"])].append(r["p_inj"] - r["clean_ref"])
print(f"{'layer':>12}{'mean sep':>10}{'|sep|':>9}{'clean_ref':>11}")
for L, fr in Ls:
    v = sep[(L, fr)]; import statistics as st
    cr = st.mean([r["clean_ref"] for r in rows if r["layer"] == L])
    print(f"{('L%d f%.2f' % (L, fr)):>12}{st.mean(v):>10.1f}{st.mean([abs(x) for x in v]):>9.1f}{cr:>11.1f}")

# best flip cells
bycell = defaultdict(lambda: [0, 0])
for r in rows:
    c = bycell[(r["frac"], r["alpha"])]; c[0] += r["flip"]; c[1] += 1
ranked = sorted(bycell.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1))
print("\ntop flip cells (frac, alpha) -> flips/n:")
for (fr, a), (k, n) in ranked[:6]:
    print(f"  frac={fr} alpha={a}: {k}/{n} ({k/n:.2f})")
