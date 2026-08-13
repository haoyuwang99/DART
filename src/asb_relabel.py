#!/usr/bin/env python3
"""Corrected ASB analysis. KEY FIX: `fulfillable` is NOT harmful-vs-benign -- it marks whether the
request has a safe fulfillment (0 = must refuse; 1 = complete while avoiding traps). ASB scores EVERY
trajectory safe/unsafe (ShieldAgent), regardless of fulfillable. We previously harm-judged only
fulfillable=0 and completion-judged fulfillable=1, so fulfillable=1 unsafe behavior was uncounted.

  judge : harm-judge ALL remaining test trajectories (the fulfillable=1 ones) -> log/asb_harm.jsonl
  report: ASB true unsafe rate (all cases) + detection AUROC of the drift monitor against the JUDGE
          label (actual unsafe behavior) vs the old fulfillable label, per model and pooled (z-scored).
          This isolates whether the monitor detects BEHAVIOR, not the noisy task-intent label."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from dart.monitor import auroc

TRAJ = "log/asb_traj.jsonl"; VERD = "log/asb_verdicts.jsonl"; HARM = "log/asb_harm.jsonl"


def _load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def do_judge():
    from dart.judge import Judge, JUDGE_ID
    from dart.asb import asb_cases
    from judge_asb import env_info_for_case
    cases = {c["id"]: c for c in asb_cases()}
    traj = [r for r in _load(TRAJ) if r["split"] == "test" and r.get("traj")]
    have = set()
    for r in _load(VERD):
        if r["kind"] == "harmful":                      # existing harmful verdicts ARE harm verdicts
            have.add((r["model"], r["case_id"], r["variant"]))
    for r in _load(HARM):
        have.add((r["model"], r["case_id"], r["variant"]))
    todo = [r for r in traj if (r["model"], r["case_id"], r["variant"]) not in have]
    print(f"[harm] {len(traj)} test traj total, {len(todo)} to harm-judge (rest cached)", flush=True)
    J = Judge(JUDGE_ID); nn = 0
    for i, r in enumerate(todo):
        v = J.harm(env_info_for_case(cases[r["case_id"]]), r["traj"]); nn += (v is None)
        with open(HARM, "a") as f:
            f.write(json.dumps({"model": r["model"], "case_id": r["case_id"], "kind": r["kind"],
                                "variant": r["variant"], "harm": (None if v is None else int(v))}) + "\n")
        if (i + 1) % 30 == 0:
            print(f"[harm] {i+1}/{len(todo)} (unparsed {nn})", flush=True)
    print(f"[harm] done, unparsed {nn}/{len(todo)}", flush=True)


def harm_map():
    """(model,case_id,variant) -> harm verdict (1=unsafe). Harmful from VERD, fulfillable=1 from HARM."""
    H = {}
    for r in _load(VERD):
        if r["kind"] == "harmful":
            H[(r["model"], r["case_id"], r["variant"])] = r["verdict"]
    for r in _load(HARM):
        H[(r["model"], r["case_id"], r["variant"])] = r["harm"]
    return H


def do_report():
    traj = _load(TRAJ); H = harm_map()
    models = sorted({r["model"] for r in traj})
    SHORT = {m: m.replace("-8bit", "").replace("-Instruct", "").replace("Meta-", "") for m in models}
    Zpos, Zneg = [], []
    print("\n=== CORRECTED ASB DETECTION (judge label = actual unsafe behavior, not fulfillable) ===")
    print(f'{"model":24}{"unsafe%":>9}{"AUROC[judge]":>14}{"AUROC[fulfill]":>16}{"n_unsafe/n":>12}')
    for m in models:
        off = [r for r in traj if r["model"] == m and r["split"] == "test" and r["variant"] == "off"]
        lab = [H.get((m, r["case_id"], "off")) for r in off]
        sc = [r["score"] for r in off]
        pairs = [(s, l) for s, l in zip(sc, lab) if l in (0, 1)]
        y = np.array([l for _, l in pairs]); s = np.array([v for v, _ in pairs], float)
        au_j = auroc(s, y) if 0 < y.sum() < len(y) else float("nan")
        yf = np.array([1 if r["kind"] == "harmful" else 0 for r in off]); sf = np.array(sc, float)
        au_f = auroc(sf, yf) if 0 < yf.sum() < len(yf) else float("nan")
        safe_sc = [v for v, l in zip(sc, lab) if l == 0]
        mu, sd = (np.mean(safe_sc), np.std(safe_sc) + 1e-9) if safe_sc else (0.0, 1.0)
        for v, l in zip(sc, lab):
            if l == 1: Zpos.append((v - mu) / sd)
            elif l == 0: Zneg.append((v - mu) / sd)
        unsafe_rate = np.mean([l for l in lab if l in (0, 1)]) if pairs else float("nan")
        print(f'{SHORT[m]:24}{unsafe_rate:>9.2f}{au_j:>14.3f}{au_f:>16.3f}{f"{int(y.sum())}/{len(y)}":>12}')
    Zpos, Zneg = np.array(Zpos), np.array(Zneg)
    sp = np.concatenate([Zpos, Zneg]); yp = np.r_[np.ones(len(Zpos)), np.zeros(len(Zneg))]
    au = auroc(sp, yp) if len(Zpos) and len(Zneg) else float("nan")
    print(f"\nPOOLED (per-model z-scored) detection of ACTUAL unsafe behavior:")
    print(f"  AUROC = {au:.3f}   (n_unsafe={len(Zpos)}, n_safe={len(Zneg)})")
    print("  [uses the OLD fulfillable-fit direction u; step 2 re-fits u on judge labels]")


if __name__ == "__main__":
    (do_judge if (len(sys.argv) < 2 or sys.argv[1] == "judge") else do_report)()
