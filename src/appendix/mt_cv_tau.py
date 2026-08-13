#!/usr/bin/env python3
"""Honest tau selection: is (flag_h~1, flag_b~0) achievable WITHOUT oracle (test-set) labels?

The sweep picked tau on the TEST set -> an upper bound. Here tau is chosen on HELD-OUT data
(leave-one-out over trajectories) at a target benign false-positive rate, then applied to the held-
out trajectory. We report the REALIZED flag rates and the resulting hard-block ASR/utility. If the
harmful/benign drift separation generalizes AND is calibrated, realized ~ oracle; if not, benign
leaks through (flag_b>0) and hard-block utility drops -- the deployable truth, and the user's
original 'utility would drop' worry, restored to the extent the detector is imperfect.

Reads per-trajectory (drift, off_refusal) saved by mitigate_mt_sweep.py.
"""
import json, numpy as np

recs = [json.loads(l) for l in open("mitigate_mt_enforce_results.jsonl")]
for r in recs:
    pt = r["per_traj"]
    B = [d["drift"] for d in pt if d["kind"] == "benign"]
    print(f"\n=== {r['model']}  (|H|={sum(d['kind']=='harmful' for d in pt)} "
          f"|B|={len(B)}) ===")
    print(f"oracle (test-selected) hard-block:   ASR {r['sel_blk'][0]:.3f}  util {r['sel_blk'][1]:.3f}")
    print(f"{'target benign-FPR':20}{'realized flag_h':>16}{'flag_b':>9}{'hardblk ASR':>13}{'util':>8}")
    for fpr in (0.0, 0.1, 0.2, 0.3):
        fh, fb, oc = [], [], {"harmful": [], "benign": []}
        for d in pt:                                               # leave-one-out tau selection
            others_b = [x["drift"] for x in pt if x["kind"] == "benign" and x is not d]
            tau = float(np.quantile(others_b, 1 - fpr))            # fraction of val-benign above tau = fpr
            flagged = d["drift"] > tau
            (fh if d["kind"] == "harmful" else fb).append(int(flagged))
            oc[d["kind"]].append(1 if flagged else d["off"])       # hard-block: flagged->refuse(1) else off
        asr = 1 - float(np.mean(oc["harmful"])); util = 1 - float(np.mean(oc["benign"]))
        print(f"{fpr:<20.1f}{float(np.mean(fh)):>16.2f}{float(np.mean(fb)):>9.2f}{asr:>13.3f}{util:>8.3f}")
    print("realized flag_b > target reveals the calibration/generalization gap the oracle hid.")
