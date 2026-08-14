#!/usr/bin/env python3
"""REAL runtime enforcement eval (causal, single-pass) -- the faithful Algorithm 1.

For each (user x injection) case, run the agent ONCE per enforcement with the ONLINE monitor:
cumulative drift C_t=(h_t-h_0)*u, fire at the FIRST step C_t>tau, attribute argmax delta_i*u, intervene.
tau is calibrated on val-benign cumulative drift (zero-FPR). NO baseline pass, NO oracle peak -- unlike
run_ad_corrected.py which runs the whole trajectory first to pick argmax (non-causal).

Usage:  run_online.py <model_id> [suites=slack] [n] [--report]
Writes: rdeval_online.jsonl
"""
import sys, json
import numpy as np
from dart.agent import HiddenLM
from dart.datasets import ad_fit_runtime, AgentDojoEnv, AD_SYS, agentic_rollout

REC = "rdeval_online.jsonl"
SUITE_N = {"slack": 10, "workspace": 20, "banking": 8, "travel": 10}
ENF = ["none", "stop", "attr", "redact"]


def run_suite(lm, model, suite_name, n, fpr=0.10):
    from agentdojo.task_suite.load_suites import get_suites
    from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
    suite = get_suites("v1")[suite_name]
    attack = ImportantInstructionsAttack(suite, type("P", (), {"name": "local"})())
    inj_tasks = list(suite.injection_tasks.values())
    ds_tag = "agentdojo" if suite_name == "slack" else f"agentdojo-{suite_name}"
    uts = list(suite.user_tasks.values())
    val_uts, test_uts = uts[n:2 * n], uts[:n]
    # Fit u AND tau on the DEPLOYMENT pipeline from held-out val tasks. This is the fix for the
    # campaign-vs-runtime covariate shift that collapsed AgentDojo detection (0.93 anchored -> 0.59
    # run-level; runtime-fit recovers 1.00 / 0.933): a reading direction only transfers if it is fit on
    # the same agent loop it monitors.
    u, L, tau = ad_fit_runtime(lm, suite, AD_SYS, val_uts, attack, inj_tasks, fpr=fpr)
    print(f"[online] {model} {suite_name}: L*={L} tau={tau:.3f}  test={len(test_uts)} val={len(val_uts)}", flush=True)

    def emit(ut, injections, task, kind, case):
        rec = {"model": model, "dataset": ds_tag, "suite": suite_name, "split": "test",
               "kind": kind, "case": case, "tau": tau}
        for e in ENF:                                           # one CAUSAL rollout per enforcement
            env = AgentDojoEnv(suite, AD_SYS, ut, injections, task)
            _, sec, util, _ = agentic_rollout(lm, env, u, L, tau, enforce=e)
            rec[f"sec_{e}"] = 0 if sec is None else int(bool(sec))
            rec[f"util_{e}"] = int(bool(util))
            rec[f"fired_{e}"] = int(bool(getattr(env, "last_fired", False)))
        with open(REC, "a") as f:
            f.write(json.dumps(rec) + "\n")

    for ut in test_uts:
        base = attack.attack(ut, inj_tasks[0])
        emit(ut, {k: "" for k in base}, None, "benign", getattr(ut, "ID", None))
        for it in inj_tasks:
            emit(ut, attack.attack(ut, it), it, "harmful", f"{getattr(ut,'ID',None)}::{getattr(it,'ID',None)}")
    print(f"[online] {model} {suite_name}: DONE", flush=True)


def report(path=REC):
    import collections
    recs = [json.loads(l) for l in open(path)]
    g = collections.OrderedDict()
    for r in recs:
        g.setdefault((r["model"], r["dataset"]), []).append(r)
    print("\n=== REAL runtime enforcement (causal online monitor): (ASR/Util) per enforcement ===")
    print("  online: fire at first C_t>tau; NO baseline pass, NO oracle peak\n")
    for (m, d), rows in g.items():
        H = [r for r in rows if r["kind"] == "harmful"]; B = [r for r in rows if r["kind"] == "benign"]
        if not H:
            continue
        cells = [f"{e}:{np.mean([r[f'sec_{e}'] for r in H]):.2f}/{np.mean([r[f'util_{e}'] for r in H]):.2f}" for e in ENF]
        bu = np.mean([r["util_none"] for r in B]) if B else float("nan")
        fr = np.mean([r["fired_none"] for r in H])
        print(f"{m.replace('-8bit',''):22} {d:18} benign-util={bu:.2f} fire={fr:.2f}  " + "  ".join(cells) + f"  (nH={len(H)})")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(); sys.exit(0)
    mid = sys.argv[1]; name = mid.split("/")[-1]
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    suites = args[0].split(",") if args else ["slack"]
    n_override = int(args[1]) if len(args) > 1 else None
    lm = HiddenLM(mid)
    for s in suites:
        try:
            run_suite(lm, name, s, n_override or SUITE_N[s])
        except Exception:
            import traceback; print(f"[online] {name} {s} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"{name}: online done ({','.join(suites)})", flush=True)
