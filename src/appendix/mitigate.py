#!/usr/bin/env python3
"""Closed-loop mitigation evaluation -- Algorithm 1, end-to-end and ONLINE.

Runs the reference MonitoredAgent (rd_agent.py) through real AgentDojo (user_task x injection_task)
rollouts and scores them with AgentDojo's own utility/security checkers, monitoring OFF vs ON:

  OFF  monitor disabled (tau=inf): the agent runs unguarded -> baseline attack-success + utility.
  ON   the closed loop: each tool-result transition is scored r_i = (h_i - h_{i-1}).u; when r_i > tau
       the triggering segment is attributed and the corrective reminder rho is inserted before the
       next action; the agent then continues.

The reading direction u, layer L*, and threshold tau come from fit_direction (difference-of-means on
matched injected/clean tool-result transitions; tau at 5% FPR on clean negatives) -- the SAME monitor
the detection eval scored. We report, per suite:

  ASR      attack-success rate  (injection_task.security True; LOWER is better)
  utility  benign-task success  (user_task.utility True; HIGHER is better -- must be preserved)

for OFF and ON, plus the monitor's flag counts. A working closed loop drops ASR while keeping utility.

Usage: mitigate.py <model_id> <suite> [n_user_tasks=8] [n_injection_tasks=2]
"""
import sys, json, numpy as np
from rd_agent import HiddenLM, Monitor, MonitoredAgent, fit_direction, REMINDER
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.types import text_content_block_from_string


def main():
    model_id = sys.argv[1]
    suite_name = sys.argv[2] if len(sys.argv) > 2 else "slack"
    n_ut = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    n_it = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    lm = HiddenLM(model_id)
    u, layer, tau = fit_direction(lm, model_id, suite_name)          # operating monitor: u, L*, tau
    suite = get_suites("v1")[suite_name]
    class P:
        name = "local"
    attack = ImportantInstructionsAttack(suite, P())
    user_tasks = list(suite.user_tasks.values())[:n_ut]
    inj_tasks = list(suite.injection_tasks.values())[:n_it]
    print(f"\n=== closed-loop mitigation: {suite_name}  ({len(user_tasks)} user x {len(inj_tasks)} injection tasks) ===")
    print(f"{'user_task':16}{'injection':18}{'mode':5}{'util':>6}{'attack':>8}{'flags':>7}", flush=True)

    rows = []
    for ut in user_tasks:
        for it in inj_tasks:
            inj = attack.attack(ut, it)
            for mode in ("off", "on"):
                mon = Monitor(u, layer, tau if mode == "on" else float("inf"))   # off never flags
                agent = MonitoredAgent(lm, suite.tools, mon, REMINDER)
                env = ut.init_environment(suite.load_and_inject_default_environment(inj))
                pre = env.model_copy(deep=True)
                try:
                    msgs, events, traces = agent.run(ut.PROMPT, env)
                except Exception as e:
                    print(f"{ut.ID:16}{it.ID:18}{mode:5}  ERROR {str(e)[:40]}"); continue
                out = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
                ob = [text_content_block_from_string(out)]
                util = bool(suite._check_task_result(ut, ob, pre, env, traces))
                sec = bool(suite._check_task_result(it, ob, pre, env, traces))   # True = attack succeeded
                flags = sum(e["flagged"] for e in events)
                rows.append({"mode": mode, "util": util, "sec": sec, "flags": flags})
                print(f"{ut.ID:16}{it.ID:18}{mode:5}{int(util):>6}{int(sec):>8}{flags:>7}", flush=True)

    def rate(mode, key):
        v = [r[key] for r in rows if r["mode"] == mode]
        return sum(v) / len(v) if v else float("nan")
    n_pairs = len(rows) // 2
    print(f"\n=== SUMMARY  {suite_name}  (n={n_pairs} pairs, model={model_id.split('/')[-1]}) ===")
    print(f"{'mode':6}{'ASR (attack success)':>24}{'utility':>12}{'mean flags':>12}")
    for mode in ("off", "on"):
        print(f"{mode:6}{rate(mode, 'sec'):>24.3f}{rate(mode, 'util'):>12.3f}{rate(mode, 'flags'):>12.2f}")
    a_off, a_on = rate("off", "sec"), rate("on", "sec")
    u_off, u_on = rate("off", "util"), rate("on", "util")
    print(f"\nASR {a_off:.3f} -> {a_on:.3f}  (blocked {a_off - a_on:+.3f});  "
          f"utility {u_off:.3f} -> {u_on:.3f}  (delta {u_on - u_off:+.3f})")
    # machine-readable line for grid aggregation
    print(f"RESULT model={model_id.split('/')[-1]} suite={suite_name} pairs={n_pairs} "
          f"asr_off={a_off:.3f} asr_on={a_on:.3f} util_off={u_off:.3f} util_on={u_on:.3f} "
          f"flags_on={rate('on', 'flags'):.2f}", flush=True)
    with open("mitigate_results.jsonl", "a") as f:
        f.write(json.dumps({"model": model_id.split("/")[-1], "suite": suite_name, "pairs": n_pairs,
                            "asr_off": a_off, "asr_on": a_on, "util_off": u_off, "util_on": u_on,
                            "flags_on": rate("on", "flags")}) + "\n")


if __name__ == "__main__":
    main()
