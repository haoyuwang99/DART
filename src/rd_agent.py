#!/usr/bin/env python3
"""CLI demo of the closed-loop MonitoredAgent (Algorithm 1) --- thin wrapper over the dart package.
Run from the repo root:  .venv311/bin/python src/rd_agent.py <model_id> <suite> [n_fit_pairs]
"""
import sys
from dart.agent import HiddenLM, MonitoredAgent, REMINDER
from dart.monitor import Monitor, fit_direction
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-8B-8bit"
    suite_name = sys.argv[2] if len(sys.argv) > 2 else "slack"
    n_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    lm = HiddenLM(model_id)
    u, layer, tau = fit_direction(lm, model_id, suite_name, n_pairs)
    monitor = Monitor(u, layer, tau)

    suite = get_suites("v1")[suite_name]
    atk = ImportantInstructionsAttack(suite, type("P", (), {"name": "local"})())
    it0 = list(suite.injection_tasks.values())[0]
    ut = list(suite.user_tasks.values())[0]
    inj = {s: list(atk.attack(ut, it0).values())[0] for s in suite.get_injection_vector_defaults()}
    env = ut.init_environment(suite.load_and_inject_default_environment(inj))

    agent = MonitoredAgent(lm, suite.tools, monitor, REMINDER)
    print(f"\n=== running {suite_name}/{ut.ID} (injected) through MonitoredAgent ===")
    msgs, events, traces = agent.run(ut.PROMPT, env)
    print("\nper-step monitor:")
    for e in events:
        print(f"  step {e['step']}  {e['tool']:28} r={e['r']:>8}  {'FLAG -> reminder' if e['flagged'] else ''}")
    final = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
    print(f"\nflags: {sum(e['flagged'] for e in events)}/{len(events)} transitions;  final: {final[:160]}")


if __name__ == "__main__":
    main()
