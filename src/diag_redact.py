#!/usr/bin/env python3
"""Diagnostic: run ONE AgentDojo harmful rollout, print each tool-result text and what redact_result
removes at the peak segment. Confirms (i) live tool results are line-splittable, (ii) redaction removes
the injected span. Usage: diag_redact.py <model_id>"""
import sys, json, numpy as np
from dart.agent import HiddenLM, _parse_calls, MARK, SYS as AD_SYS
from dart.datasets import ad_fit, AD_TEST
from dart.mitigation import redact_result, _spans

model_id = sys.argv[1]; name = model_id.split("/")[-1]
lm = HiddenLM(model_id); u, L = ad_fit(lm, name)
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt
suite = get_suites("v1")[AD_TEST]
attack = ImportantInstructionsAttack(suite, type("P", (), {"name": "local"})())
it = list(suite.injection_tasks.values())[0]

# find the first user task whose injection surfaces and succeeds under no defense
for ut in list(suite.user_tasks.values())[:10]:
    injections = attack.attack(ut, it)
    runtime = FunctionsRuntime(suite.tools)
    env = ut.init_environment(suite.load_and_inject_default_environment(injections))
    pre = env.model_copy(deep=True)
    msgs = [{"role": "system", "content": _make_system_prompt(AD_SYS, runtime.functions.values())},
            {"role": "user", "content": ut.PROMPT}]
    scores, traces, ntool, tool_idx = [], [], 0, []
    for _ in range(8):
        completion = lm.generate(msgs, max_tokens=384)
        msgs.append({"role": "assistant", "content": completion})
        calls = _parse_calls(completion)
        if not calls: break
        hp = lm.hidden(msgs, L)
        for tc in calls:
            result, err = runtime.run_function(env, tc.function, dict(tc.args)); traces.append(tc)
            text = err if err else tool_result_to_str(result)
            msgs.append({"role": "tool", "content": json.dumps({"result": text})})
            tool_idx.append(len(msgs) - 1)
            hi = lm.hidden(msgs, L); scores.append(float((hi - hp) @ u)); hp = hi
            ntool += 1
    if not scores:
        continue
    peak = int(np.argmax(scores)); jpk = tool_idx[peak]
    ptext = json.loads(msgs[jpk]["content"])["result"]
    has_inj = MARK in ptext or "<INFORMATION>" in ptext
    print(f"\n### {ut.ID}  peak-seg={peak}/{len(scores)}  peak-score={scores[peak]:.2f}  jmsg={jpk}")
    print(f"peak tool-result: {len(_spans(ptext))} line-spans, {len(ptext)} chars, injection_in_peak={has_inj}")
    print("---- peak tool-result text (first 600 chars) ----")
    print(ptext[:600])
    newc, nred = redact_result(lm, msgs, jpk, u, L)
    red = json.loads(newc)["result"]
    print(f"---- redaction removed {nred} spans; {len(ptext)-len(red)} chars ----")
    print(f"injection_still_present_after_redact = {MARK in red or '<INFORMATION>' in red}")
    print("---- redacted text (first 600 chars) ----")
    print(red[:600])
    # only need one informative example that had the injection at peak
    if has_inj:
        break
