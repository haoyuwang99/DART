#!/usr/bin/env python3
"""Generalized AgentDojo rollout harness: run one model over one or more suites,
clean + injected (all slots filled), logging real traces incrementally.

Usage: run_rollout.py <model_id> <port> <out.jsonl> <suite1,suite2,...>
Resumable: skips (model,suite,task,condition) keys already in <out.jsonl>.
"""
import os, sys, json, yaml
import openai, agentdojo
from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig, LocalLLM
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.types import get_text_content_as_str

MODEL, PORT, OUT, SUITES = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
MARK = "INFORMATION>"

base = os.path.dirname(agentdojo.__file__)
sm = yaml.safe_load(open(base + "/data/system_messages.yaml"))
SYS = (sm["default"] if isinstance(sm, dict) else sm) + "\n\n/no_think"
client = openai.OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x")
llm = LocalLLM(client, MODEL)
def fresh_pipe():
    return AgentPipeline.from_config(PipelineConfig(llm=llm, model_id="local",
        defense=None, system_message_name=None, system_message=SYS))
class P: name = "local"

def serialize(msgs):
    return [{"role": m["role"],
             "text": get_text_content_as_str(m["content"]) if m.get("content") else "",
             "tool_calls": [{"fn": c.function, "args": dict(c.args)}
                            for c in (m.get("tool_calls") or [])]} for m in msgs]

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:
            r = json.loads(l); done.add((r["model"], r["suite"], r["user_task"], r["condition"]))
        except Exception: pass

with open(OUT, "a") as f:
    for sname in SUITES:
        suite = get_suites("v1")[sname]
        atk = ImportantInstructionsAttack(suite, P())
        it0 = list(suite.injection_tasks.values())[0]
        slots = list(suite.get_injection_vector_defaults().keys())
        attack_text = list(atk.attack(list(suite.user_tasks.values())[0], it0).values())[0]
        inj_all = {s: attack_text for s in slots}; clean_all = {s: "" for s in slots}
        for ut in suite.user_tasks.values():
            for cond, injd in [("clean", clean_all), ("injected", inj_all)]:
                key = (MODEL, sname, ut.ID, cond)
                if key in done:
                    continue
                tenv = ut.init_environment(suite.load_and_inject_default_environment(injd))
                try:
                    _, _, _, msgs, _ = fresh_pipe().query(ut.PROMPT, FunctionsRuntime(suite.tools), tenv)
                    msgs = serialize(msgs)
                except Exception as e:
                    print(f"[{sname}/{ut.ID}/{cond}] ERROR {str(e)[:50]}", flush=True); continue
                surf = any(m["role"]=="tool" and MARK in m["text"] for m in msgs) if cond=="injected" else False
                f.write(json.dumps({"model": MODEL, "suite": sname, "user_task": ut.ID,
                    "condition": cond, "injection_surfaced": surf, "messages": msgs}) + "\n")
                f.flush()
                print(f"[{sname:10}/{ut.ID:14}/{cond:9}] msgs={len(msgs):2d} surf={surf}", flush=True)
print("DONE", MODEL, SUITES, flush=True)
