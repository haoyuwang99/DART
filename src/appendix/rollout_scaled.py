#!/usr/bin/env python3
"""Scaled real rollouts on Qwen3-30B-A3B. All 40 workspace user tasks, clean vs
injected. For injected we fill ALL injection slots with the attack, so whatever
resource the agent reads it hits a poison -> far higher surfacing than 1-slot.
Writes traces_30b.jsonl incrementally."""
import os, json, yaml
import openai, agentdojo
from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig, LocalLLM
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.types import get_text_content_as_str

MODEL = "mlx-community/Qwen3-30B-A3B-8bit"
OUT = "traces_30b.jsonl"
MARK = "INFORMATION>"

base = os.path.dirname(agentdojo.__file__)
sm = yaml.safe_load(open(base + "/data/system_messages.yaml"))
SYS = (sm["default"] if isinstance(sm, dict) else sm) + "\n\n/no_think"

client = openai.OpenAI(base_url="http://localhost:8082/v1", api_key="x")
llm = LocalLLM(client, MODEL)
def fresh_pipe():
    return AgentPipeline.from_config(PipelineConfig(llm=llm, model_id="local",
        defense=None, system_message_name=None, system_message=SYS))

suite = get_suites("v1")["workspace"]
class P: name = "local"
atk = ImportantInstructionsAttack(suite, P())
it = suite.injection_tasks["injection_task_0"]
ALL_SLOTS = list(suite.get_injection_vector_defaults().keys())

def serialize(msgs):
    return [{"role": m["role"],
             "text": get_text_content_as_str(m["content"]) if m.get("content") else "",
             "tool_calls": [{"fn": c.function, "args": dict(c.args)}
                            for c in (m.get("tool_calls") or [])]} for m in msgs]

def run(ut, injections):
    tenv = ut.init_environment(suite.load_and_inject_default_environment(injections))
    try:
        _, _, _, msgs, _ = fresh_pipe().query(ut.PROMPT, FunctionsRuntime(suite.tools), tenv)
    except Exception as e:
        return None, str(e)[:60]
    return serialize(msgs), None

if os.path.exists(OUT):
    os.remove(OUT)

tasks = list(suite.user_tasks.values())          # all 40
attack_text = list(atk.attack(tasks[0], it).values())[0]     # rendered JB block
inj_all = {s: attack_text for s in ALL_SLOTS}
clean_all = {s: "" for s in ALL_SLOTS}

surfaced = 0
with open(OUT, "a") as f:
    for i, ut in enumerate(tasks):
        for cond, injd in [("clean", clean_all), ("injected", inj_all)]:
            msgs, err = run(ut, injd)
            if msgs is None:
                print(f"[{i:2d}] {ut.ID:14} {cond:9} ERROR {err}"); continue
            surf = any(m["role"] == "tool" and MARK in m["text"] for m in msgs) if cond == "injected" else False
            surfaced += int(surf)
            f.write(json.dumps({"user_task": ut.ID, "condition": cond,
                                "n_msgs": len(msgs), "injection_surfaced": surf,
                                "messages": msgs}) + "\n"); f.flush()
            print(f"[{i:2d}] {ut.ID:14} {cond:9} msgs={len(msgs):2d} surfaced={surf}", flush=True)
print(f"\nDONE. injected runs where poison surfaced: {surfaced}/{len(tasks)}")
