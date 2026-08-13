#!/usr/bin/env python3
"""Run REAL AgentDojo rollouts with our MLX model (via mlx_lm.server) and log the
full message traces to disk. Each user task is run twice: clean and injected.

Writes traces.jsonl incrementally (one JSON object per run) so partial progress
is saved. A run's 'injection_surfaced' flags whether the poisoned content actually
appeared in a tool result (i.e. the agent read it)."""
import os, json, yaml
import openai, agentdojo
from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig, LocalLLM
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.types import get_text_content_as_str

N_TASKS = 15
OUT = "traces.jsonl"
MARK = "INFORMATION>"          # marker of the important_instructions injection

base = os.path.dirname(agentdojo.__file__)
sm = yaml.safe_load(open(base + "/data/system_messages.yaml"))
SYS = (sm["default"] if isinstance(sm, dict) else sm) + "\n\n/no_think"

client = openai.OpenAI(base_url="http://localhost:8082/v1", api_key="x")
llm = LocalLLM(client, "mlx-community/Qwen3-8B-8bit")
def fresh_pipe():
    return AgentPipeline.from_config(PipelineConfig(llm=llm, model_id="local",
        defense=None, system_message_name=None, system_message=SYS))

suite = get_suites("v1")["workspace"]
class P: name = "local"
atk = ImportantInstructionsAttack(suite, P())
it = suite.injection_tasks["injection_task_0"]

def serialize(msgs):
    out = []
    for m in msgs:
        txt = get_text_content_as_str(m["content"]) if m.get("content") else ""
        out.append({"role": m["role"], "text": txt,
                    "tool_calls": [{"fn": c.function, "args": dict(c.args)}
                                   for c in (m.get("tool_calls") or [])]})
    return out

def run(ut, injections):
    tenv = ut.init_environment(suite.load_and_inject_default_environment(injections))
    try:
        _, _, _, msgs, _ = fresh_pipe().query(ut.PROMPT, FunctionsRuntime(suite.tools), tenv)
    except Exception as e:
        return None, str(e)
    return serialize(msgs), None

if os.path.exists(OUT):
    os.remove(OUT)

tasks = list(suite.user_tasks.values())[:N_TASKS]
surfaced = 0
with open(OUT, "a") as f:
    for i, ut in enumerate(tasks):
        inj = atk.attack(ut, it)
        clean = {k: "" for k in inj}
        for cond, injd in [("clean", clean), ("injected", inj)]:
            msgs, err = run(ut, injd)
            if msgs is None:
                print(f"[{i}] {ut.ID:14} {cond:9} ERROR {err[:50]}"); continue
            surfaced_here = any(r["role"] == "tool" and MARK in r["text"] for r in msgs) \
                if cond == "injected" else False
            surfaced += int(surfaced_here)
            rec = {"user_task": ut.ID, "condition": cond, "n_msgs": len(msgs),
                   "injection_surfaced": surfaced_here, "messages": msgs}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{i}] {ut.ID:14} {cond:9} msgs={len(msgs):2d} "
                  f"surfaced={surfaced_here}")
print(f"\nDONE. injected runs where poison surfaced in a tool result: {surfaced}")
