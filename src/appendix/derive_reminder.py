#!/usr/bin/env python3
"""Derive the corrective reminder FROM the contrastive pairs (not hand-written).

The same P (misbehavior) / N (benign) contrast that defines the reading direction also says
what the misbehavior IS -- so a model can verbalize their difference into a corrective
instruction. Here we sample injected vs clean tool outputs from the real traces and ask the
model for a one-sentence reminder. This is the general, injection-agnostic construction; the
hand-written note becomes its special case. Usage: derive_reminder.py <model_id>
"""
import sys, json, random
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

MODEL = sys.argv[1]
MARK = "<INFORMATION>"
random.seed(0)

def tool_snippets():
    pos, neg = [], []
    for l in open("traces_campaign.jsonl"):
        r = json.loads(l)
        for m in r["messages"]:
            if m["role"] != "tool" or not m.get("text"):
                continue
            t = m["text"].replace("\n", " ").strip()
            if r["condition"] == "injected" and MARK in m["text"] and len(pos) < 60:
                pos.append(t[:320])
            elif r["condition"] == "clean" and MARK not in m["text"] and len(neg) < 60:
                neg.append(t[:220])
    return pos, neg

model, tok = load(MODEL)
pos, neg = tool_snippets()
P = random.sample(pos, 4); N = random.sample(neg, 4)

blockP = "\n".join(f"  [A{i+1}] {s}" for i, s in enumerate(P))
blockN = "\n".join(f"  [B{i+1}] {s}" for i, s in enumerate(N))
prompt = (
    "You are hardening an autonomous LLM agent against context that derails it.\n"
    "Below are tool outputs the agent received while working on a user's task.\n"
    "GROUP A outputs contain instructions inserted by third parties that try to make the agent "
    "act against the user; GROUP B outputs are normal.\n\n"
    f"GROUP A (derailing):\n{blockP}\n\nGROUP B (normal):\n{blockN}\n\n"
    "Write ONE short, GENERAL reminder (a single sentence) to append to the agent's context that "
    "will keep it on the user's ORIGINAL task despite GROUP A content. State the rule generally "
    "(about how to treat instructions embedded in retrieved or tool content); do NOT mention any "
    "specific name, tool, address, or value from the examples. Output only the reminder text.")

msgs = [{"role": "user", "content": prompt + " /no_think"}]
ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
c = make_prompt_cache(model)
lg = model(mx.array(ids)[None], cache=c); t = int(mx.argmax(lg[0, -1])); out = []
for _ in range(120):
    out.append(t)
    if t == tok.eos_token_id:
        break
    t = int(mx.argmax(model(mx.array([[t]]), cache=c)[0, -1]))
text = tok.decode(out)
for j in ("<think>", "</think>", "<|im_end|>", "<|endoftext|>"):
    text = text.replace(j, "")
text = text.strip().strip('"')
print("=== DERIVED REMINDER (from the injected-vs-clean contrast) ===")
print(text)
