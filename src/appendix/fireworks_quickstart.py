"""Fireworks AI quickstart — first chat completion via the OpenAI-compatible API.

Run:
    source /Users/haoyu/AIDX/BenchScore/env.sh   # sets FIREWORKS_API_KEY
    .venv311/bin/python fireworks_quickstart.py
"""
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["FIREWORKS_API_KEY"],
    base_url="https://api.fireworks.ai/inference/v1",
)

MODEL = "accounts/fireworks/models/deepseek-v4-pro"

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Say hello in Spanish"}],
)

print(f"model:  {response.model}")
print(f"reply:  {response.choices[0].message.content}")
if response.usage:
    print(f"tokens: {response.usage.total_tokens}")
