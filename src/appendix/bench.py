#!/usr/bin/env python3
"""Simple MLX LLM benchmark: load a model, run prompts, report speed + memory.

Usage:
    python bench.py mlx-community/Qwen3-8B-8bit
    python bench.py mlx-community/Qwen3-30B-A3B-8bit --max-tokens 256 --no-think
"""
import argparse, time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

PROMPTS = [
    ("short_qa", "What is the capital of France? Answer in one word."),
    ("reasoning", "A farmer has 17 sheep. All but 9 run away. How many are left? Think briefly, then answer."),
    ("agent_toolcall", "You are an agent with tools read_file(path) and edit_file(path, content). "
        "A unit test in tests/test_math.py fails because add() returns a-b instead of a+b in src/math.py. "
        "List the exact sequence of tool calls you would make. Be concise."),
    ("code_gen", "Write a Python function `is_palindrome(s)` that ignores case and non-alphanumeric chars. Code only."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--no-think", action="store_true", help="append /no_think (Qwen3) to disable reasoning")
    args = ap.parse_args()

    print(f"\n{'='*70}\nLoading {args.model} ...")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"Load time: {time.time()-t0:.1f}s | weights in memory: {mx.get_active_memory()/1e9:.2f} GB\n")

    sampler = make_sampler(temp=0.7, top_p=0.95)
    tot_gen_tok, tot_gen_time = 0, 0.0
    for name, prompt in PROMPTS:
        p = prompt + (" /no_think" if args.no_think else "")
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        t0 = time.time()
        out = generate(model, tokenizer, prompt=text, max_tokens=args.max_tokens,
                       sampler=sampler, verbose=False)
        dt = time.time() - t0
        ntok = len(tokenizer.encode(out))
        tps = ntok / dt if dt else 0
        tot_gen_tok += ntok; tot_gen_time += dt
        print(f"[{name:14}] {ntok:4d} tok in {dt:6.2f}s = {tps:6.1f} tok/s")
        print(f"   -> {out.strip()[:220].replace(chr(10),' ')}{'...' if len(out)>220 else ''}\n")

    print(f"{'-'*70}")
    print(f"AVG generation speed: {tot_gen_tok/tot_gen_time:.1f} tok/s")
    print(f"Peak memory: {mx.get_peak_memory()/1e9:.2f} GB")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
