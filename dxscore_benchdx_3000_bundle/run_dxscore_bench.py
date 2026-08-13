"""Feed each prompt in a JSONL benchmark file into every Fireworks chat model
and save the raw responses (one line per prompt-model pair).

Usage:
    source /Users/haoyu/AIDX/BenchScore/env.sh          # sets FIREWORKS_API_KEY
    .venv311/bin/python run_dxscore_bench.py [--limit N] [--workers 12]

Long-format output (resumable): each line = one (prompt, model) result carrying
all original record fields plus model/response/usage/latency/error.
Re-running skips (traceid, model) pairs already present in the output file.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

BASE_URL = "https://api.fireworks.ai/inference/v1"

# 5 chat models on this account. flux-1-schnell-fp8 is an image model -> excluded.
DEFAULT_MODELS = [
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/gpt-oss-120b",
]

DEFAULT_INPUT = "/Users/haoyu/Downloads/dxscore_benchdx_3000.jsonl"
MAX_RETRIES = 4


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path):
    """Return set of (traceid, model) already written, for resume."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    done.add((r.get("traceid"), r.get("model")))
                except json.JSONDecodeError:
                    continue
    return done


def call_model(client, model, prompt, max_tokens):
    t0 = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            usage = resp.usage
            return {
                "response": choice.message.content,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "latency_s": round(time.time() - t0, 2),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 - record any failure, retry transient ones
            msg = f"{type(e).__name__}: {e}"
            transient = any(s in msg for s in ("429", "500", "502", "503", "504", "Timeout", "Connection"))
            if transient and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "response": None,
                "finish_reason": "error",
                "prompt_tokens": None,
                "completion_tokens": None,
                "latency_s": round(time.time() - t0, 2),
                "error": msg,
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N records (smoke test)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        sys.exit("FIREWORKS_API_KEY not set — run: source /Users/haoyu/AIDX/BenchScore/env.sh")

    models = args.models or DEFAULT_MODELS
    out_path = args.output or (
        os.path.splitext(args.input)[0] + ("_smoke" if args.limit else "") + "_outputs.jsonl"
    )

    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]
    done = load_done(out_path)

    tasks = [
        (rec, model)
        for rec in records
        for model in models
        if (rec.get("traceid"), model) not in done
    ]
    total = len(records) * len(models)
    print(f"records={len(records)}  models={len(models)}  "
          f"tasks total={total}  already done={total - len(tasks)}  to run={len(tasks)}")
    if not tasks:
        print("nothing to do.")
        return

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    write_lock = threading.Lock()
    out_f = open(out_path, "a")
    counter = {"n": 0, "err": 0, "ctok": 0}
    t_start = time.time()

    def worker(rec, model):
        result = call_model(client, model, rec["prompt"], args.max_tokens)
        row = {**rec, "model": model, **result}
        line = json.dumps(row, ensure_ascii=False)
        with write_lock:
            out_f.write(line + "\n")
            out_f.flush()
            counter["n"] += 1
            if result["error"]:
                counter["err"] += 1
            if result["completion_tokens"]:
                counter["ctok"] += result["completion_tokens"]
            n = counter["n"]
            if n % 25 == 0 or n == len(tasks):
                rate = n / (time.time() - t_start)
                eta = (len(tasks) - n) / rate if rate else 0
                print(f"  {n}/{len(tasks)}  errors={counter['err']}  "
                      f"{rate:.1f}/s  eta={eta/60:.1f}m  compl_tokens={counter['ctok']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, rec, model) for rec, model in tasks]
        for _ in as_completed(futures):
            pass

    out_f.close()
    dt = time.time() - t_start
    print(f"\ndone in {dt/60:.1f}m -> {out_path}")
    print(f"  wrote {counter['n']} rows, {counter['err']} errors, "
          f"{counter['ctok']} completion tokens")


if __name__ == "__main__":
    main()
