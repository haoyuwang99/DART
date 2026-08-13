#!/usr/bin/env python3
"""Introspection API for local MLX models.

Endpoints:
  GET  /health                      -> ok + loaded models
  GET  /models                      -> known aliases
  POST /generate   {model,prompt,max_tokens}        -> plain completion
  POST /introspect {model,prompt,...}               -> completion-style forward pass
                                                       + internal states (hybrid)

Run:  uvicorn server:app --port 8081
"""
from __future__ import annotations
import os, time, hashlib, threading
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate
from introspect import Introspector

ALIASES = {
    "qwen3-8b":     "mlx-community/Qwen3-8B-8bit",
    "qwen3-30b":    "mlx-community/Qwen3-30B-A3B-8bit",
    "gpt-oss-120b": "halley-ai/gpt-oss-120b-MLX-6bit-gs64",
}
NPZ_DIR = os.path.join(os.path.dirname(__file__), "introspect_runs")
os.makedirs(NPZ_DIR, exist_ok=True)

app = FastAPI(title="MLX Introspection API")
_cache: dict = {}          # repo_id -> (model, tok, Introspector)
_lock = threading.Lock()


def _resolve(name: str) -> str:
    return ALIASES.get(name, name)


def _get(name: str):
    repo = _resolve(name)
    with _lock:                       # serialize loads (and GPU use) across requests
        if repo not in _cache:
            t0 = time.time()
            model, tok = load(repo)
            ins = Introspector(model, tok, repo)
            _cache[repo] = (model, tok, ins, time.time() - t0)
        return _cache[repo]


class GenReq(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 256


class IntrospectReq(BaseModel):
    model: str
    prompt: str
    chat_template: bool = False
    top_k: int = 8
    lens_layers: Optional[List[int]] = None
    position: int = -1
    save_npz: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(_cache.keys()), "aliases": ALIASES}


@app.get("/models")
def models():
    return {"aliases": ALIASES, "loaded": list(_cache.keys())}


@app.post("/generate")
def gen(req: GenReq):
    model, tok, _ins, _ = _get(req.model)
    with _lock:
        t0 = time.time()
        out = generate(model, tok, prompt=tok.apply_chat_template(
            [{"role": "user", "content": req.prompt}], add_generation_prompt=True),
            max_tokens=req.max_tokens, verbose=False)
        dt = time.time() - t0
    return {"model": _resolve(req.model), "text": out,
            "n_tokens": len(tok.encode(out)), "seconds": round(dt, 2),
            "tok_per_sec": round(len(tok.encode(out)) / dt, 1)}


@app.post("/introspect")
def introspect(req: IntrospectReq):
    try:
        _m, _t, ins, _load_dt = _get(req.model)
    except Exception as e:
        raise HTTPException(400, f"could not load model: {e}")
    npz_path = None
    if req.save_npz:
        h = hashlib.sha1(f"{ins.name}:{req.prompt}:{time.time()}".encode()).hexdigest()[:10]
        npz_path = os.path.join(NPZ_DIR, f"{h}.npz")
    with _lock:
        t0 = time.time()
        r = ins.run(req.prompt, apply_chat_template=req.chat_template,
                    top_k=req.top_k, lens_layers=req.lens_layers,
                    position=req.position, npz_path=npz_path)
        r["forward_seconds"] = round(time.time() - t0, 3)
    return r
