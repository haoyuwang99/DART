#!/usr/bin/env python3
"""White-box introspection for MLX LLMs.

Runs a single forward pass over a prompt and captures internal states:
  - hidden_states  : residual stream after every layer (+ embeddings)
  - logit_lens     : top-k next-token prediction read off each layer's residual
  - entropy        : per-position next-token uncertainty (final layer)
  - moe_routing    : which experts each token routes to (MoE models only)

Big tensors are written to a .npz; compact summaries are returned inline (hybrid).

This works by temporarily monkeypatching the decoder-layer class __call__ (to grab
the residual stream) and the MoE block class __call__ (to grab router logits),
running one forward pass, then restoring the originals. No model code is modified.
"""
from __future__ import annotations
import contextlib
import numpy as np
import mlx.core as mx
import mlx.nn as nn


def _final_logits(h_normed_last_proj):  # placeholder for readability
    raise NotImplementedError


class Introspector:
    def __init__(self, model, tokenizer, model_name: str):
        self.model = model
        self.tok = tokenizer
        self.name = model_name
        inner = model.model                       # the transformer body
        self.layers = inner.layers
        self.final_norm = inner.norm              # pre-unembedding RMSNorm
        self.embed = inner.embed_tokens
        self.lm_head = model.lm_head
        self.n_layers = len(self.layers)
        # Detect MoE layers + the router attribute name per layer
        self.router_attr = {}                     # layer_idx -> "gate" | "router"
        for i, L in enumerate(self.layers):
            mlp = getattr(L, "mlp", None)
            if mlp is None:
                continue
            # gate/router may be quantized (QuantizedLinear), so detect by the
            # presence of an expert bank (switch_mlp / experts), not by type.
            if hasattr(mlp, "gate") and (hasattr(mlp, "switch_mlp") or hasattr(mlp, "experts")):
                self.router_attr[i] = "gate"      # Qwen3-MoE
            elif hasattr(mlp, "router"):
                self.router_attr[i] = "router"    # gpt-oss
        self.is_moe = len(self.router_attr) > 0

    # ---- helpers ---------------------------------------------------------
    def _topk(self, logits_vec, k):
        """logits_vec: mx.array [vocab] -> list of {id, token, prob}."""
        probs = np.array(mx.softmax(logits_vec.astype(mx.float32), axis=-1))
        idx = np.array(mx.argpartition(logits_vec, kth=-k)[-k:])
        p = probs[idx]
        order = np.argsort(-p)
        out = []
        for j in order:
            tid = int(idx[j])
            out.append({"id": tid,
                        "token": self.tok.decode([tid]),
                        "prob": round(float(p[j]), 5)})
        return out

    @contextlib.contextmanager
    def _capture(self, store):
        """Patch layer + MoE __call__ to record states into `store`."""
        layer_cls = type(self.layers[0])
        orig_layer = layer_cls.__call__

        def layer_call(self_layer, x, *a, **kw):
            idx = self_layer._cap_idx
            if idx == 0:
                store["hidden"][0] = x            # embeddings (input to layer 0)
            out = orig_layer(self_layer, x, *a, **kw)
            store["hidden"][idx + 1] = out
            return out

        patched = [(layer_cls, "__call__", orig_layer)]
        layer_cls.__call__ = layer_call

        # Patch each distinct MoE block class to record routing
        moe_classes = {}
        for i, attr in self.router_attr.items():
            mlp = self.layers[i].mlp
            moe_classes.setdefault(type(mlp), attr)
        for cls, attr in moe_classes.items():
            orig_moe = cls.__call__

            def make(orig_moe, attr):
                def moe_call(self_mlp, x, *a, **kw):
                    router = getattr(self_mlp, attr)
                    g = router(x)                                  # [1, L, n_experts]
                    gates = mx.softmax(g.astype(mx.float32), axis=-1)
                    k = getattr(self_mlp, "top_k",
                                getattr(self_mlp, "num_experts_per_tok", 4))
                    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                    store["route"][self_mlp._cap_idx] = {
                        "experts": np.array(inds)[0],              # [L, k]
                        "weights": np.array(mx.take_along_axis(gates, inds, axis=-1))[0],
                    }
                    return orig_moe(self_mlp, x, *a, **kw)
                return moe_call
            cls.__call__ = make(orig_moe, attr)
            patched.append((cls, "__call__", orig_moe))

        # tag instances with their layer index
        for i, L in enumerate(self.layers):
            L._cap_idx = i
            if i in self.router_attr:
                L.mlp._cap_idx = i
        try:
            yield
        finally:
            for cls, nm, orig in patched:
                setattr(cls, nm, orig)

    # ---- main API --------------------------------------------------------
    def run(self, text, *, apply_chat_template=True, top_k=8,
            lens_layers=None, position=-1, npz_path=None):
        if apply_chat_template:
            ids = self.tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True)
        else:
            ids = self.tok.encode(text)
        ids = mx.array(ids)[None]                 # [1, L]
        L = ids.shape[1]
        toks = [self.tok.decode([int(t)]) for t in ids[0]]

        store = {"hidden": {}, "route": {}}
        with self._capture(store):
            logits = self.model(ids)              # [1, L, vocab]; fills store
            mx.eval(logits, *[v for v in store["hidden"].values()])

        # stack residual stream: [n_layers+1, L, H]
        hs = mx.concatenate([store["hidden"][i][None] for i in
                             range(self.n_layers + 1)], axis=0)[:, 0]  # drop batch
        H = hs.shape[-1]

        pos = position if position >= 0 else L + position
        if lens_layers is None:                   # default: sample ~12 layers across depth
            lens_layers = sorted(set(np.linspace(0, self.n_layers, 12, dtype=int).tolist()))

        # --- logit lens: project each chosen layer's residual @ `pos` ---
        lens = []
        for li in lens_layers:
            h = hs[li, pos]                        # [H]
            lg = self.lm_head(self.final_norm(h[None]))[0]
            lens.append({"layer": int(li), "top": self._topk(lg, top_k)})

        # --- entropy per position (final-layer next-token dist) ---
        probs = mx.softmax(logits[0].astype(mx.float32), axis=-1)
        ent = -mx.sum(probs * mx.log(probs + 1e-12), axis=-1)
        ent = np.array(ent)

        # --- hidden-state norms: [n_layers+1, L] ---
        norms = np.array(mx.linalg.norm(hs.astype(mx.float32), axis=-1))

        # --- MoE routing summary @ pos ---
        routing = None
        if self.is_moe:
            routing = []
            for li in sorted(store["route"]):
                r = store["route"][li]
                e, w = r["experts"][pos], r["weights"][pos]
                order = np.argsort(-w)
                routing.append({"layer": int(li),
                                "experts": [int(e[j]) for j in order],
                                "weights": [round(float(w[j]), 4) for j in order]})

        result = {
            "model": self.name,
            "n_layers": self.n_layers,
            "hidden_size": int(H),
            "is_moe": self.is_moe,
            "n_tokens": int(L),
            "tokens": toks,
            "analysis_position": int(pos),
            "predicted_next": self._topk(logits[0, pos], top_k),
            "logit_lens": lens,
            "entropy_per_token": [round(float(x), 4) for x in ent],
            "hidden_norm_per_layer_at_pos": [round(float(norms[li, pos]), 3)
                                             for li in range(self.n_layers + 1)],
            "moe_routing_at_pos": routing,
        }

        if npz_path:
            arrays = {"hidden_states": np.array(hs.astype(mx.float32)),
                      "entropy": ent, "hidden_norms": norms,
                      "tokens": np.array(toks, dtype=object)}
            if self.is_moe:
                for li, r in store["route"].items():
                    arrays[f"route_experts_l{li}"] = r["experts"]
                    arrays[f"route_weights_l{li}"] = r["weights"]
            np.savez_compressed(npz_path, **arrays)
            result["npz_path"] = npz_path
            result["npz_contents"] = sorted(arrays.keys())

        return result
