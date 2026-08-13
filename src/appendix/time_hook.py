import time, numpy as np, mlx.core as mx
from mlx_lm import load
from introspect import Introspector
mid=[{"role":"user","content":"Tell me a fact about the ocean in one sentence."}]
# time WITHOUT the hook (raw mlx_lm)
model,tok=load("mlx-community/Qwen3-8B-8bit"); ins=Introspector(model,tok,"x")
def fwd():
    ids=tok.encode(tok.apply_chat_template(mid,add_generation_prompt=True,tokenize=False))
    st={"hidden":{},"route":{}}
    with ins._capture(st):
        out=model(mx.array(ids)[None]); mx.eval(out)
t0=time.time(); [fwd() for _ in range(10)]; print(f"NO-HOOK: {(time.time()-t0)/10*1000:.0f} ms/fwd")
# now install the steering hook (same as HiddenLM) and re-time
layers=model.model.layers; idx={id(l):i for i,l in enumerate(layers)}; LC=type(layers[0]); _o=LC.__call__
STEER={"vec":None,"blk":-1}
def _p(sl,x,*a,**k):
    out=_o(sl,x,*a,**k)
    if STEER["vec"] is not None and idx.get(id(sl),-1)==STEER["blk"]: out=out+STEER["vec"]
    return out
LC.__call__=_p
t0=time.time(); [fwd() for _ in range(10)]; print(f"HOOK(no-steer): {(time.time()-t0)/10*1000:.0f} ms/fwd")
