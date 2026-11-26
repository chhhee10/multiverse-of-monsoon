"""Multiverse of Monsoon inference server.

The judge runs this exact file offline:  python serve.py --port 8000
Contract (team kit API_CONTRACT.md): POST /health, POST /simulate; physical units on the wire, nested JSON lists.

The model (monsoon/engine.py) is loaded and warmed up once at import, so /health only reports ready when a
rollout is already fast. Only the repo's own files are read; nothing is written and no network is used.
"""
import argparse
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from monsoon.engine import Engine

torch.set_num_threads(4)
ENGINE = Engine.load()
ENGINE.simulate(np.repeat(ENGINE.norm.inv(ENGINE.clim_z[150])[None].numpy(), 3, axis=0), [0.5, -0.87], 2)  # warm-up
MODEL_NAME = ENGINE.name
MAX_STEPS = 30

app = FastAPI()
# The dashboard (a separate static web app) calls this server from the browser.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "OPTIONS"], allow_headers=["*"])


def _arr(x, shape):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated); check the shape."""
    if isinstance(x, dict):
        import base64
        a = np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    else:
        a = np.asarray(x, dtype="f4")
    if a.shape != shape:
        raise ValueError(f"expected shape {shape}, got {a.shape}")
    return a


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta (added to the last day)
    or None. Returns [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    return ENGINE.simulate(init_states, doy_sincos, steps, perturbation)


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0 = time.perf_counter()
    try:
        body = await request.json()
        init = _arr(body["init_states"], (3, 6, 32, 36))
        pert = None if body.get("perturbation") is None else _arr(body["perturbation"], (6, 32, 36))
        sc = [float(v) for v in body["doy_sincos"]][:2]
        steps = int(body["steps"])
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be 1..{MAX_STEPS}")
    except Exception as e:                                   # malformed request: say why, never crash
        return JSONResponse(status_code=400, content={"error": f"{type(e).__name__}: {e}"})
    out = simulate(init, sc, steps, pert)
    return {"rollout": np.asarray(out, dtype="f4").tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
