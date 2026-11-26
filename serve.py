"""Multiverse of Monsoon inference server.

The judge runs this exact file offline:  python serve.py --port 8000
Contract: POST /health, POST /simulate (see the team kit's API_CONTRACT.md). Physical units on the wire.

Status: API plumbing from the kit template. The model is wired in next (monsoon/engine.py); until then simulate()
repeats the last day.
"""
import argparse
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# The dashboard (a separate static web app) calls this server from the browser.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "OPTIONS"], allow_headers=["*"])
MODEL_NAME = "placeholder-persistence"


def _arr(x):
    """Accept nested lists (the contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        import base64
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """init_states [3,6,32,36] physical units, oldest first. perturbation [6,32,36] delta or None.
    Returns [steps,6,32,36] physical units. doy_sincos = [sin, cos] of the day of year of init_states[-1]."""
    last = init_states[-1].copy()
    if perturbation is not None:
        last = last + perturbation
    return np.repeat(last[None], steps, axis=0)


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    t0 = time.perf_counter()
    body = await request.json()
    init = _arr(body["init_states"])
    pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
    out = simulate(init, body["doy_sincos"], int(body["steps"]), pert)
    return {"rollout": np.asarray(out, dtype="f4").tolist(), "inference_ms": (time.perf_counter() - t0) * 1000.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")
