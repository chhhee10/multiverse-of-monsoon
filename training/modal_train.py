"""Run training on a Modal GPU. Data lives in the `monsoon-data` volume (chunks/ + the kit's .nc and stats.json);
each run writes runs/<name>/{model.pt,last.pt,metrics.jsonl,config.json} back to the same volume.

  modal run training/modal_train.py --name v1
  modal run training/modal_train.py --name v1 --args "--dim 192 --depth 8"
  modal run training/modal_train.py --sweep "a=--dim 128;b=--dim 192 --depth 8"   # parallel runs, one GPU each
  modal volume get monsoon-data runs/v1 runs/ --force                              # fetch results
"""
from __future__ import annotations

import shlex

import modal

KIT = "monsoon-r1-kit @ git+https://github.com/OfficialAbhinavSingh/yug-monsoon-kit@689c8798bbff7f318a1e74878a6985abc646080a"

app = modal.App("monsoon-train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "numpy", "xarray", "netCDF4", "scipy", "requests", KIT)
    .add_local_python_source("monsoon", "training")
    .add_local_dir("assets", "/root/assets")
)
vol = modal.Volume.from_name("monsoon-data")


@app.function(image=image, gpu="A10", volumes={"/vol": vol}, timeout=4 * 3600, cpu=4.0, memory=16384)
def train(name: str, args: list[str]) -> str:
    import os
    os.environ["MONSOON_DATA"] = "/vol"
    from training.train import main
    out = f"/vol/runs/{name}"
    main(args + ["--out", out], commit=vol.commit)
    vol.commit()
    return f"{name}: done -> runs/{name}"


@app.local_entrypoint()
def main(name: str = "", args: str = "", sweep: str = ""):
    if sweep:
        jobs = [(p.split("=", 1)[0].strip(), shlex.split(p.split("=", 1)[1])) for p in sweep.split(";") if p.strip()]
        calls = [train.spawn(n, a) for n, a in jobs]
        for c in calls:
            print(c.get(), flush=True)
    else:
        print(train.remote(name or "dev", shlex.split(args)))
