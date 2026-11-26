"""Build the static model assets the server ships with (assets/).

  climatology.npz  [366,6,32,36] smoothed daily climatology of 2005-2021 (one calendar year, not an archive)
  static.npz       [2,32,36] land_sea_mask, orography_m
  norm.json        channel mean/std (kit stats.json) + log1p-rain mean/std

  python -m training.build_assets
"""
from __future__ import annotations

import json

import numpy as np

from training.dataset import ROOT, climatology_2005_2021, load_kit, stats


def main():
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    kit = load_kit()
    clim = climatology_2005_2021().astype("f4")
    np.savez_compressed(out / "climatology.npz", clim=clim)
    np.savez_compressed(out / "static.npz", static=kit.static.astype("f4"))
    st = stats()
    ltp = np.log1p(np.maximum(kit.data[:, 1], 0))
    norm = {"channels": st["channels"], "mean": st["mean"], "std": st["std"],
            "tp_log_mean": float(ltp.mean()), "tp_log_std": float(ltp.std())}
    (out / "norm.json").write_text(json.dumps(norm, indent=1))
    print("clim", clim.shape, "norm", {k: norm[k] for k in ("tp_log_mean", "tp_log_std")})
    for p in sorted(out.iterdir()):
        print(f"{p.name:18s} {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
