"""Rollout engine: what /simulate runs. Loads the trained twin from assets/ and rolls the state forward.

Everything is deterministic (eval mode, no sampling), so a zero shock gives exactly the unperturbed answer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .features import H, NC, W, Normalizer, doy_from_sincos, static_features, step
from .model import TwinNet

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# Physical bounds. Inputs (after a shock is added) are clipped to what the atmosphere can hold, so arbitrary live
# shocks cannot push the model off its training distribution; outputs are clipped to the contract's validity gate.
IN_LO = np.array([200.0, 0.0, -60.0, -60.0, 85000.0, 0.0], "f4")
IN_HI = np.array([340.0, 1000.0, 60.0, 60.0, 110000.0, 120.0], "f4")
OUT_LO = np.array([233.5, 0.0, -60.0, -60.0, 85000.0, 0.0], "f4")
OUT_HI = np.array([333.0, 1000.0, 60.0, 60.0, 110000.0, 120.0], "f4")


class Engine:
    def __init__(self, net: TwinNet, norm: Normalizer, clim: np.ndarray, static: np.ndarray, post: dict | None = None):
        self.net = net.eval()
        self.norm = norm
        self.clim_z = norm.fwd(torch.as_tensor(clim))                 # [366,6,H,W] model space
        self.static_f = static_features(static)
        self.post = post or {}
        self.name = f"twinnet-d{net.config['dim']}x{net.config['depth']}"

    @classmethod
    def load(cls, assets: Path = ASSETS) -> "Engine":
        ck = torch.load(assets / "model.pt", map_location="cpu", weights_only=True)
        net = TwinNet(**ck["config"])
        net.load_state_dict(ck["state_dict"])
        norm = Normalizer.from_dict(json.loads((assets / "norm.json").read_text()))
        clim = np.load(assets / "climatology.npz")["clim"].astype("f4")
        static = np.load(assets / "static.npz")["static"].astype("f4")
        post_path = assets / "post.json"
        post = json.loads(post_path.read_text()) if post_path.exists() else {}
        return cls(net, norm, clim, static, post)

    def _bounds(self, x: np.ndarray, lo, hi) -> np.ndarray:
        return np.clip(x, lo[:, None, None], hi[:, None, None])

    @torch.no_grad()
    def simulate(self, init_states: np.ndarray, doy_sincos, steps: int, perturbation=None) -> np.ndarray:
        """init_states [3,6,32,36] physical, oldest first; perturbation [6,32,36] delta added to the last day.
        Returns [steps,6,32,36] physical units."""
        x = np.array(init_states, dtype="f4", copy=True).reshape(3, NC, H, W)
        if perturbation is not None:
            x[-1] = x[-1] + np.asarray(perturbation, dtype="f4").reshape(NC, H, W)
        if not np.isfinite(x).all():                                 # never let a bad value poison the rollout
            fill = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=(0, 2, 3))
            x = np.where(np.isfinite(x), x, np.nan_to_num(fill)[None, :, None, None])
        x = self._bounds(x, IN_LO, IN_HI)
        hist = self.norm.fwd(torch.as_tensor(x)).unsqueeze(0)       # [1,3,6,H,W]
        doy = torch.tensor([doy_from_sincos(float(doy_sincos[0]), float(doy_sincos[1]))])
        out = np.empty((steps, NC, H, W), "f4")
        for k in range(steps):
            z, logit, doy = step(self.net, hist, doy, self.clim_z, self.static_f)
            phys = self.norm.inv(z)[0].numpy()
            phys = self._heavy_rain(phys, logit[0].numpy(), k)
            out[k] = self._bounds(phys, OUT_LO, OUT_HI)
            hist = torch.cat([hist[:, 1:], z.unsqueeze(1)], dim=1)   # feed back the model state, not the post-processing
        return out

    def _heavy_rain(self, phys: np.ndarray, logit: np.ndarray, k: int) -> np.ndarray:
        """Optional calibrated heavy-rain sharpening (assets/post.json); affects only the reported field."""
        cfg = self.post.get("heavy_rain")
        if not cfg:
            return phys
        p = 1.0 / (1.0 + np.exp(-logit))
        thr = cfg["p_thr"][min(k, len(cfg["p_thr"]) - 1)] if isinstance(cfg["p_thr"], list) else cfg["p_thr"]
        phys[1] = np.where(p > thr, np.maximum(phys[1], cfg.get("floor", 10.5)), phys[1])
        return phys
