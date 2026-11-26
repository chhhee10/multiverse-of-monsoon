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


class Ensemble(torch.nn.Module):
    """Several independently trained twins stepped together: every day their predictions are averaged and the
    average is fed back, so the rollout stays one deterministic trajectory (no sampling)."""

    def __init__(self, nets: list[TwinNet]):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.config = nets[0].config

    def forward(self, x):
        outs = [n(x) for n in self.nets]
        return torch.stack([o[0] for o in outs]).mean(0), torch.stack([o[1] for o in outs]).mean(0)


def _load_net(path: Path) -> TwinNet:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = TwinNet(**ck["config"])
    net.load_state_dict(ck["state_dict"])
    return net.eval()


class Engine:
    def __init__(self, net: torch.nn.Module, norm: Normalizer, clim: np.ndarray, static: np.ndarray,
                 post: dict | None = None):
        self.net = net.eval()
        self.norm = norm
        self.clim_z = norm.fwd(torch.as_tensor(clim))                 # [366,6,H,W] model space
        self.static_f = static_features(static)
        self.post = post or {}
        n = len(net.nets) if isinstance(net, Ensemble) else 1
        self.name = f"twinnet-d{net.config['dim']}x{net.config['depth']}" + (f"-ens{n}" if n > 1 else "")

    @classmethod
    def load(cls, assets: Path = ASSETS, model_path=None) -> "Engine":
        """model_path: one checkpoint, a list of checkpoints (ensemble), or None for assets/ (an ensemble when
        assets/ensemble.json lists several checkpoints, else assets/model.pt)."""
        if model_path is None:
            ens = assets / "ensemble.json"
            model_path = [assets / p for p in json.loads(ens.read_text())] if ens.exists() else assets / "model.pt"
        paths = model_path if isinstance(model_path, (list, tuple)) else [model_path]
        nets = [_load_net(Path(p)) for p in paths]
        net = nets[0] if len(nets) == 1 else Ensemble(nets)
        norm = Normalizer.from_dict(json.loads((assets / "norm.json").read_text()))
        clim = np.load(assets / "climatology.npz")["clim"].astype("f4")
        static = np.load(assets / "static.npz")["static"].astype("f4")
        post_path = assets / "post.json"
        post = json.loads(post_path.read_text()) if post_path.exists() else {}
        return cls(net, norm, clim, static, post)

    def _bounds(self, x: np.ndarray, lo, hi) -> np.ndarray:
        return np.clip(x, lo[:, None, None], hi[:, None, None])

    def simulate(self, init_states: np.ndarray, doy_sincos, steps: int, perturbation=None) -> np.ndarray:
        """init_states [3,6,32,36] physical, oldest first; perturbation [6,32,36] delta added to the last day.
        Returns [steps,6,32,36] physical units."""
        raw, prob = self.rollout_raw(init_states, doy_sincos, steps, perturbation)
        out = np.empty_like(raw)
        for k in range(steps):
            out[k] = self._bounds(self._heavy_rain(raw[k].copy(), prob[k], k), OUT_LO, OUT_HI)
        return out

    @torch.no_grad()
    def rollout_raw(self, init_states: np.ndarray, doy_sincos, steps: int, perturbation=None):
        """The model's own rollout before post-processing: (states [steps,6,H,W] physical, P(rain>10) [steps,H,W])."""
        x = np.array(init_states, dtype="f4", copy=True).reshape(3, NC, H, W)
        if perturbation is not None:
            x[-1] = x[-1] + np.asarray(perturbation, dtype="f4").reshape(NC, H, W)
        if not np.isfinite(x).all():                                 # never let a bad value poison the rollout
            fill = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=(0, 2, 3))
            x = np.where(np.isfinite(x), x, np.nan_to_num(fill)[None, :, None, None])
        x = self._bounds(x, IN_LO, IN_HI)
        hist = self.norm.fwd(torch.as_tensor(x)).unsqueeze(0)       # [1,3,6,H,W]
        doy = torch.tensor([doy_from_sincos(float(doy_sincos[0]), float(doy_sincos[1]))])
        raw = np.empty((steps, NC, H, W), "f4")
        prob = np.empty((steps, H, W), "f4")
        for k in range(steps):
            z, logit, doy = step(self.net, hist, doy, self.clim_z, self.static_f)
            raw[k] = self._bounds(self.norm.inv(z)[0].numpy(), OUT_LO, OUT_HI)
            prob[k] = torch.sigmoid(logit[0]).numpy()
            hist = torch.cat([hist[:, 1:], z.unsqueeze(1)], dim=1)   # feed back the model state, not the post-processing
        return raw, prob

    def _heavy_rain(self, phys: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
        """Calibrated heavy-rain flagging (assets/post.json): where the heavy-rain head is confident, the reported
        rain is raised to at least `floor` mm/day. Affects only the reported field, never the model's own state."""
        cfg = self.post.get("heavy_rain")
        if not cfg:
            return phys
        thr = cfg["p_thr"][min(k, len(cfg["p_thr"]) - 1)] if isinstance(cfg["p_thr"], list) else cfg["p_thr"]
        phys[1] = np.where(p > thr, np.maximum(phys[1], cfg.get("floor", 10.5)), phys[1])
        return phys
