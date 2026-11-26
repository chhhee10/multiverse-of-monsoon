"""Shared building blocks for the twin: grid, normalisation, input features and the one-day step.

Used by the server engine and by training, so both see exactly the same maths. No data access here.
"""
from __future__ import annotations

import math

import numpy as np
import torch

NC, H, W = 6, 32, 36
TP = 1                                     # rain channel; it is modelled in log1p space
LAT = np.arange(6.5, 37.5 + 1e-9, 1.0)
LON = np.arange(65.5, 100.5 + 1e-9, 1.0)


def doy_from_sincos(s: float, c: float) -> int:
    """Invert the contract's [sin, cos](2 pi doy / 365) to a day of year in 1..365."""
    d = int(round((math.atan2(s, c) % (2 * math.pi)) * 365.0 / (2 * math.pi))) % 365
    return 365 if d == 0 else d


def clim_slot(doy):
    """Day of year (1..365) -> climatology slot (0..364). Works on ints and integer tensors."""
    return (doy - 1) % 365


def next_doy(doy):
    return doy % 365 + 1


def lat_weights() -> np.ndarray:
    w = np.cos(np.deg2rad(LAT))
    return (w / w.mean()).astype("f4")


class Normalizer:
    """Physical <-> model space. Channels are standardised; rain is log1p'd first so heavy rain is not crushed."""

    def __init__(self, mean, std, tp_log_mean: float, tp_log_std: float):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(NC, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(NC, 1, 1)
        self.lm, self.ls = float(tp_log_mean), float(tp_log_std)

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(d["mean"], d["std"], d["tp_log_mean"], d["tp_log_std"])

    def to(self, device) -> "Normalizer":
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self

    def fwd(self, x: torch.Tensor) -> torch.Tensor:
        """[..., 6, H, W] physical -> model space."""
        z = (x - self.mean) / self.std
        tp = (torch.log1p(x[..., TP, :, :].clamp(min=0)) - self.lm) / self.ls
        return torch.cat([z[..., :TP, :, :], tp.unsqueeze(-3), z[..., TP + 1:, :, :]], dim=-3)

    def inv(self, z: torch.Tensor) -> torch.Tensor:
        x = z * self.std + self.mean
        tp = torch.expm1(z[..., TP, :, :] * self.ls + self.lm).clamp(min=0)
        return torch.cat([x[..., :TP, :, :], tp.unsqueeze(-3), x[..., TP + 1:, :, :]], dim=-3)


def static_features(static: np.ndarray) -> torch.Tensor:
    """[2, H, W] (land_sea_mask, orography_m) -> [4, H, W] model features (mask, orography, lat, lon)."""
    lsm = static[0]
    oro = static[1] / 1000.0
    la = np.repeat(((LAT - 22.0) / 9.0)[:, None], W, axis=1)
    lo = np.repeat(((LON - 83.0) / 10.0)[None, :], H, axis=0)
    return torch.as_tensor(np.stack([lsm, oro, la, lo]).astype("f4"))


N_FEATURES = 3 * NC + NC + NC + NC + 4 + 2       # history, anomaly, clim(next), clim change, static, doy(next)


def build_input(hist: torch.Tensor, clim_last: torch.Tensor, clim_next: torch.Tensor, static_f: torch.Tensor,
                doy_n: torch.Tensor) -> torch.Tensor:
    """hist [B,3,6,H,W] (model space, oldest first), clims [B,6,H,W], static_f [4,H,W], doy_n [B] -> [B,F,H,W]."""
    b = hist.shape[0]
    last = hist[:, -1]
    ang = 2 * math.pi * doy_n.to(hist.dtype) / 365.0
    sc = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[:, :, None, None].expand(b, 2, H, W)
    return torch.cat([hist.reshape(b, 3 * NC, H, W), last - clim_last, clim_next, clim_next - clim_last,
                      static_f.unsqueeze(0).expand(b, -1, -1, -1), sc], dim=1)


def step(net: torch.nn.Module, hist: torch.Tensor, doy_last: torch.Tensor, clim_z: torch.Tensor,
         static_f: torch.Tensor):
    """One day forward. hist [B,3,6,H,W] model space; doy_last [B] long (1..365); clim_z [366,6,H,W] model space.
    Returns (next state [B,6,H,W] model space, heavy-rain logit [B,H,W], doy of the new day)."""
    doy_n = next_doy(doy_last)
    cl_last, cl_next = clim_z[clim_slot(doy_last)], clim_z[clim_slot(doy_n)]
    delta, logit = net(build_input(hist, cl_last, cl_next, static_f, doy_n))
    return hist[:, -1] + delta, logit, doy_n
