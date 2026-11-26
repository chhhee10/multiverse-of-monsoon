"""Train the twin on a GPU (Modal) or on CPU for a smoke test.

Curriculum:
  phase K=1      one-day-ahead training (teacher forcing)
  phases K=2..8  autoregressive fine-tuning: the model is fed its own predictions for K days and the loss covers
                 every day, so it learns to stay stable and sharp over a long rollout (anti-drift, anti-blur)
Loss: latitude-weighted MSE in model space (rain in log1p space) + a heavy-rain (>10 mm/day) BCE head.
Model selection: the organisers' own scoring functions on held-out years (training.evaluate maths).

  python train.py --out runs/smoke --phases 1:200:16 --eval-every 100 --max-val 24       # CPU smoke test
  python train.py --out runs/v1 --phases 1:20000:64,2:3000:48,4:3000:24,8:2000:12
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monsoon_r1 import scoring as S

from monsoon.features import NC, Normalizer, lat_weights, static_features, step
from monsoon.model import TwinNet
from training.dataset import ROOT, doy_index, load_all, segments

LEADS = (0, 3, 7)


def day_of_year(dates: np.ndarray) -> np.ndarray:
    """1..365, with Dec 31 of leap years folded onto 1 exactly as the contract's sin/cos does."""
    doy = (dates - dates.astype("datetime64[Y]")).astype(int) + 1
    return ((doy - 1) % 365) + 1


def valid_last_days(ok: np.ndarray, seg_id: np.ndarray, k: int) -> np.ndarray:
    """Indices t (last input day) such that days t-2 .. t+k are all `ok` and in one consecutive segment."""
    c = np.r_[0, np.cumsum(ok.astype(int))]
    t = np.arange(2, len(ok) - k)
    good = (c[t + k + 1] - c[t - 2] == k + 3) & (seg_id[t - 2] == seg_id[t + k])
    return t[good]


def parse_phases(s: str) -> list[tuple[int, int, int]]:
    return [tuple(int(v) for v in p.split(":")) for p in s.split(",") if p]


class Trainer:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        self.rng = np.random.default_rng(a.seed)

        assets = ROOT / "assets"
        self.norm = Normalizer.from_dict(json.loads((assets / "norm.json").read_text())).to(self.dev)
        self.clim_phys = np.load(assets / "climatology.npz")["clim"].astype("f4")

        arch = load_all()
        if a.first_year:
            arch = arch.years(a.first_year, 2100)
        assert np.isfinite(arch.data).all(), "non-finite values in the training data"
        self.dates = arch.dates
        self.x_phys = arch.data                                           # numpy, for evaluation
        years = arch.dates.astype("datetime64[Y]").astype(int) + 1970
        val_years = [int(y) for y in a.val_years.split(",") if y]
        seg_id = np.zeros(len(arch.dates), int)
        for i, (s, e) in enumerate(segments(arch.dates)):
            seg_id[s:e] = i
        self.seg_id = seg_id
        self.train_ok = ~np.isin(years, val_years)
        self.val_ok = np.isin(years, val_years)
        print(f"days={len(arch.dates)} {arch.dates[0]}..{arch.dates[-1]} train_days={self.train_ok.sum()} "
              f"val_days={self.val_ok.sum()} device={self.dev}", flush=True)

        with torch.no_grad():
            self.xz = self.norm.fwd(torch.as_tensor(arch.data).to(self.dev))          # [N,6,H,W]
            self.heavy = torch.as_tensor(arch.data[:, 1] > 10.0).to(self.dev)         # [N,H,W]
            self.clim_z = self.norm.fwd(torch.as_tensor(self.clim_phys).to(self.dev))
        self.doy = torch.as_tensor(day_of_year(arch.dates)).to(self.dev)
        self.static_f = static_features(arch.static).to(self.dev)
        self.wlat = torch.as_tensor(lat_weights()).to(self.dev).view(1, 1, -1, 1)
        self.ref_slot = np.array([doy_index(d) for d in arch.dates])                # kit calendar slots (grader refs)

        self.net = TwinNet(dim=a.dim, depth=a.depth, kernel=a.kernel, mult=a.mult).to(self.dev)
        print(f"params={sum(p.numel() for p in self.net.parameters()) / 1e6:.2f}M radius={self.net.radius} cells/day",
              flush=True)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=a.lr, weight_decay=a.wd)
        self.out = Path(a.out); self.out.mkdir(parents=True, exist_ok=True)
        self.best = -1e9
        self.it = 0
        self.t0 = time.time()

        vt = valid_last_days(self.val_ok, seg_id, 10)
        self.val_t = vt[::a.val_stride][: a.max_val] if a.max_val else vt[::a.val_stride]

    # ---- one rollout --------------------------------------------------------------------------------------
    def rollout(self, t_last: torch.Tensor, k: int, train: bool = True):
        idx = t_last[:, None] + torch.arange(-2, 1, device=self.dev)[None]
        hist = self.xz[idx]                                                  # [B,3,6,H,W]
        doy = self.doy[t_last]
        states, logits = [], []
        for j in range(k):
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.a.bf16 and self.dev.type == "cuda"):
                z, logit, doy = step(self.net, hist, doy, self.clim_z, self.static_f)
            z, logit = z.float(), logit.float()
            states.append(z); logits.append(logit)
            hist = torch.cat([hist[:, 1:], z.unsqueeze(1)], dim=1)
        return torch.stack(states, 1), torch.stack(logits, 1)                # [B,k,6,H,W], [B,k,H,W]

    def loss(self, t_last: torch.Tensor, k: int):
        pred, logit = self.rollout(t_last, k)
        tgt_idx = t_last[:, None] + torch.arange(1, k + 1, device=self.dev)[None]
        tgt = self.xz[tgt_idx]
        mse = ((pred - tgt) ** 2 * self.wlat).mean()
        heavy = self.heavy[tgt_idx].float()
        bce = F.binary_cross_entropy_with_logits(logit, heavy, pos_weight=torch.tensor(self.a.pos_weight, device=self.dev),
                                                 reduction="none")
        bce = (bce * self.wlat[:, 0]).mean()
        return mse + self.a.rain_w * bce, mse.item(), bce.item()

    # ---- evaluation with the organisers' scoring maths ---------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.net.eval()
        preds, logits = [], []
        for i in range(0, len(self.val_t), 64):
            tl = torch.as_tensor(self.val_t[i:i + 64] - 1, device=self.dev)    # val_t = first forecast day
            z, lg = self.rollout(tl, 10, train=False)
            preds.append(self.norm.inv(z).cpu().numpy()); logits.append(lg.cpu().numpy())
        self.net.train()
        pred = np.concatenate(preds); L = list(LEADS)
        ss, cs, blur = [], [], []
        for w, t in enumerate(self.val_t):
            p = pred[w]; truth = self.x_phys[t:t + 10]
            if S.validity(np.clip(p, None, None), 10):
                ss.append(-1.0); cs.append(-1.0); blur.append([0.0] * NC); continue
            per = np.repeat(self.x_phys[t - 1][None], len(L), 0)
            cl = self.clim_phys[self.ref_slot[t + np.array(L)]]
            sk = S.skill_vs_best_ref(p[L], truth[L], per, cl)
            ss.append(float(np.nanmean(sk["ss"])))
            cs.append(S.csi_skill(S.csi(p[L, 1], truth[L, 1]), S.csi(per[:, 1], truth[L, 1]), S.csi(cl[:, 1], truth[L, 1])))
            blur.append(S.variance_ratio(p[7], truth[7]).tolist())
        fp = S.forecast_points(ss, cs)
        blur = np.array(blur)
        res = {"it": self.it, "mean_ss": fp["mean_ss"], "mean_csi_skill": fp["mean_csi_skill"],
               "rmse_pts": fp["rmse_pts"], "csi_pts": fp["csi_pts"], "blur_pts": S.anti_blur_points(blur),
               "blur_ratio": np.nanmean(blur, 0).round(2).tolist(), "windows": len(self.val_t),
               "minutes": round((time.time() - self.t0) / 60, 1)}
        # raw skill matters beyond the point cap (hidden windows are extreme events), so select on it
        res["select"] = res["mean_ss"] + 0.5 * (res["mean_csi_skill"] if np.isfinite(res["mean_csi_skill"]) else 0)
        return res

    def save(self, name: str):
        torch.save({"config": self.net.config, "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()}},
                   self.out / name)

    def log(self, rec: dict):
        with open(self.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    # ---- main loop -------------------------------------------------------------------------------------------
    def fit(self, commit=None):
        phases = parse_phases(self.a.phases)
        deadline = self.t0 + self.a.time_limit * 60
        for pi, (k, iters, batch) in enumerate(phases):
            pool = valid_last_days(self.train_ok, self.seg_id, k)
            lr0 = self.a.lr if pi == 0 else self.a.lr_ft
            for g in self.opt.param_groups:
                g["lr"] = lr0
            run = dict(mse=0.0, bce=0.0, n=0)
            for i in range(iters):
                warm = min(1.0, (i + 1) / 300) if pi == 0 else 1.0
                lr = lr0 * warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * i / iters)))
                for g in self.opt.param_groups:
                    g["lr"] = lr
                t_last = torch.as_tensor(self.rng.choice(pool, batch), device=self.dev)
                loss, mse, bce = self.loss(t_last, k)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                self.it += 1
                run["mse"] += mse; run["bce"] += bce; run["n"] += 1
                last = i == iters - 1
                if self.it % self.a.eval_every == 0 or last or time.time() > deadline:
                    rec = {"phase": f"K{k}", "lr": round(lr, 6), "train_mse": run["mse"] / run["n"],
                           "train_bce": run["bce"] / run["n"], **self.evaluate()}
                    run = dict(mse=0.0, bce=0.0, n=0)
                    self.save("last.pt")
                    # only rollout-trained checkpoints can be "best" once rollout phases exist
                    if rec["select"] > self.best and (k > 1 or len(phases) == 1 or pi == len(phases) - 1):
                        self.best = rec["select"]; self.save("model.pt"); rec["best"] = True
                    self.log(rec)
                    if commit:
                        commit()
                if time.time() > deadline:
                    print("time limit reached", flush=True)
                    break
            if time.time() > deadline:
                break
        if not (self.out / "model.pt").exists():
            self.save("model.pt")
        (self.out / "config.json").write_text(json.dumps(vars(self.a), indent=1))
        if commit:
            commit()


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/dev")
    ap.add_argument("--phases", default="1:20000:64,2:3000:48,4:3000:24,8:2000:12", help="K:iters:batch,...")
    ap.add_argument("--dim", type=int, default=160)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--mult", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--rain-w", type=float, default=0.1)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--first-year", type=int, default=0, help="drop years before this (0 = use everything)")
    ap.add_argument("--val-years", default="2021,2022")
    ap.add_argument("--val-stride", type=int, default=3)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--time-limit", type=float, default=600, help="minutes")
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None, commit=None):
    a = parser().parse_args(argv)
    Trainer(a).fit(commit)


if __name__ == "__main__":
    main()
