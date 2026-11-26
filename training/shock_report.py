"""Shock (multiverse) test on many real windows, with the organisers' causality checks, plus a diagnosis of failures.

The judge applies its two shocks to ONE unknown real window, so we estimate the expected causality points over many
windows across the year instead of trusting a single case.

  python -m training.shock_report --years 2021,2022 --every 15
  python -m training.shock_report --model runs/v1big/model.pt
"""
from __future__ import annotations

import argparse

import numpy as np
from monsoon_r1 import scoring as S
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL

from monsoon.engine import Engine
from training.dataset import doy_sincos, load_all, stats
from training.evaluate import _valid_starts

CH = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
BOXES = {"warm_ocean": BOX_BAY_OF_BENGAL, "monsoon_stall": BOX_ARABIAN_SEA}


def diagnose(base, shock, box, std) -> str:
    d = shock - base
    z = np.abs(d / std[None, :, None, None])
    k, c, i, j = np.unravel_index(np.argmax(z), z.shape)
    e = ((d / std[None, :, None, None]) ** 2).sum(axis=1)
    near = S.near_mask(box)
    out_e = np.array([e[s][~near].sum() for s in range(len(e))])
    peak = e.sum(axis=(1, 2)).max()
    return (f"max |d|={z.max():.2f} std ({CH[c]} day{k + 1} cell {i},{j}); outside energy day1={out_e[0]:.3g} "
            f"days4-8={out_e[3:8].mean():.3g} (need >{1.5 * out_e[0]:.3g} and >{0.02 * peak:.3g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--years", default="2021,2022")
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--no-post", action="store_true", help="disable the heavy-rain flag (assets/post.json)")
    ap.add_argument("--post", default=None, help="heavy-rain config JSON to use instead of assets/post.json")
    ap.add_argument("--dump", default=None, help="write per-window causality points to this JSON")
    a = ap.parse_args()
    eng = Engine.load(model_path=a.model.split(",") if a.model else None)
    if a.no_post:
        eng.post = {}
    elif a.post:
        import json
        cfg = json.loads(open(a.post).read()).get("cfg")
        eng.post = {"heavy_rain": cfg} if cfg else {}
    arch = load_all()
    std = np.array(stats()["std"], "f4")
    ok = _valid_starts(arch)
    yr = arch.dates.astype("datetime64[Y]").astype(int) + 1970
    ts = np.where(ok & np.isin(yr, [int(y) for y in a.years.split(",")]))[0][::a.every]
    pts, fails = [], {"warm_ocean": {}, "monsoon_stall": {}}
    for t in ts:
        init = arch.data[t - 3:t]; sc = doy_sincos(arch.dates[t - 1])
        base = eng.simulate(init, sc, 10)
        null_ok = S.null_consistent(base, eng.simulate(init, sc, 10, np.zeros((6, 32, 36), "f4")), std)
        res = []
        for name, box in BOXES.items():
            shock = eng.simulate(init, sc, 10, S.make_shock(name, init[-1], std))
            r = S.causality_checks(base, shock, box, std)
            res.append(r)
            for chk in ("nonzero", "localized", "propagates", "stable"):
                if not r[chk]:
                    fails[name][chk] = fails[name].get(chk, 0) + 1
                    if a.verbose:
                        print(f"{arch.dates[t]} {name:13s} FAIL {chk:10s} {diagnose(base, shock, box, std)}")
        pts.append(S.causality_points(res, null_ok))
    pts = np.array(pts)
    if a.dump:
        import json
        open(a.dump, "w").write(json.dumps({"pts": pts.tolist()}))
    print(f"\n{len(ts)} windows: expected causality {pts.mean():.2f}/15  min {pts.min():.2f}  "
          f"full marks in {np.mean(pts >= 15 - 1e-6) * 100:.0f}% of windows")
    print("failures by check:", fails)


if __name__ == "__main__":
    main()
