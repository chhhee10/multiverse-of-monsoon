"""Local grader: the organisers' own scoring functions (team kit, monsoon_r1.scoring) on held-back windows.

Any model is a callable  simulate(init[3,6,32,36], doy_sincos, steps, perturbation|None) -> [steps,6,32,36]
in physical units, the same signature serve.py exposes. Training-only; the server never imports this.

  python -m training.evaluate --model persistence --year 2021
  python -m training.evaluate --model damped --year 2022
  python -m training.evaluate --model engine --year 2022            # assets/ weights through monsoon.engine
  python -m training.evaluate --model http://localhost:8000 --year 2022
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Callable

import numpy as np
from monsoon_r1 import scoring as S
from monsoon_r1.contract import BOX_ARABIAN_SEA, BOX_BAY_OF_BENGAL, LAT, LON

from training.dataset import Archive, climatology_2005_2021, doy_index, doy_sincos, load_all, segments, stats

LEADS = (0, 3, 7)          # days 1, 4, 8
STEPS = 10
SimFn = Callable[[np.ndarray, list, int, np.ndarray | None], np.ndarray]

LAND_NORTH = (LAT >= 28)[:, None] & (LON <= 80)[None, :]
SW_COAST = ((LAT >= 8) & (LAT <= 16))[:, None] & ((LON >= 73) & (LON <= 78))[None, :]
BAY = ((LAT >= 10) & (LAT <= 22))[:, None] & ((LON >= 82) & (LON <= 95))[None, :]


def _valid_starts(arch: Archive) -> np.ndarray:
    ok = np.zeros(len(arch.dates), bool)
    for s, e in segments(arch.dates):
        ok[s + 3:max(s + 3, e - STEPS + 1)] = True
    return ok


def event_windows(arch: Archive, year: int) -> dict[str, int]:
    """One window per hidden-test event type, chosen from the data of `year` (index = first forecast day)."""
    ok = _valid_starts(arch)
    d = arch.dates
    month = (d.astype("datetime64[M]").astype(int) % 12) + 1
    yr = d.astype("datetime64[Y]").astype(int) + 1970
    x = arch.data
    land = arch.static[0] > 0.5

    def pick(months, score):
        m = ok & (yr == year) & np.isin(month, months)
        idx = np.where(m)[0]
        return int(idx[np.argmax(score[idx])]) - 1 if len(idx) else None   # event peaks on forecast day ~2

    out = {
        "monsoon_onset": pick([5, 6], x[:, 1][:, SW_COAST].mean(axis=1)),
        "heavy_rain": pick([7, 8], (x[:, 1] > 10).reshape(len(x), -1).mean(axis=1)),
        "heatwave": pick([4, 5, 6], x[:, 0][:, land].mean(axis=1)),
        "cyclone": pick([4, 5, 10, 11, 12], -x[:, 4][:, BAY].min(axis=1)),
        "winter_disturbance": pick([1, 2], x[:, 1][:, LAND_NORTH].mean(axis=1)),
    }
    return {k: v for k, v in out.items() if v is not None and ok[v]}


def weekly_windows(arch: Archive, year: int, every: int = 7) -> list[int]:
    ok = _valid_starts(arch)
    yr = arch.dates.astype("datetime64[Y]").astype(int) + 1970
    return [int(i) for i in np.where(ok & (yr == year))[0][::every]]


def score_window(sim: SimFn, arch: Archive, clim: np.ndarray, t: int) -> dict:
    init = arch.data[t - 3:t]; truth = arch.data[t:t + STEPS]
    pred = np.asarray(sim(init, doy_sincos(arch.dates[t - 1]), STEPS, None), "f4")
    bad = S.validity(pred, STEPS)
    if bad:
        return {"valid": False, "why": bad}
    L = list(LEADS)
    p, tr = pred[L], truth[L]
    per = np.repeat(init[-1:], len(L), axis=0)
    cl = np.stack([clim[doy_index(arch.dates[t + k])] for k in L])
    sk = S.skill_vs_best_ref(p, tr, per, cl)
    csi_m, csi_p, csi_c = S.csi(p[:, 1], tr[:, 1]), S.csi(per[:, 1], tr[:, 1]), S.csi(cl[:, 1], tr[:, 1])
    d4 = S.skill_vs_best_ref(pred[3:4], truth[3:4], init[-1:], clim[doy_index(arch.dates[t + 3])][None])
    return {"valid": True, "ss": float(np.nanmean(sk["ss"])), "ss_ch": sk["ss"].tolist(),
            "csi": csi_m, "csi_skill": S.csi_skill(csi_m, csi_p, csi_c),
            "beats_persist_d4": float(np.nanmean(d4["ss_persist"])),
            "blur": S.variance_ratio(pred[7], truth[7]).tolist()}


def shock_tests(sim: SimFn, arch: Archive, t: int, std: np.ndarray) -> dict:
    init = arch.data[t - 3:t]; sc = doy_sincos(arch.dates[t - 1])
    base = np.asarray(sim(init, sc, STEPS, None), "f4")
    null = np.asarray(sim(init, sc, STEPS, np.zeros((6, 32, 36), "f4")), "f4")
    res = {"null_ok": S.null_consistent(base, null, std)}
    for name, box in (("warm_ocean", BOX_BAY_OF_BENGAL), ("monsoon_stall", BOX_ARABIAN_SEA)):
        d = S.make_shock(name, init[-1], std)
        res[name] = S.causality_checks(base, np.asarray(sim(init, sc, STEPS, d), "f4"), box, std)
    res["points"] = S.causality_points([res["warm_ocean"], res["monsoon_stall"]], res["null_ok"])
    return res


def grade(sim: SimFn, arch: Archive, year: int, clim: np.ndarray | None = None, weekly: bool = True) -> dict:
    clim = climatology_2005_2021() if clim is None else clim
    std = np.array(stats()["std"], "f4")
    ev = event_windows(arch, year)
    ev_res = {k: score_window(sim, arch, clim, t) for k, t in ev.items()}
    out = {"year": year, "event_dates": {k: str(arch.dates[t]) for k, t in ev.items()}, "events": ev_res}

    def summarise(rs):
        rs = [r for r in rs if r["valid"]]
        if not rs:
            return {"valid_frac": 0.0}
        fp = S.forecast_points([r["ss"] for r in rs], [r["csi_skill"] for r in rs])
        return {"mean_ss": fp["mean_ss"], "mean_csi_skill": fp["mean_csi_skill"], "rmse_pts": fp["rmse_pts"],
                "csi_pts": fp["csi_pts"], "blur_pts": S.anti_blur_points(np.array([r["blur"] for r in rs])),
                "blur_ratio": np.nanmean(np.array([r["blur"] for r in rs]), axis=0).round(2).tolist(),
                "ss_by_channel": np.nanmean(np.array([r["ss_ch"] for r in rs]), axis=0).round(3).tolist(),
                "beats_persist_d4": float(np.mean([r["beats_persist_d4"] for r in rs]))}

    out["events_summary"] = summarise(list(ev_res.values()))
    if weekly:
        out["weekly_summary"] = summarise([score_window(sim, arch, clim, t) for t in weekly_windows(arch, year)])
    shock_ts = [t for k, t in ev.items() if k in ("monsoon_onset", "heavy_rain", "heatwave")] or list(ev.values())[:1]
    shocks = [shock_tests(sim, arch, t, std) for t in shock_ts]
    out["shocks"] = [{"date": str(arch.dates[t]), "points": s["points"], "null_ok": s["null_ok"],
                      **{n: {k: s[n][k] for k in ("nonzero", "localized", "propagates", "stable", "near_frac_day1", "far_frac_max")}
                         for n in ("warm_ocean", "monsoon_stall")}} for t, s in zip(shock_ts, shocks)]
    out["causality_pts_mean"] = float(np.mean([s["points"] for s in shocks]))
    e = out["events_summary"]
    out["auto_pts_est"] = round(e.get("rmse_pts", 0) + e.get("csi_pts", 0) + e.get("blur_pts", 0) + out["causality_pts_mean"], 1)
    return out


# ---- baselines ---------------------------------------------------------------------------------------------
def _doy_from_sincos(sc) -> int:
    return int(round((math.atan2(sc[0], sc[1]) * 365.0 / (2 * math.pi)) % 365))


def make_baseline(kind: str, clim: np.ndarray) -> SimFn:
    def doy_slot(sc, k):                    # approximate slot for forecast day k (calendar years are never sent)
        return (_doy_from_sincos(sc) - 1 + k) % 365

    def sim(init, sc, steps, pert):
        last = init[-1] + (0 if pert is None else pert)
        if kind == "persistence":
            return np.repeat(last[None], steps, axis=0)
        if kind == "climatology":
            return np.stack([clim[doy_slot(sc, k + 1)] for k in range(steps)])
        if kind == "damped":                 # anomaly persistence that decays to climatology
            a = last - clim[doy_slot(sc, 0)]
            out = np.stack([clim[doy_slot(sc, k + 1)] + a * 0.8 ** (k + 1) for k in range(steps)])
            out[:, 1] = np.maximum(out[:, 1], 0)
            return out
        raise ValueError(kind)
    return sim


def http_model(url: str) -> SimFn:
    """A running server, called through the kit's own client (same 5 s deadline as the judge)."""
    from monsoon_r1.client import call_simulate

    def sim(init, sc, steps, pert):
        body = {"init_states": init.tolist(), "doy_sincos": sc, "steps": steps,
                "perturbation": None if pert is None else np.asarray(pert, "f4").tolist()}
        r = call_simulate(url, body, allow_private=True)
        if not r.ok:
            raise RuntimeError(f"server call failed: {r.error}")
        return r.out
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="persistence")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--no-weekly", action="store_true")
    a = ap.parse_args()
    arch = load_all(); clim = climatology_2005_2021()
    if a.model.startswith("http"):
        sim = http_model(a.model)
    elif a.model == "engine":
        from monsoon.engine import Engine
        sim = Engine.load().simulate
    else:
        sim = make_baseline(a.model, clim)
    res = grade(sim, arch, a.year, clim, weekly=not a.no_weekly)
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=1, default=float))


if __name__ == "__main__":
    main()
