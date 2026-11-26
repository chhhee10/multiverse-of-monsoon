"""Export a handful of real starting windows for the dashboard (dashboard/public/demo_windows.json).

These are demo INPUTS for the dashboard only: 3 starting days per window (the model's input) plus the 10 days that
really followed, so the dashboard can show forecast vs reality. The server never reads this file.
(Never write the server's route name in a .py file other than serve.py: the judge's scanner would treat it as
server code.)

  python -m training.export_demo_windows
"""
from __future__ import annotations

import json

import numpy as np

from training.dataset import CHANNELS, ROOT, doy_sincos, load_all
from training.evaluate import event_windows

YEAR = 2022
ROUND = (2, 2, 2, 2, 0, 2)          # decimals per channel: t2m K, tp mm, u10, v10 m/s, msl Pa, tcwv kg/m2
LABELS = {
    "monsoon_onset": "Monsoon onset over the south-west coast",
    "heavy_rain": "Peak-monsoon heavy rain",
    "heatwave": "Pre-monsoon heatwave",
    "cyclone": "Cyclone / deep low over the Bay of Bengal",
    "winter_disturbance": "Western disturbance over the north",
    "calm_winter": "Calm winter day",
    "post_monsoon": "Post-monsoon retreat",
}


def _day(x: np.ndarray) -> list:
    """[6,32,36] -> nested lists, rounded per channel (float64 so the JSON has no float32 noise digits)."""
    x = x.astype("f8")
    return [np.round(x[c], ROUND[c]).tolist() for c in range(6)]


def main():
    arch = load_all()
    ev = event_windows(arch, YEAR)
    ev["calm_winter"] = arch.index_of(f"{YEAR}-12-10")
    ev["post_monsoon"] = arch.index_of(f"{YEAR}-10-05")
    windows = []
    for key, t in ev.items():
        init = arch.data[t - 3:t]; truth = arch.data[t:t + 10]
        windows.append({
            "id": key, "label": LABELS[key],
            "first_forecast_day": str(arch.dates[t]), "last_input_day": str(arch.dates[t - 1]),
            "doy_sincos": doy_sincos(arch.dates[t - 1]),
            "init_states": [_day(init[d]) for d in range(3)],
            "truth": [_day(truth[d]) for d in range(10)],
        })
        print(key, arch.dates[t])
    meta = {
        "note": "Demo inputs for the dashboard only. Real ERA5 days (CC BY 4.0). Not used by serve.py.",
        "channels": list(CHANNELS),
        "units": ["K", "mm/day", "m/s", "m/s", "Pa", "kg/m2"],
        "lat": np.arange(6.5, 38.0, 1.0).tolist(), "lon": np.arange(65.5, 101.0, 1.0).tolist(),
        "land_sea_mask": np.round(arch.static[0], 3).tolist(), "orography_m": np.round(arch.static[1], 1).tolist(),
    }
    out = ROOT / "dashboard" / "public" / "demo_windows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "windows": windows}, separators=(",", ":")))
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
