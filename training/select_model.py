"""Pick the submission model by EXPECTED automated points under the judge's real setup.

The judge averages forecast skill over only 5 hidden windows and shocks a single window, so a model's validation
mean is not enough: we resample our per-window results into many random 5-window draws (bootstrap) and report the
expected forecast points, the chance of dropping below full marks, and the expected shock points.

  python -m training.calibrate --model runs/X/model.pt --write 0 --dump runs/select/X.cal.json
  python -m training.shock_report --model runs/X/model.pt --post runs/select/X.cal.json --verbose 0 --dump runs/select/X.shock.json
  python -m training.select_model X Y Z
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from monsoon_r1 import scoring as S

DIR = Path(__file__).resolve().parents[1] / "runs" / "select"


def main(names: list[str], draws: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    print(f"{'candidate':24s} {'E[forecast]':>11s} {'P(<45)':>7s} {'5% worst':>8s} {'E[shock]':>8s} {'E[total/60]':>11s}")
    for n in names:
        cal = json.loads((DIR / f"{n}.cal.json").read_text())
        shock = np.array(json.loads((DIR / f"{n}.shock.json").read_text())["pts"])
        ss, cs = np.array(cal["ss"]), np.array(cal["cs"], dtype=float)
        idx = rng.integers(0, len(ss), size=(draws, 5))
        fp = np.array([S.forecast_points(ss[i].tolist(), cs[i].tolist())["points"] for i in idx])
        e_total = fp.mean() + shock.mean()
        print(f"{n:24s} {fp.mean():11.2f} {np.mean(fp < 45 - 1e-6):7.1%} {np.percentile(fp, 5):8.2f} "
              f"{shock.mean():8.2f} {e_total:11.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
