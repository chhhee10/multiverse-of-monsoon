"""Load the daily India archive for training and evaluation (training-only; the server never imports this).

Sources, all on the same 32 x 36 contract grid:
  data/india_climate_train.nc   organisers' file, 2005-2021, plus the 2 static fields
  data/chunks/*.npz             extra years rebuilt with the organisers' build_dataset (see data/pull_era5_modal.py)
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MONSOON_DATA", ROOT / "data"))      # Modal mounts the data volume elsewhere
CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")


@dataclass
class Archive:
    data: np.ndarray      # [N, 6, 32, 36] float32, physical units, consecutive days
    dates: np.ndarray     # [N] datetime64[D]
    static: np.ndarray    # [2, 32, 36] land_sea_mask, orography_m

    def index_of(self, day) -> int:
        return int(np.searchsorted(self.dates, np.datetime64(day, "D")))

    def years(self, y0: int, y1: int) -> "Archive":
        m = (self.dates >= np.datetime64(f"{y0}-01-01")) & (self.dates <= np.datetime64(f"{y1}-12-31"))
        return Archive(self.data[m], self.dates[m], self.static)


def load_kit() -> Archive:
    import xarray as xr
    ds = xr.open_dataset(DATA / "india_climate_train.nc")
    return Archive(ds["data"].values.astype("f4"), ds["time"].values.astype("datetime64[D]"),
                   ds["static"].values.astype("f4"))


def load_chunks(pattern: str = "*.npz") -> tuple[np.ndarray, np.ndarray]:
    parts = sorted((DATA / "chunks").glob(pattern))
    if not parts:
        return np.zeros((0, 6, 32, 36), "f4"), np.zeros((0,), "datetime64[D]")
    xs, ds = [], []
    for p in parts:
        z = np.load(p)
        xs.append(z["data"].astype("f4"))
        ds.append(np.array([np.datetime64(f"{int(v) // 10000:04d}-{int(v) // 100 % 100:02d}-{int(v) % 100:02d}") for v in z["dates"]]))
    data, dates = np.concatenate(xs), np.concatenate(ds).astype("datetime64[D]")
    order = np.argsort(dates)
    data, dates = data[order], dates[order]
    data[:, 1] = np.maximum(data[:, 1], 0.0)          # ERA5 rain has tiny negatives; the kit clamps them too
    return data, dates


def load_all() -> Archive:
    """Kit file + every extra chunk, de-duplicated by date (kit wins), sorted."""
    kit = load_kit()
    x, d = load_chunks()
    keep = ~np.isin(d, kit.dates)
    data = np.concatenate([x[keep], kit.data]); dates = np.concatenate([d[keep], kit.dates])
    order = np.argsort(dates)
    return Archive(data[order], dates[order], kit.static)


def segments(dates: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of consecutive days (rollout windows must not cross a gap)."""
    gaps = np.where(np.diff(dates).astype(int) != 1)[0]
    starts = np.r_[0, gaps + 1]; ends = np.r_[gaps + 1, len(dates)]
    return list(zip(starts.tolist(), ends.tolist()))


def stats() -> dict:
    return json.loads((DATA / "stats.json").read_text())


def doy_index(day: np.datetime64) -> int:
    """0..365 calendar-day slot, same convention as the kit's climatology (Feb 29 has its own slot)."""
    d = dt.date.fromisoformat(str(day))
    return (dt.date(2000, d.month, d.day) - dt.date(2000, 1, 1)).days


def doy_sincos(day: np.datetime64) -> list[float]:
    """The contract's doy_sincos for a given day (day-of-year counted from 1)."""
    doy = dt.date.fromisoformat(str(day)).timetuple().tm_yday
    return [float(np.sin(2 * np.pi * doy / 365.0)), float(np.cos(2 * np.pi * doy / 365.0))]


def climatology_2005_2021() -> np.ndarray:
    """[366, 6, 32, 36] smoothed daily climatology from the organisers' years, built with the kit's own function."""
    from monsoon_r1.stats import climatology
    kit = load_kit()
    ints = np.array([int(str(d).replace("-", "")) for d in kit.dates])
    return climatology(kit.data, ints)
