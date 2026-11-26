# Dashboard: task spec

The dashboard is what the judges see in the desk review (top 10 → top 2). They will **apply their own shock on the
spot**, look at the impact module, and ask about the model. Everything on screen must come from **live** calls to our
own `/simulate`.

## Hard rules (break these and we lose points or get flagged)

1. **JS/TS only in `dashboard/`.** Never create a `.py` file that contains the text `/simulate` anywhere in the repo
   (the judge's scanner would treat it as server code). No Streamlit, no Python Plotly/Dash.
2. **No pre-rendered results.** Every forecast map comes from a live `/simulate` response. Caching a response in memory
   during the session is fine; shipping saved model outputs is not.
3. The only data file the dashboard ships is `public/demo_windows.json` (real starting days). Don't add other
   weather archives.
4. Keep the server code (`serve.py`, `monsoon/`) untouched; ask Chetan if the API needs something new.

## Run the backend locally

```bash
python -m venv .venv && . .venv/bin/activate     # from the repo root
pip install -r requirements.txt
python serve.py --port 8000
```

Right now `serve.py` is a placeholder that repeats the last input day. The real model drops in later with
**exactly the same API**, so build against it now. CORS is already enabled.

## API

`POST http://localhost:8000/simulate` with JSON:

```json
{
  "init_states": "3 x 6 x 32 x 36 nested list (oldest day first)",
  "doy_sincos": [0.71, -0.69],
  "steps": 10,
  "perturbation": "6 x 32 x 36 nested list of deltas, or null"
}
```

Response: `{"rollout": steps x 6 x 32 x 36, "inference_ms": 140.2}` (about 1 MB of JSON, under 1 s).

- Channels, in this order: `t2m` (K), `tp` rain (mm/day), `u10` east wind (m/s), `v10` north wind (m/s),
  `msl` pressure (Pa), `tcwv` moisture (kg/m²). Show t2m in °C (K − 273.15) and pressure in hPa (Pa / 100).
- `rollout[k]` is day k+1 after the last input day.
- The perturbation is a physical delta **added to the last input day** (`init_states[2]`).
- `POST /health` returns `{"status": "ready", "model_name": ...}`.

## Grid

32 rows × 36 columns of 1° cells. **Row 0 is the south** (lat 6.5°N) and row 31 is the north (37.5°N), so flip
vertically when drawing. Columns go west → east, lon 65.5°E → 100.5°E. Cell centres are in `meta.lat` / `meta.lon`.
Each cell spans ±0.5°. `meta.land_sea_mask` (0 = sea, 1 = land) and `meta.orography_m` are included.

For the India outline use Natural Earth (public domain) admin-0 / admin-1 GeoJSON, clipped to the box
6°–38°N, 65°–101°E.

## `public/demo_windows.json`

```text
meta:    channels, units, lat[32], lon[36], land_sea_mask[32][36], orography_m[32][36]
windows: [{ id, label, first_forecast_day, last_input_day, doy_sincos,
            init_states: [3][6][32][36],   ← send this as init_states
            truth:       [10][6][32][36] }] ← what really happened, for "forecast vs reality"
```

Windows (all 2022): monsoon onset, heavy rain, heatwave, cyclone, western disturbance, calm winter, post-monsoon.

## Features, in priority order

### P0: core (must work by ~22:00)
- **Window picker** (the 7 demo windows) → run the normal forecast.
- **Map** of one channel for day 0 (last input day) to day 10, with a day slider and play button. Channel selector.
  Fixed colour scale per channel with a legend in real units (so frames don't flicker).
- **Forecast vs reality** side by side using `truth`.

### P0: the what-if ("multiverse") panel, the centrepiece of the demo
- **Presets**, exactly as the judges' tests:
  - *Warm Bay of Bengal*: +3 K on t2m and +15% of the current tcwv over lat 8–20°N, lon 82–94°E.
  - *Monsoon stall*: −40% of the current u10 and v10 over lat 8–20°N, lon 65–75°E.
- **Custom shock**: draw a box (or brush cells) on the map, pick a variable and a magnitude
  (absolute, e.g. +3 K, or % of the current value). Build the `[6][32][36]` delta (zeros elsewhere) and send it.
- Clamp magnitudes in the UI: temperature ±10 K, wind ±80%, moisture ±50%, rain change ≥ −100%, pressure ±2000 Pa.
- Show three maps per day: **normal | shocked | difference** (diverging colour scale centred on 0), with the same
  day slider. Animate the difference over days 1–10 so the ripple is visible. Outline the shocked box.
- A zero shock must look identical to normal (a good live sanity check to show judges).

### P1: impact module (judged by people, so this is where we stand out)
Computed in the browser from the live normal and shocked rollouts:
- **Kharif crop water stress**: 10-day rain total per region (normal vs shocked) and the change. Flag regions whose
  rain drops more than 25%.
- **Heat stress**: days with t2m above 40 °C over land, per region.
- **Wind power proxy**: mean of |wind|³ over coastal and land cells, normal vs shocked (%).
- Regions (simple lat/lon boxes are fine): Kerala, Konkan/Mumbai, Vidarbha, Gangetic plain, Odisha coast,
  Delhi NCR, Gujarat, Tamil Nadu.
- A table plus a small bar chart per metric. One sentence of plain-English summary, e.g. "A 40% weaker Arabian Sea
  wind cuts Konkan rain by 31% over 10 days".

### P2: nice to have
- Wind arrows overlay (from u10/v10).
- Time series of a clicked cell (normal vs shocked vs truth).
- An "uncertainty" view if we add an ensemble option to the API later.

## Suggested stack

Vite + React + TypeScript. Draw the grid on a `<canvas>` (36×32 cells, cheap to redraw), with the India outline as
SVG on top (d3-geo, equirectangular). Recharts for charts. Put the API URL in `VITE_API_URL`
(default `http://localhost:8000`).

```bash
cd dashboard && npm create vite@latest . -- --template react-ts && npm i d3-geo recharts && npm run dev
```

## Definition of done

- Works against the real server from a fresh clone, with only `npm install && npm run dev`.
- Judge flow in under 2 minutes: pick a window → normal forecast → apply a custom shock → watch the ripple →
  read the impact panel.
- Screenshots of the main views for page 2 of the brief.
