# Multiverse of Monsoon: a digital twin of India's weather

DevForge 2026, Track 2. A fast, physics-aware emulator of India's daily weather (32 × 36 grid, 6 channels) that
rolls the atmosphere forward day by day and branches into "what if" universes when you apply a localized shock.

> Work in progress during the hackathon. Sections marked TODO are filled in as the pieces land.

## Repo layout

| Path | What |
|---|---|
| `serve.py` | Inference server (the judge runs `python serve.py --port 8000`) |
| `monsoon/` | Model definition and rollout engine used by the server |
| `assets/` | Trained weights, normalisation stats, static fields, climatology |
| `train.py` | Training entry point |
| `training/` | Training pipeline and data preparation (never imported by the server) |
| `dashboard/` | Interactive map dashboard (separate from the server; talks to `/simulate`) |
| `scripts/` | Checks we run before submitting |

## Run the server

```bash
pip install -r requirements.txt
python serve.py --port 8000
```

## Train

TODO

## Data

- `india_climate_train.nc` (2005–2021) from the organisers' team kit.
- Extra training years (1959–2004 and 2022) rebuilt with the organisers' own `build_dataset` code from the same
  ERA5 source (WeatherBench 2), in exactly the same format. Used for training only. The server never reads any
  data source; it only uses its weights and the request.

ERA5: Hersbach et al. 2020, Copernicus Climate Change Service, CC BY 4.0, accessed through WeatherBench 2
(Rasp et al. 2024).

## Dashboard

TODO
