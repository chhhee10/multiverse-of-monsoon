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

```bash
pip install -r requirements-train.txt
# data: put the kit's india_climate_train.nc + stats.json in data/, then (optional) extra ERA5 years:
modal run training/data/pull_era5_modal.py --start 1959-01-01 --end 2004-12-31
modal run training/data/pull_era5_modal.py --start 2022-01-01 --end 2022-12-31
modal volume get monsoon-data chunks data/

python -m training.build_assets             # climatology, normalisation, static fields -> assets/
python train.py --out runs/v1               # local GPU/CPU, or on Modal:
modal run training/modal_train.py --name v1
cp runs/v1/model.pt assets/model.pt
python -m training.calibrate                # heavy-rain calibration -> assets/post.json
```

Training curriculum: one-day-ahead training, then autoregressive fine-tuning on 2-, 4- and 8-day rollouts in which
the model is fed its own predictions (this is what keeps 10-day rollouts stable and sharp). Model selection uses the
organisers' own scoring functions on the held-out years 2021-2022.

## Evaluate and check

```bash
python -m training.evaluate --model engine --year 2022    # organisers' scoring maths on held-out event windows
scripts/preflight.sh                                      # repo scan + offline judge-like container + schema validator
```

## Data

- `india_climate_train.nc` (2005–2021) from the organisers' team kit.
- Extra training years (1959–2004 and 2022) rebuilt with the organisers' own `build_dataset` code from the same
  ERA5 source (WeatherBench 2), in exactly the same format. Used for training only. The server never reads any
  data source; it only uses its weights and the request.

ERA5: Hersbach et al. 2020, Copernicus Climate Change Service, CC BY 4.0, accessed through WeatherBench 2
(Rasp et al. 2024).

## Dashboard

TODO
