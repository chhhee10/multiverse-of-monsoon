# Multiverse of Monsoon: a fast, physics-aware digital twin of India's weather

*DevForge 2026, Track 2.* Repo: github.com/chhhee10/multiverse-of-monsoon

## Page 1: the engine

**USP.** A 1–4 M-parameter neural twin that rolls India's atmosphere forward day by day in about 0.1–0.3 s for
10 days on 4 CPU cores, stays stable and sharp over long rollouts, and answers "what if" questions causally: a shock
starts where you put it and spreads at physical speeds.

**Data.** The organisers' 2005–2021 archive plus 1959–2004 and 2022 rebuilt with the organisers' own
`build_dataset` code from the same ERA5 source (WeatherBench 2), bit-identical in format (we verified every value of
an overlapping month matches exactly). That is 3.7× more training weather. Training only: the server reads nothing but
its weights and the request. 2021–2022 are held out for model selection; nothing after 2022 is ever used.

**Model.** State = 6 fields on the 32 × 36 grid. Each day the network sees:
- the last 3 days;
- today's anomaly from climatology, plus tomorrow's climatology and its change;
- land–sea mask, orography, latitude and longitude, and the day of year.

It predicts **tomorrow's change** (a residual), plus the probability of heavy rain (> 10 mm/day). The core is a
ConvNeXt-style stack of depthwise 5 × 5 convolutions and per-cell MLPs.
- **Strictly local.** No attention, no pooling and no spatial normalisation, so information travels at most ~13–17
  cells (~1,500 km) per day.
- **Edges.** Replicate padding at the domain edges.
- **Starting point.** A zero-initialised output, so the untrained twin is exactly persistence and training only adds
  skill.

**Preventing drift.** One-day training is followed by autoregressive fine-tuning on 2-, 4- and 8-day rollouts.
The model is fed its own predictions and penalised on every day, so it learns to correct its own errors instead of
compounding them. Model selection uses the organisers' own scoring functions on held-out rollouts.

**Preventing blur.**
- Rain is modelled in log space, so heavy-rain cells are not averaged away.
- The model predicts residuals, so spatial structure is kept unless the dynamics change it.
- A separate heavy-rain head is calibrated on held-out years. Where it is confident, the reported rain is raised to
  at least 10.5 mm/day. This only affects the reported field, never the model's internal state.
- Day-8 spatial variance stays at RESULT_BLUR of the real fields.

**Shocks (the multiverse).** The perturbation is added to the last input day exactly as the contract specifies.
- **Clipping:** the result is clipped to physical bounds, e.g. no negative moisture.
- **Locality:** because every layer is local, the response is local on day 1 and spreads outward on later days.
- **Zero shock:** inference is deterministic, so a zero shock reproduces the unperturbed run bit for bit.

**Results (held-out 2021–2022, organisers' scoring maths).** RESULTS_TABLE

**Failure modes and limits.**
- 6 surface fields only (no upper air), so cyclones and western disturbances are steered without the jet stream.
- 1° resolution smooths orographic rain (Western Ghats, Himalaya).
- Predictions regress towards climatology after about 6–8 days.
- The heavy-rain flag trades a little RMSE for hits.
- Trained on a climate up to 2022. 2023–24 is warmer, so a small t2m bias is likely.
- Shocks far outside the training range are clipped rather than extrapolated.

## Page 2: the dashboard and the impact module

SCREENSHOT_1 (forecast vs reality) · SCREENSHOT_2 (what-if: warm Bay of Bengal)

**The dashboard** (Vite + React + TypeScript, separate from the server) runs entirely on live `/simulate` calls.
- **Forecast vs reality:** pick one of 7 real 2022 starting situations (monsoon onset, heavy rain, heatwave, cyclone,
  western disturbance, calm winter, post-monsoon) and see the twin's 10-day forecast side by side with what ERA5
  recorded, plus the error compared with "repeat yesterday".
- **Multiverse:** apply the judges' presets or paint any shock on the map (box or brush, any variable, absolute or %).
  The dashboard shows normal | shocked | difference for every day, an animated ripple, and a chart of the response
  inside vs outside the shocked area.

**Impact module: who is affected?** For 8 regions (Kerala, Konkan, Vidarbha, Gangetic plain, Odisha coast, Delhi NCR,
Gujarat, Tamil Nadu) the dashboard compares the normal and shocked futures on:
- **Kharif water stress:** the 10-day rain total, flagging regions that lose more than 25%;
- **Heat stress:** land cell-days above a mean-temperature threshold;
- **Wind-power potential:** |wind|³ over land and coast.

It ends with one plain-English sentence, e.g. IMPACT_EXAMPLE.

**Why it matters.** Monsoon variability decides India's Kharif harvest, reservoir levels and power demand. A twin that
answers "what if the Arabian Sea branch weakens this week?" in under a second lets an agriculture or grid planner
explore scenarios interactively instead of waiting for a supercomputer run.
