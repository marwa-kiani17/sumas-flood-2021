# Sumas Prairie 2021: Mapping and Forecasting the November 2021 British Columbia Floods

Satellite flood mapping (computer vision) and machine-learning river forecasting for the November 2021
atmospheric-river floods in southern BC. A small demo project built in one session; every number below is
produced by the scripts in this repository (see `reports/results.md`).

**The event.** On 14-15 Nov 2021 an atmospheric river (a long band of very moist air from the Pacific) dropped
record rain on southern BC. The Coquihalla, Coldwater, Tulameen and Similkameen rivers reached extreme flows, and
the Nooksack River (Washington, USA) overflowed into the Sumas River basin and flooded Sumas Prairie (Abbotsford).

## What it does

| Part | Question | Method |
|---|---|---|
| **C. Forecasting** | Will a river flood in the next 1-3 days, and how sure are we? | One global LightGBM quantile model (10/50/90%) for 10 rivers, compared with persistence and climatology |
| **A. Flood mapping** | Where was the water? | U-Net (ResNet-18) trained on Sen1Floods11, applied to Sentinel-1 radar over Sumas Prairie |
| **B. Hydraulics** | How rare was it, and how does flooded area relate to river level? | GEV flood frequency, rating curves, flooded area vs gauge level |

## Headline results (test years 2017-2023, never used for training)

**Forecasting (11 rivers, median over rivers).**

| Lead | NSE persistence | NSE climatology | NSE LightGBM | NSE LightGBM without future weather |
|---|---|---|---|---|
| 1 day | 0.84 | 0.44 | **0.93** | 0.90 |
| 2 days | 0.69 | 0.44 | **0.91** | 0.79 |
| 3 days | 0.60 | 0.44 | **0.89** | 0.71 |

- **Warnings (rule: 0.9 quantile > 5-year flood level) depend on the river type.** Pooled over all rivers the
  model is only slightly better than persistence at 2-3 days (3 days: POD 0.54 / FAR 0.42 / CSI 0.38 versus 0.53 /
  0.47 / 0.36), because 142 of the 162 flood days in the test years are slow snowmelt floods, where "tomorrow =
  today" already works well (on the Salmon River persistence is clearly better). On rain-driven and mixed rivers,
  the Nov 2021 type, the model clearly wins (3 days ahead: POD 0.73 on rain rivers and 0.56 on mixed rivers versus
  0.00 for persistence), but these have only 11 and 9 test flood days and a high false alarm rate on rain rivers
  (FAR 0.69). See `reports/results.md` (by-regime table).
- **Nov 2021 replay:** with the (perfect-forecast) future weather the model warned **3 days earlier than persistence**
  on the Coquihalla, Coldwater, Tulameen and Nooksack. Without future weather the gain is 0-1 days. On the
  Similkameen no warning fired (see limitations).
- Adding the rain-on-snow and AR-like storm features did **not** improve the scores: rain, temperature and snow depth
  already carry most of that signal.

**Flood mapping (Sen1Floods11 test chips, water class).**

| Method | IoU | F1 |
|---|---|---|
| Otsu threshold, per chip | 0.213 | 0.351 |
| Otsu threshold, global (fit on train) | 0.461 | 0.631 |
| **U-Net (ResNet-18)**, mean of 3 training seeds | **0.668 +/- 0.008** | **0.801 +/- 0.005** |

**Sumas Prairie (Sentinel-1, relative orbit 13, same view geometry for every date).** About 19 km2 flooded on
16 Nov 2021 (12 km2 within the 257 km2 region seen by every scene), falling to about 3 km2 by 22 Dec. Flooded area
follows the Nooksack hydrograph and the Sumas River level (Spearman 0.79, n = 8): a descriptive relationship from a
few dates, not a hydraulic model.

Key figures in `reports/figures/`: `nov2021_replay_<river>.png`, `sumas_flood_maps.png`,
`flooded_area_vs_level.png`, `flooded_area_timeseries.png`, `return_periods.png`, `shap_*.png`, `cv_examples.png`.

## Data (all free, no queued services)

| Data | Source |
|---|---|
| River flow and level, Canada | ECCC GeoMet API (HYDAT daily values) |
| Nooksack River flow | USGS NWIS daily values |
| Weather (rain, snow, temperature, snow depth, soil moisture) | Open-Meteo historical API (ERA5), one point per basin |
| Radar | Sentinel-1 RTC via Microsoft Planetary Computer |
| Training labels for the U-Net | Sen1Floods11 (Bonafilia et al., 2020), hand-labelled part only |

Rivers (11): Coquihalla, Coldwater, Tulameen, Similkameen, Chilliwack, Englishman, Chehalis, Clearwater, Salmon
(Falkland), Elk, Nooksack. (The Kettle River was dropped: a seasonal gauge that cannot give flood thresholds.)

## Method notes

- **Splits by time, no shuffling:** train 1981-2012, validation 2013-2016 (early stopping), test 2017-2023.
  Flood thresholds, mean flows, climatology and scalers use training years only.
- **Forecast target:** log1p(Q / basin mean Q), so all rivers share one model; one model per lead (1, 2, 3 days) and
  per quantile; quantiles are sorted so they never cross.
- **Features:** flow lags, rain and snowfall sums, temperature, degree-days, snow depth and its change, soil
  moisture, rain-on-snow, an AR-like storm flag (3-day rain >= 50 mm and mean temperature >= 2 C, Oct-Feb),
  day-of-year, basin area. Optional perfect-forecast weather for days t+1..t+k.
- **Flood frequency:** GEV on annual maxima of daily flow, shape bounded to [-0.1, 0.4], bootstrap 90% intervals.
  Rating curve Q = a (h - h0)^b fitted on 2013-2023.
- **Flood mapping:** U-Net on VV, VH and VV-VH (dB) with BCE + Dice loss ignoring no-data pixels; scenes tiled
  into overlapping 512x512 patches. Flooded area = water now and not permanent water (water in the 4 Nov scene or
  dark in at least 80% of the full-coverage scenes, which keeps the Fraser River out of the flood).

## Limitations (honest)

- **Weather is one ERA5 point per basin** (the gauge) and has no IVT; the atmospheric river influence is a rain +
  warmth proxy. Real IVT / AR catalogues are future work.
- **"Future weather" is observed reanalysis**, a perfect forecast. Real forecasts will be less accurate; without it
  the 3-day skill drops (NSE 0.89 to 0.71).
- **Record peaks are under-predicted.** Tree models cannot go beyond the flows seen in training, and Nov 2021 broke
  records on several rivers (for example the Similkameen peak was 932 m3/s). The 10-90% ranges also cover only about
  72-75% of observations instead of 80%.
- **Attempted fix: LightGBM linear leaves** (can extrapolate in principle). Result (11 rivers): modestly better
  warnings (POD 0.88/0.72/0.68 and CSI 0.69/0.50/0.46 at 1/2/3 days versus 0.81/0.67/0.54 and 0.59/0.50/0.38) and
  slightly better 10-90% coverage (0.77/0.75/0.73 versus 0.75/0.73/0.72), but slightly lower NSE
  (0.92/0.89/0.86 versus 0.93/0.91/0.89) and **no real improvement in peak size** (median error -13/-19/-23% versus
  -14/-17/-21%). Nov 2021 peaks are still under-predicted by roughly 40-60% (Similkameen: median forecast about 360
  versus observed 932 m3/s), and the linear model only warned on the Similkameen on the same day as persistence.
  The validation years contain no record floods, so early stopping cannot tune extrapolation. The default model
  stays the standard LightGBM; the linear variant is reported for comparison (`reports/results.md`).
- Regulation of the gauges was judged from station names only. Several gauges failed or were estimated in Nov 2021.
- The U-Net is trained on global flood chips, not BC farmland. It under-maps the solid flooded area on Sumas Prairie
  (flooded fields can look partly rough), and radar confuses flat wet fields and wet snow with water. The four
  Sentinel-1A scenes cover only part of the area, so areas are compared on the common region.
- Flooded area vs level uses 8-9 satellite dates with one dominant date (16 Nov).

## How to run

```bash
conda env create -f environment.yml
conda activate floodml
# force the CUDA build of PyTorch (the CPU build is pulled in first):
pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128

python scripts/01_gauges.py --build          # stations + daily flow (stations come from config.yaml)
python scripts/02_weather.py --wait          # Open-Meteo; resumable (free tier is rate limited)
python scripts/03_hydraulics.py              # GEV flood levels, rating curves
python scripts/04_features_models.py         # baselines + LightGBM quantile models
python scripts/05_replay_shap.py             # Nov 2021 replay + SHAP
python scripts/06_cv_train.py --download     # Sen1Floods11 hand-labelled chips (~750 MB)
python scripts/06_cv_train.py                # Otsu + U-Net
python scripts/07_sumas_maps.py --map        # Sentinel-1 flood maps of Sumas Prairie
python scripts/08_area_vs_level.py           # flooded area vs level
python scripts/09_report.py                  # reports/results.md
python scripts/10_explain_figures.py         # docs figures + reports/evaluation_matrix.md
python -m pytest tests -q

python scripts/06_cv_train.py --seed 1       # extra seeds for the U-Net error bar (outputs get a _seedN suffix)
python scripts/05_replay_shap.py linear      # Nov 2021 replay for the linear-leaf variant

pip install streamlit                        # optional viewer
streamlit run app/streamlit_app.py
```

## Documentation for readers

| File | What it is |
|---|---|
| `docs/PROJECT_EXPLAINED.md` | The whole project in the simplest words, with pictures |
| `docs/PROJECT_DETAILS.md` | Complete technical walkthrough: every step, setting, formula, number and deviation from the plan |
| `docs/METRICS_EXPLAINED.md` | Every score explained simply, with toy examples |
| `reports/evaluation_matrix.md` | One table with all scores and our results |
| `docs/CHALLENGES.md` | Problems hit during development and how each was solved |
| `docs/PROPOSAL_SUMMARY.md` | One-page summary and short research proposal |
| `reports/results.md` | All result tables (generated) |

Station IDs are looked up by name with `python scripts/01_gauges.py --search "NAME"` and checked with `--check`.
All settings are in `config.yaml`; seeds are fixed. Downloaded data lives in `data/` (git-ignored).

## Future work

Real IVT and AR catalogues as inputs; a multi-basin LSTM (the main comparison planned); real archived weather
forecasts instead of reanalysis; more basins and flood events; a fix for extrapolation to record peaks; a simple
dashboard for flood managers.

## References

Curry, Islam et al. (2019), *Geophysical Research Letters*. Nearing et al. (2024), *Nature*. Bonafilia et al. (2020),
Sen1Floods11, CVPR Workshops. ECCC GeoMet, USGS Water Services, Open-Meteo, Microsoft Planetary Computer.
