# Complete technical walkthrough (every step, every choice)

This is the detailed companion to `PROJECT_EXPLAINED.md` (simple version). It follows the project in the order it was
built, step by step, and says for each piece: **what it does, how, with which settings, and why**. Use it to answer
"how exactly did you do X?" Numbers are from the final runs (11 rivers, test years 2017-2023) and match
`reports/results.md`.

Contents: 0 Layout and rules - 1 Gauges - 2 Weather - 3 Hydraulics - 4 Features and models - 5 Evaluation - 6 Replay
and SHAP - 7 Linear-leaf test - 8 Flood-mapping data and U-Net - 9 Sumas Prairie maps - 10 Area vs level -
11 Reports and docs - 12 Tests - 13 What differs from the original plan - 14 Complete file inventory

---

## 0. Layout, rules and environment

**Repository** (`E:\flooding project`, GitHub `marwa-kiani17/sumas-flood-2021`):

| Folder | Content |
|---|---|
| `config.yaml` | every setting (dates, stations, thresholds, model, CV, Sumas). No absolute paths in code. |
| `src/floodml/` | library code: `gauges.py`, `weather.py`, `hydraulics.py`, `features.py`, `models.py`, `evaluate.py`, `cv/dataset.py`, `cv/train_unet.py`, `cv/sentinel1.py`, `cv/map_flood.py` |
| `scripts/` | numbered steps `01_gauges.py` ... `11_warning_by_river.py`, run in order |
| `tests/` | `test_hydraulics.py` (5 tests), `test_metrics.py` (5 tests) |
| `data/` (git-ignored) | `raw/` downloads, `interim/` cleaned tables, `processed/` model outputs, `cache/` library weights |
| `reports/` | `figures/` (PNG), result tables (`tables_*.csv`), `results.md`, `evaluation_matrix.md` |
| `docs/` | the explanation files and `docs/figures/` |
| `logs/` (git-ignored) | console output of runs |

**Environment:** conda env `floodml` (Python 3.11) from `environment.yml`. PyTorch is the CUDA 12.8 build
(`torch 2.11.0+cu128`), verified on the NVIDIA RTX 3060. **Seeds:** `seed: 42` in the config, plus seeds 1 and 2 for the
extra U-Net runs. **Your rules (CLAUDE.md and `.claude/settings.json`):** stay inside the project folder, never read or
write `.env` files, use only the conda env. Library caches are redirected to `data/cache/` (`TORCH_HOME`, `HF_HOME`).
`CLAUDE.md`, `.claude/`, the proposal PDF, `data/` and `logs/` are git-ignored.

---

## 1. Step 1: river gauges (`scripts/01_gauges.py`, `src/floodml/gauges.py`)

**Goal:** a clean daily flow table for each river.

**Sources.**
- Canada: ECCC GeoMet OGC API, collection `hydrometric-daily-mean` (fields `STATION_NUMBER`, `DATE`, `DISCHARGE`,
  `LEVEL`, `DISCHARGE_SYMBOL_EN`). Pages of 10,000 rows with `offset`. Station metadata from `hydrometric-stations`
  (`STATION_NAME`, `DRAINAGE_AREA_GROSS`, point geometry).
- USA: USGS NWIS daily values (`parameterCd=00060`, discharge in ft3/s) converted to m3/s with x 0.0283168.
- Every response is cached as CSV in `data/raw/gauges/`.

**Station choice (never guessed).** `--search "NAME"` lists BC stations whose name contains the text; `--check IDs`
downloads them and reports first/last year, years of data, % missing, whether 8-20 Nov 2021 has all 13 days (and the
peak), and level availability. Decisions:
- Coquihalla: `08MF003` (near Hope) ended in 1983, so `08MF068` (above Alexander Creek, 1987-2024) was used.
- Coldwater `08LG010` (Merritt), Tulameen `08NL024` (Princeton; `08NL005` had only about 1 year), Similkameen `08NL038` (Hedley).
- Nooksack: `12213100` (Ferndale, 1980-2023). Deming `12210500` ended in 2005; Cedarville (2004-) and Everson (2016-) are
  too short. Ferndale is downstream of the overflow point (a stated limitation).
- Extra rivers: Chilliwack `08MH103`, Englishman `08HB002`, Chehalis `08MG001` (coastal, rain-fed); Clearwater `08LA001`,
  Salmon at Falkland `08LE020`, Elk at Fernie `08NK002` (interior, snow-fed). Regulated rivers (Cheakamus, Capilano, Nicola
  lake outlet, Salmon near Sayward with the Campbell diversion, Cowichan weir) were excluded **from the names only**.
- Kettle River `08NN026` was first included then dropped: a seasonal gauge with no winter record (no complete year, so no flood
  thresholds) and no flood in Nov 2021.
- Regime labels (rain / mixed / snow) in the config are my judgement.

**Cleaning (`clean_daily`).** Put on a full daily index (`asfreq("D")`); find runs of missing days; **interpolate only runs
of 3 days or fewer** and flag them (`interpolated` column); longer gaps stay NaN and those rows are dropped later. Only 3 days
were interpolated in the whole data set, so nearly all gaps are long (e.g. Coldwater about 1995-2005, Chehalis 1982-85).

**Outputs:** `data/interim/stations.csv` (id, name, regime, area, lon, lat, years, % missing, mean flow, Nov 2021 peak),
`flow_daily.parquet`, `reports/figures/hydrographs.png`. All 11 rivers have the full 13 days of Nov 2021.

---

## 2. Step 2: weather (`scripts/02_weather.py`, `src/floodml/weather.py`)

**Source:** Open-Meteo Historical API (ERA5 reanalysis), free, no key. `timezone=America/Vancouver`, 1980-01-01 to 2023-12-31.

**Variables.** Daily: `precipitation_sum`, `rain_sum`, `snowfall_sum`, `temperature_2m_max`, `temperature_2m_mean`.
Hourly: `snow_depth`, `soil_moisture_0_to_7cm`, averaged to daily means. The first 4 rivers were also downloaded with
`temperature_2m_min` and `soil_moisture_7_to_28cm`; these were dropped later to reduce cost and are unused (that is why those two
columns are about 60% empty).

**Points per basin:** one (the gauge location). The plan asked for 3-5; `basin_points()` can build a ring of extra points but
it was not used because of rate limits (`points_per_basin: 1`). Basin weather = mean over points (here just one).

**Rate limits (HTTP 429).** The free tier is billed by days x variables, with minutely, hourly and daily limits. The script:
caches each 15-year chunk as JSON in `data/raw/openmeteo/` (3 chunks per river), waits 65 s on a "Minutely" limit and retries,
stops cleanly on an hourly/daily limit and resumes from the cache, can wait an hour (`--wait`), can build the table from the
cache without any API call (`--cache-only`), and downloads rivers in a priority order from the config. About 4 rivers fit in
one hourly window; the last river (Salmon) was fetched after the daily limit reset.

**Output:** `data/interim/weather_daily.parquet`, 11 rivers x 16,071 days.

---

## 3. Step 3: flood frequency and rating curves (`scripts/03_hydraulics.py`, `src/floodml/hydraulics.py`)

**Annual maxima.** Highest daily flow per calendar year, using years with at least 300 valid days (daily max, not
instantaneous peaks; the ECCC annual-peaks collection was not used). Fitted on **training years only (1981-2012)**.
A fallback to March-October maxima (90% coverage) exists for seasonal gauges; Kettle still failed it and was dropped.

**GEV fit.** Generalized Extreme Value distribution by maximum likelihood, started from a Gumbel fit, with the shape
parameter bounded to xi in [-0.1, 0.4] (scipy's `c = -xi`). Reason: unconstrained fits on 20-30 peaks produced absurd results
(a 25-year flood of millions of m3/s) or a hard upper limit that the 2021 flood exceeded (infinite return period).
Return levels Q2, Q5, Q10, Q25 = `genextreme.ppf(1 - 1/T)`. **Bootstrap:** 300 resamples of the peaks, refit each time, 5th-95th
percentile = 90% interval. **Gumbel cross-check** (Q10 column). **Return period of a flow** = `1 / (1 - cdf)`, capped at
10,000 years. Return periods above 100 years are reported as ">100" because a 21-32-year fit cannot support them.

**Key results (m3/s, 1981-2012 fit):** for example Similkameen Q2 362 / Q5 466 / Q10 529 / Q25 602; Tulameen 171 / 227 / 269 / 330;
Nooksack 671 / 930 / 1131 / 1424. Nov 2021 return periods: Coquihalla 22, Chilliwack 28, Chehalis 30, Nooksack 24 years;
Coldwater, Tulameen, Similkameen above 100 years; Clearwater, Salmon, Elk about 1 (no flood in that month).

**Rating curves.** `Q = a (h - h0)^b` by `scipy.optimize.curve_fit` with bounds a > 0, 1 <= b <= 3.5, h0 below the lowest observed
level; fitted on paired level/flow days 2013-2023 (older ratings drift as channels change). 10 ECCC stations (not the Nooksack,
which has no level data in this project). `flow_to_level()` inverts the curve. **Not done:** the curve was fitted and plotted but not
used to convert the model's forecast into a forecast water level.

**Outputs:** `data/processed/thresholds.csv`, `rating_params.csv`, `return_periods.png`, `rating_curves.png`.

---

## 4. Step 4: features and models (`src/floodml/features.py`, `models.py`, `scripts/04_features_models.py`)

### 4.1 The forecast set-up
Each row is **one river on one issue day t**. The model predicts flow on day t+1, t+2 or t+3 (three separate models per
quantile, one per lead time). All rivers share the same model.

**Target:** `y_k = log1p( Q(t+k) / meanQ )`, where meanQ is the river's mean flow over the **training years only**. This puts
small and large rivers on one scale. Predictions are converted back with `expm1(...) x meanQ`.

### 4.2 The features (32 for lead 1; 34 / 36 for leads 2 / 3)
Only information available on day t, except the optional `fw_*`.

| Group | Features |
|---|---|
| Flow | `q_lag0..q_lag7` (log-scaled flow today and the 7 previous days), `q_mean7` |
| Rain | `rain`, `rain_3d`, `rain_7d`, `rain_30d` (mm sums) |
| Snow | `snowfall`, `snowfall_7d`, `snow_depth`, `snow_depth_chg7` (7-day change, negative = melting) |
| Temperature | `t_mean`, `t_max`, `degday_7d` (sum of positive daily mean temperature over 7 days) |
| Soil | `soil_0_7` (soil moisture 0-7 cm) |
| Storm | `ros` = rain x (snow_depth > 0.05 m and T > 0); `ros_3d`; `ar_flag` = (rain_3d >= 50 mm and mean T >= 2 C and month in Oct-Feb); `ar_rain3` = rain_3d x flag |
| Season | `doy_sin`, `doy_cos` |
| Static | `log_area`, `regime` (0 rain, 1 mixed, 2 snow), `log_meanq` |
| Future weather | `fw_rain_j`, `fw_t_j` for j = 1..k (the recorded rain and mean temperature of days t+1..t+k: a perfect forecast) |

Rows are dropped when any of the 8 flow lags or the target is missing (this removes the long gaps). Missing weather values are
left to LightGBM (it handles NaN).

### 4.3 Time split (no leakage)
Train 1981-2012, validation 2013-2016 (early stopping only), test 2017-2023 (contains Nov 2021). By issue date; **the last 3 days of
the train and validation periods are dropped** so their targets never reach into the next period. meanQ, GEV thresholds and the
climatology use training years only.

### 4.4 Baselines
- **Persistence:** forecast = today's flow.
- **Climatology:** mean flow for the target's day of year over the training years, smoothed with a circular 15-day window.

### 4.5 LightGBM quantile models
One `LGBMRegressor(objective="quantile", alpha=a)` for each a in {0.1, 0.5, 0.9} and each lead. Settings: learning rate 0.06,
up to 2000 trees, 31 leaves, min 50 samples per leaf, subsample 0.8 (every iteration), column sample 0.8, L2 = 1, seed 42,
early stopping after 100 rounds without improvement on the validation pinball loss. The three outputs are **sorted per row** so
10% <= 50% <= 90% never cross. (The first version with learning rate 0.03 and 1500 trees stopped at the cap, so it was raised.)

**Four variants are trained:** `full`; `no_storm` (drops `ros`, `ros_3d`, `ar_flag`, `ar_rain3`); `no_future` (drops `fw_*`);
`linear` (see section 7). Models are saved to `data/processed/models/*.txt`; predictions for validation and test years to `preds.parquet`.

---

## 5. Evaluation (`src/floodml/evaluate.py`)

- **NSE** = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2). **KGE** = 1 - sqrt((r-1)^2 + (std_sim/std_obs - 1)^2 +
  (mean_sim/mean_obs - 1)^2), where r is the correlation. Computed **per river and lead on the test years**, then the **median over rivers**.
- **Warning rule:** warn on day t if the forecast **0.9 quantile for t+k exceeds the river's Q5** (from the GEV). An "event day" is a day
  where observed flow > Q5. POD = hits/(hits+misses); FAR = false alarms/(hits+false alarms); CSI = hits/(hits+misses+false alarms).
  Pooled over all rivers and test days (162 event days at each lead). Persistence is scored the same way (warn if today's flow > Q5).
- **Coverage** = share of observations between the 0.1 and 0.9 forecasts (target 0.80).
- **Peak error:** events = runs of days with observed flow > Q2 (runs closer than 3 days merged); within +-3 days of each event, compare
  the observed maximum with the forecast median's maximum: size error % and timing error in days (69 events per lead).
- `11_warning_by_river.py` repeats the warning scores **per river and per regime** (rain / mixed / snow).

**Final results (test years, 11 rivers):**

| Lead | NSE pers. | NSE clim. | NSE LGBM | NSE no future | KGE pers. | KGE LGBM |
|---|---|---|---|---|---|---|
| 1 | 0.839 | 0.439 | 0.926 | 0.902 | 0.920 | 0.879 |
| 2 | 0.687 | 0.439 | 0.907 | 0.794 | 0.843 | 0.823 |
| 3 | 0.598 | 0.439 | 0.887 | 0.713 | 0.799 | 0.808 |

(By KGE, persistence is better than the model at 1 and 2 days and slightly worse at 3 days.) Warnings, LightGBM `full` / persistence: POD 0.815 / 0.784 (1 day), 0.673 / 0.617 (2),
0.537 / 0.531 (3); FAR 0.320 / 0.216, 0.335 / 0.383, 0.424 / 0.469; CSI 0.589 / 0.645, 0.502 / 0.446, 0.385 / 0.361.
Coverage 0.747 / 0.727 / 0.716. Median peak error -13.5 / -17.3 / -20.9%, timing 1 day. **By regime** (3 days, POD model vs
persistence): rain 0.73 vs 0.00 (11 event days), mixed 0.56 vs 0.00 (9), snow 0.52 vs 0.61 (142). The no-storm variant is as good as `full`
(storm features add nothing).

---

## 6. Step 5: Nov 2021 replay and SHAP (`scripts/05_replay_shap.py`)

**Replay.** For each flooded river with weather (Coquihalla, Coldwater, Tulameen, Similkameen, Nooksack) the figure shows observed flow
(black), the model median and 10-90% band, and persistence, for 1-day and 3-day forecasts, with Q2/Q5/Q10 lines. **First warning day** = the
first issue day in 8-20 Nov on which any lead's 0.9 quantile exceeds Q5; for persistence, the first day today's flow exceeds Q5.
Days gained = persistence day minus model day. Results: **3 days** on Coquihalla (11 vs 14 Nov), Coldwater, Tulameen and Nooksack (12 vs
15 Nov); **no warning at all** on the Similkameen. Without future weather: 0-1 days. The 1-day median forecast still under-predicts the peaks
(for example the Similkameen: a 1-day median forecast in the range of 300-400 m3/s against the observed 932; read the exact value from `nov2021_replay_08NL038.png`).

**SHAP** (`TreeExplainer` on the median model of `full`): a global summary for leads 1 and 3 (4,000 random test rows) and a waterfall
for the Nov 2021 peak (issue day = peak day - 1) of three rivers. The strongest feature is today's flow (`q_lag0`), then rain today and
the next day. SHAP values are in the target's log units. Skipped for the linear variant (SHAP does not support linear trees).

---

## 7. The linear-leaf test (the attempted fix)

Problem: tree models cannot predict above the highest flows seen in training. `linear` uses `linear_tree=True`, `linear_lambda=1.0`
(straight-line fits inside each leaf can extrapolate). It stopped after only 88-306 trees (validation years contain no record
floods, so early stopping cannot tune extrapolation). Result: POD 0.877 / 0.722 / 0.679, CSI 0.693 / 0.504 / 0.462, coverage 0.772 / 0.749 / 0.734
(better than `full`), but NSE 0.915 / 0.894 / 0.864 (slightly worse) and peak error -13 / -19 / -23% (no real change). Nov 2021 peaks stay
under-predicted by roughly 40-60%; the Similkameen warning fires only on the same day as persistence. The default stays `full`.

---

## 8. Step 6: flood-mapping data and U-Net (`src/floodml/cv/`, `scripts/06_cv_train.py`)

**Data:** Sen1Floods11 hand-labelled set from the public bucket (HTTP, no account): 252 train, 89 validation, 90 test chips, 512x512 px,
using the official split CSVs (about 750 MB). Each chip has VV and VH in dB; the label is -1 (no data, ignored), 0 (not water) or 1 (water).

**Input (3 channels, 0-1):** VV and VH clipped to [-30, 5] dB and scaled; VV-VH clipped to [-5, 20] dB and scaled (a deliberate change from
the plan's [-30, 5], which would flatten this mostly positive channel). NaN replaced by -30 dB. All chips preloaded in memory as float16.
**Augmentation (training only):** random horizontal flip, vertical flip and a random 90-degree turn.

**Model:** `segmentation_models_pytorch.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)`; ImageNet weights are
downloaded from the Hugging Face hub into `data/cache/hf`.
**Loss:** BCE (binary cross-entropy) + Dice, both computed only over valid pixels. **Training:** AdamW, learning rate 1e-3, weight decay 1e-4,
cosine schedule, batch size 8, 30 epochs, mixed precision, about 7 s per epoch on the RTX 3060; the checkpoint with the best **validation IoU** is kept.

**Baselines (Otsu):** pick the VV threshold that maximises between-class variance of the histogram (256 bins); water = VV below it.
"Per chip" fits one threshold per image (fails on chips with little water); "global" is fitted on a subsample (every 50th pixel) of all
training chips: **-15.6 dB**.

**Scores:** IoU, F1, precision, recall of the water class, **pooled over all test pixels** (sum of TP, FP, FN) with -1 pixels ignored.

| Method | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| Otsu per chip | 0.213 | 0.351 | 0.233 | 0.711 |
| Otsu global | 0.461 | 0.631 | 0.672 | 0.594 |
| U-Net, seed 42 | 0.663 | 0.798 | 0.812 | 0.784 |
| U-Net, seed 1 | 0.677 | 0.807 | 0.830 | 0.786 |
| U-Net, seed 2 | 0.665 | 0.798 | 0.806 | 0.791 |

Mean over 3 seeds: **IoU 0.668 +- 0.008, F1 0.801 +- 0.005** (sample standard deviation). Validation IoU jumped around (0.35 to 0.64)
during the high-learning-rate epochs, then settled at about 0.64.

---

## 9. Step 7: Sumas Prairie maps (`scripts/07_sumas_maps.py`, `cv/sentinel1.py`, `cv/map_flood.py`)

**Search:** Planetary Computer STAC API, collection `sentinel-1-rtc` (terrain-corrected VV and VH), anonymous signing. Area
lon -122.35 to -122.05, lat 48.98 to 49.15 (about 423 km2, 1913 x 2211 pixels of 10 m). Scenes exist from two satellites and several tracks;
to avoid viewing-angle artefacts **only relative orbit 13 (descending)** is used: 4, 16, 22, 28 Nov and 4, 10, 16, 22, 28 Dec 2021.
Scenes of Sentinel-1A (22 Nov, 4, 16, 28 Dec) cover only 61% of the area.

**Processing:** read each band cropped to the area from the cloud, convert to dB (10 log10), predict with the trained U-Net in **overlapping
512x512 tiles (stride 384)**, average overlaps, threshold the water probability at 0.5. All scenes are resampled (nearest neighbour) to the pixel
grid of the 4 Nov "before" scene.

**Flooded area = water now AND NOT permanent water AND valid in both scenes.** Permanent water = water in the 4 Nov scene **or** pixels that are
dark (VV < -15.6 dB) in at least 80% of the 5 full-coverage scenes. (The first version used only the 4 Nov scene and counted a wind-roughened Fraser
River as flood; a second version using the U-Net's own persistence also failed; this one works.) Area = pixels x 100 m2. Two area series are given:
over the whole covered area, and over the **common region seen by every scene (257 km2)**, which is comparable across dates.

**Results (km2):** 16 Nov: 19.4 (whole area, 11.8 in the common region); 28 Nov: 13.7 (5.6); 10 Dec: 5.1 (2.8); 22 Dec: 3.5 (2.6); 4 Nov: 0.
The Otsu version of the same area is erratic on Sentinel-1A scenes (10-23 km2 on dates where the U-Net gives 1-3 km2), so it is shown for
comparison only. The U-Net map under-covers the large solid flooded patch visible in the radar image.

---

## 10. Step 8: flooded area vs river level (`scripts/08_area_vs_level.py`)

Each satellite date is paired with the **daily mean** of: Nooksack flow (Ferndale, m3/s), Sumas River near Huntingdon level (`08MH029`, m),
Fraser River at Mission level (`08MH024`, m; tidal). The common-region area is used. Spearman rank correlation: Sumas level 0.79 (n = 8),
Nooksack flow 0.57 (n = 9), Fraser level 0.45 (n = 8). With so few dates and one dominant date (16 Nov), this is **descriptive only**, not a
hydraulic model. A satellite pass is about 06:20 local time while the gauge value is a daily mean.

---

## 11. Step 9-11: reports, figures, docs

- `09_report.py` builds `reports/results.md` from the saved tables (so numbers cannot drift), including the by-regime and seed tables.
- `10_explain_figures.py` draws `docs/figures/` (pipeline, NSE idea, warning table, IoU idea, coverage idea, evaluation matrix) from the real
  results and writes `reports/evaluation_matrix.md`.
- `11_warning_by_river.py` writes `tables_warning_by_river.csv` and `tables_warning_by_regime.csv`.
- `app/streamlit_app.py` is a viewer of the saved outputs (needs `pip install streamlit`; not run).

---

## 12. Tests (10, all passing)

`test_hydraulics.py`: GEV recovers known quantiles; return period round-trip; bootstrap interval brackets the estimate; annual maxima skip
incomplete years; rating curve recovery and inversion. `test_metrics.py`: NSE/KGE perfect and mean cases; POD/FAR/CSI on a known table;
coverage; flood-event grouping with peak timing/size error; the split leaves a 3-day gap.

---

## 13. What differs from the original plan (say this openly)

| Plan | What was done |
|---|---|
| 10-15 stations | 11 (12 selected, Kettle dropped) |
| 3-5 weather points per basin | 1 (the gauge), because of API limits |
| Instantaneous annual peaks if available | daily maxima of daily flow |
| Forecast flow converted to forecast water level with the rating curve | curves fitted and plotted, **not applied** to forecasts |
| Nooksack rating curve | none (no level data used) |
| Optional LSTM | not built (future work) |
| Real IVT | not available; rain + warmth proxy |
| Sentinel-1 "during" 14-30 Nov | one orbit only (orbit 13) for like-for-like comparison |
| Streamlit "if time remains" | written, not executed |
| 80% coverage expected | 72-75% achieved (ranges too narrow) |
| Warning benefit assumed general | benefit is real for rain-driven rivers, small for slow snowmelt rivers |

## 14. Output inventory

**Tables (`reports/`):** `tables_skill_per_river.csv`, `tables_warning.csv`, `tables_warning_by_river.csv`, `tables_warning_by_regime.csv`,
`tables_peak_errors.csv`, `tables_nov2021_warning.csv` (+ `_linear`), `tables_cv.csv` (+ `_seed1`, `_seed2`, training histories), `results.md`,
`evaluation_matrix.md`. **Processed data:** `thresholds.csv`, `rating_params.csv`, `features.parquet`, `preds.parquet`, `sumas_area.csv`,
`area_vs_level.csv`, `sumas/*.npz` (masks), `unet_resnet18*.pt`, `models/*.txt`.
**Figures (`reports/figures/`):** `hydrographs`, `return_periods`, `rating_curves`, `nov2021_replay_<river>` (+ `_linear`), `shap_summary_lead1/3`,
`shap_nov2021_<river>`, `cv_examples`, `sumas_flood_maps`, `flooded_area_vs_level`, `flooded_area_timeseries`.
