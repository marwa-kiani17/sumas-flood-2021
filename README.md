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

**Forecasting (10 rivers, median over rivers).**

| Lead | NSE persistence | NSE climatology | NSE LightGBM | NSE LightGBM without future weather |
|---|---|---|---|---|
| 1 day | 0.81 | 0.41 | **0.92** | 0.90 |
| 2 days | 0.64 | 0.41 | **0.89** | 0.77 |
| 3 days | 0.54 | 0.41 | **0.86** | 0.69 |

- Warning rule "0.9 quantile > 5-year flood": at 3 days ahead POD 0.66 / FAR 0.47 / CSI 0.42 versus persistence
  0.39 / 0.61 / 0.24.
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
| **U-Net (ResNet-18)** | **0.663** | **0.798** |

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

Rivers (12 chosen, 10 used; the Salmon at Falkland was not downloaded because of Open-Meteo's free-tier limit):
Coquihalla, Coldwater, Tulameen, Similkameen, Chilliwack, Englishman, Chehalis, Clearwater, Elk, Nooksack.

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
  the 3-day skill drops (NSE 0.86 to 0.69).
- **Record peaks are under-predicted.** Tree models cannot go beyond the flows seen in training, and Nov 2021 broke
  records on several rivers (for example the Similkameen peak was 932 m3/s). The 10-90% ranges also cover only about
  72-75% of observations instead of 80%.
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
python -m pytest tests -q
```

Station IDs are looked up by name with `python scripts/01_gauges.py --search "NAME"` and checked with `--check`.
All settings are in `config.yaml`; seeds are fixed. Downloaded data lives in `data/` (git-ignored).

## Future work

Real IVT and AR catalogues as inputs; a multi-basin LSTM (the main comparison planned); real archived weather
forecasts instead of reanalysis; more basins and flood events; a fix for extrapolation to record peaks; a simple
dashboard for flood managers.

## References

Curry, Islam et al. (2019), *Geophysical Research Letters*. Nearing et al. (2024), *Nature*. Bonafilia et al. (2020),
Sen1Floods11, CVPR Workshops. ECCC GeoMet, USGS Water Services, Open-Meteo, Microsoft Planetary Computer.
