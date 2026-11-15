# Evaluation matrix

One table with every score used in this project. Results are on data the model never saw.

| Score | What it tells you | Good value | Our result | Notes |
|---|---|---|---|---|
| NSE (flow forecast) | How close forecast flow is to real flow, compared with just using the average | 1 = perfect; 0 = same as the average | 0.93 / 0.91 / 0.89  (persistence 0.84 / 0.69 / 0.60) | 1 / 2 / 3 days |
| KGE (flow forecast) | Like NSE but also checks bias and spread | 1 = perfect | 0.88 / 0.82 / 0.81 | 1 / 2 / 3 days |
| POD (warning) | Share of real floods (above the 5-year level) that were warned | 1 = none missed | 0.81 / 0.67 / 0.54  (persistence 0.78 / 0.62 / 0.53) | 1 / 2 / 3 days; pooled over rivers (snowmelt-dominated), see by-regime table in results.md |
| FAR (warning) | Share of warnings that were false alarms | 0 = no false alarms | 0.32 / 0.34 / 0.42 | 1 / 2 / 3 days |
| CSI (warning) | One combined warning score (hits, misses, false alarms) | 1 = perfect | 0.59 / 0.50 / 0.38 | 1 / 2 / 3 days |
| Coverage 10-90% | How often reality falls inside the forecast range | about 0.80 expected | 0.75 / 0.73 / 0.72 | 1 / 2 / 3 days |
| Peak error | How far off the forecast flood peak is (negative = too low) | 0 = perfect | -14% / -17% / -21% (median) | 1 / 2 / 3 days |
| IoU (water map) | Overlap of mapped water with true water divided by their union | 1 = perfect | U-Net 0.67 +/- 0.01 (3 seeds), threshold 0.46 | Sen1Floods11 test chips |
| F1 (water map) | Balance of precision and recall for water pixels | 1 = perfect | U-Net 0.80 +/- 0.01, threshold 0.63 | Sen1Floods11 test chips |
| Spearman (area vs level) | Do bigger floods go with higher river level? (rank correlation) | 1 = always together | 0.79 with Sumas level | only 8 dates: descriptive |
