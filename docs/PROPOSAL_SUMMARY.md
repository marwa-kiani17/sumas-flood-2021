# Proposal summary

**Applicant:** Marwa Kiani, BS Data Science (COMSATS University Islamabad), AI Engineer
**Proposed supervisor:** Dr. Siraj Ul Islam, Hydro-Climate Informatics Lab, University of Northern British Columbia
**Code and results:** https://github.com/marwa-kiani17/sumas-flood-2021

---

# Part 1. One-page summary (send with two figures)

**Title: Mapping and Forecasting the November 2021 British Columbia Floods**

In November 2021 an atmospheric river brought record rain to southern BC. The Nooksack River overflowed into the
Sumas River basin and flooded Sumas Prairie, and rivers such as the Coquihalla, Coldwater, Tulameen and Similkameen
reached extreme flows. As a first step toward research in your lab, I built a small project that looks at this event
from two sides: how much water was coming, and where it went.

**Forecasting.** One LightGBM model, trained on 11 BC and Washington rivers (1981-2012), forecasts river flow 1-3 days
ahead as a low-middle-high range and warns when the high value passes the 5-year flood level. On the test years
2017-2023 its accuracy (NSE) is 0.93 / 0.91 / 0.89 at 1 / 2 / 3 days, compared with 0.84 / 0.69 / 0.60 for "tomorrow =
today". In a day-by-day replay of 8-20 November 2021, the model warned **3 days before** the simple method on four of
five flood rivers when given the next days' recorded weather (a best case; 0-1 days without it). The warning benefit is
concentrated on rain-driven rivers; slow snowmelt rivers gain little. The model under-predicts record peaks, and I
tested one fix (linear-leaf trees), which improved warnings only modestly.

**Flood mapping.** A U-Net trained on the Sen1Floods11 radar dataset maps flood water with IoU 0.67 +/- 0.01 on held-out
chips, against 0.46 for the best threshold method. Applied to Sentinel-1 scenes from a single orbit, it shows about
19 km2 flooded on Sumas Prairie on 16 Nov 2021, draining to about 3 km2 by late December, following the Nooksack and
Sumas River levels (a descriptive link from 8-9 dates).

**Honest limits.** Weather is one reanalysis point per basin with no direct atmospheric river measure; "future weather"
is recorded, not forecast; the U-Net is trained on global floods and under-maps the Sumas flood; the area-versus-level
link rests on a few dates.

*Figures to attach:* `reports/figures/nov2021_replay_08NL024.png` (forecast replay for the Tulameen River) and
`reports/figures/sumas_flood_maps.png` (before / during / after flood maps).

---

# Part 2. Draft research proposal

**Proposed title:** Atmospheric-River Floods in British Columbia: Probabilistic Machine Learning Forecasting and
Satellite-Based Flood Mapping

## 1. Background
Atmospheric rivers cause many of British Columbia's largest autumn and winter floods, and their impact grows when
warm rain falls on snow and on basins that are already wet. Curry, Islam et al. (2019) showed that atmospheric rivers
increase future flood risk in the Fraser River, and Dr. Islam's studies of runoff and of snowpack and streamflow timing
show that a warming climate changes when and how strongly BC rivers rise. The November 2021 floods, which flooded Sumas
Prairie and damaged communities across southern BC, showed the need for tools that turn this understanding into early,
reliable warnings and fast flood maps.

Machine learning has become a strong tool in hydrology: models trained on many rivers can forecast floods well, even at
poorly gauged sites (Nearing et al., 2024). Satellite radar combined with deep learning can map flood water through
clouds (Bonafilia et al., 2020). Few studies bring the two together for atmospheric-river floods in British Columbia.

## 2. Preliminary work
I built a small, fully reproducible project on the November 2021 floods (code and results on GitHub):
- A LightGBM quantile model for 11 rivers forecasts flow 1-3 days ahead (NSE 0.93 / 0.91 / 0.89, versus 0.84 / 0.69 /
  0.60 for persistence) and warns when a river may pass its 5-year flood level. In a replay of 8-20 November 2021 it
  warned up to 3 days before persistence on four of five rivers, using recorded weather for the coming days.
- Flood frequency analysis (GEV) shows November 2021 was roughly a 1-in-20 to 1-in-30-year event on the Coquihalla,
  Chilliwack, Chehalis and Nooksack, and far rarer on the Coldwater, Tulameen and Similkameen.
- A U-Net trained on Sen1Floods11 maps flood water with IoU 0.67 +/- 0.01 and shows about 19 km2 flooded on Sumas Prairie
  on 16 November 2021.
- Weaknesses found and documented: the tree model under-predicts record peaks; warning benefit depends on river type
  (strong for rain-driven rivers, weak for slow snowmelt rivers); the U-Net under-maps flooded fields.

## 3. Research questions
1. How much do atmospheric river measures (IVT, AR duration and direction) improve 1-5 day flood forecasts for BC
   rivers, compared with rain and temperature alone?
2. Can one deep learning model trained on many basins forecast both rain-driven coastal floods and snowmelt or
   rain-on-snow floods in the interior, with reliable uncertainty ranges, and can it predict record-breaking peaks that
   tree models cannot?
3. How well can satellite radar and deep learning map flood extent in BC's agricultural valleys, and how does flooded
   area relate to river level and forecast flow?
4. Which conditions (AR strength, rain on snow, antecedent wetness) drive the largest forecast floods, and do model
   explanations agree with known physical processes?

## 4. Objectives and methods
- Build a dataset for 30-60 BC basins: ERA5 and ERA5-Land weather and snow, IVT and AR catalogues, river flow and water
  level from HYDAT.
- Develop a multi-basin probabilistic LSTM and compare it with LightGBM, persistence and climatology, with and without
  atmospheric river inputs; test whether sequence models extrapolate better to record floods.
- Turn forecasts into probabilities of passing each river's 2, 5 and 10-year flood levels, and into water levels using
  rating curves; calibrate the forecast ranges (the preliminary ranges cover about 72-75% instead of 80%).
- Test with real archived weather forecasts (for example ECMWF or Environment Canada ensembles), not only recorded
  weather.
- Extend the U-Net flood mapping to several BC flood events, fine-tune on BC examples, and relate flood extent to river
  level.
- Explain forecasts with SHAP and compare with physical understanding.

## 5. How this supports Dr. Islam's research
| Dr. Islam's work | Contribution of this thesis |
|---|---|
| Atmospheric rivers and Fraser River flood risk (GRL, 2019) | Turns long-term AR flood risk findings into short-term, probabilistic flood forecasts that use AR measures directly. |
| Snowpack, runoff timing and variability | Uses snow and temperature inputs and tests rain-on-snow floods specifically. |
| Flood management focus (including the Sumas / Barrowtown area) | Provides flood warnings, forecast water levels and flood extent maps, practical outputs for flood managers. |
| Hydro-climate informatics and ensemble forecasting | Adds deep learning and computer vision methods that can be combined with ensemble and physical approaches. |

## 6. Timeline (two-year MSc)
| Months | Work |
|---|---|
| 1-4 | Courses, literature review, full dataset with ERA5, IVT and AR catalogues |
| 5-9 | Baselines and probabilistic LSTM; AR vs no-AR experiments |
| 10-14 | Real archived forecasts; flood mapping for several events; first conference abstract |
| 15-19 | Flood extent vs river level analysis; explanations; journal paper draft |
| 20-24 | Thesis writing and defence |

## 7. Expected outputs
One journal paper and one conference presentation; open code and data for AR-aware flood forecasting and flood mapping
in BC; a simple dashboard that could be shared with local flood managers.

## 8. My suitability
I bring hands-on experience in data engineering, machine learning and deep learning: predictive models with CatBoost
and XGBoost, fine-tuning large language models, computer vision with Mask R-CNN, and careful GPU experiments and
evaluation. Time-series forecasting, uncertainty estimates, image segmentation and model explanation are within my
skills. I have not yet worked with GIS software or physical hydrological models, and I would build that knowledge
through coursework and the lab's guidance. The project above is my first step into hydro-climate research, and I
deliberately report its weaknesses along with its results.

## 9. References
Curry, C. L., Islam, S. U., et al. (2019). Atmospheric rivers increase future flood risk in Western Canada's largest
Pacific river. *Geophysical Research Letters*. doi:10.1029/2018GL080720
Nearing, G., et al. (2024). Global prediction of extreme floods in ungauged watersheds. *Nature*.
doi:10.1038/s41586-024-07145-1
Bonafilia, D., et al. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms
for Sentinel-1. *CVPR Workshops*.
