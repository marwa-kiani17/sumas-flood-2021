"""Step 4: features, baselines (persistence, climatology) and global LightGBM quantile models.

Three LightGBM variants are trained so we can report the ablations:
  full       : all features + perfect-forecast future weather
  no_storm   : full minus rain-on-snow and AR-like storm features
  no_future  : full minus future weather (only information up to day t)
Splits by issue date: train 1981-2012, valid 2013-2016 (early stopping), test 2017-2023.
Run:  python scripts/04_features_models.py        (needs steps 1-2 finished)
Outputs: data/processed/{features,preds}.parquet, models/*.txt, reports/tables_*.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import yaml

from floodml.evaluate import coverage, kge, nse, peak_errors, pod_far_csi
from floodml.features import LEADS, build_all, feature_columns, split_of
from floodml.models import climatology, persistence, predict_flow, train_quantile_models

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
INTERIM, PROC = ROOT / CFG["paths"]["interim"], ROOT / CFG["paths"]["processed"]
MODELS = PROC / "models"
REP = ROOT / "reports"
for p in (MODELS, REP):
    p.mkdir(parents=True, exist_ok=True)
Q = CFG["model"]["quantiles"]
SEED = CFG["seed"]
np.random.seed(SEED)

flow = pd.read_parquet(INTERIM / "flow_daily.parquet")
wx = pd.read_parquet(INTERIM / "weather_daily.parquet")
stn = pd.read_csv(INTERIM / "stations.csv", dtype={"id": str})
thr = pd.read_csv(PROC / "thresholds.csv", dtype={"id": str}).set_index("id")
# Use only rivers that have weather (weather download may be partial because of API limits).
stn = stn[stn["id"].isin(wx["station"].unique())].reset_index(drop=True)
print("rivers used:", list(stn["id"]))

df = build_all(flow, wx, stn, CFG)
df["split"] = split_of(df["date"], CFG)
df = df[df["split"].notna()].reset_index(drop=True)
df.to_parquet(PROC / "features.parquet", index=False)
print("feature table:", df.shape, df["split"].value_counts().to_dict())

VARIANTS = {"full": dict(future=True, storm=True),
            "no_storm": dict(future=True, storm=False),
            "no_future": dict(future=False, storm=True),
            # linear leaves can extrapolate beyond the training range (record peaks); regularised
            "linear": dict(future=True, storm=True, params=dict(linear_tree=True, linear_lambda=1.0))}
ACTIVE = list(VARIANTS)  # variants that trained successfully
tr, va = df["split"] == "train", df["split"] == "valid"
evalm = df["split"].isin(["valid", "test"])

preds = []
for k in LEADS:
    base = df.loc[evalm, ["station", "date", "split", "meanq"]].copy()
    base["lead"] = k
    base["obs"] = df.loc[evalm, f"obs{k}"].values
    base["persistence"] = persistence(df.loc[evalm], k)
    base["climatology"] = climatology(df.loc[evalm].reset_index(drop=True), flow, CFG["dates"]["train"], k)
    for name, v in VARIANTS.items():
        if name not in ACTIVE:
            continue
        cols = feature_columns(df, k, future=v["future"], storm=v["storm"])
        try:
            models = train_quantile_models(df.loc[tr, cols], df.loc[tr, f"y{k}"], df.loc[va, cols],
                                           df.loc[va, f"y{k}"], Q, SEED, params=v.get("params"))
        except Exception as e:  # e.g. LightGBM may reject linear_tree with some objectives
            print(f"!! variant '{name}' failed ({type(e).__name__}: {e}); skipped")
            ACTIVE.remove(name)
            continue
        P, qs = predict_flow(models, df.loc[evalm, cols], df.loc[evalm, "meanq"].values)
        for j, a in enumerate(qs):
            base[f"{name}_q{int(a * 100)}"] = P[:, j]
            models[a].booster_.save_model(str(MODELS / f"{name}_lead{k}_q{int(a * 100)}.txt"))
        print(f"lead {k} {name}: {len(cols)} features, best iters {[m.best_iteration_ for m in models.values()]}")
    preds.append(base)
preds = pd.concat(preds).reset_index(drop=True)
preds.to_parquet(PROC / "preds.parquet", index=False)

# ---------------- evaluation on the TEST years ----------------
te = preds[preds["split"] == "test"]
rows = []
for (sid, k), g in te.groupby(["station", "lead"]):
    r = dict(station=sid, lead=k)
    for m, col in (("persistence", "persistence"), ("climatology", "climatology"),
                   ("lgbm", "full_q50"), ("lgbm_no_future", "no_future_q50"), ("lgbm_no_storm", "no_storm_q50"),
                   *((("lgbm_linear", "linear_q50"),) if "linear" in ACTIVE else ())):
        ok = g[col].notna()
        r[f"NSE_{m}"], r[f"KGE_{m}"] = nse(g.loc[ok, "obs"], g.loc[ok, col]), kge(g.loc[ok, "obs"], g.loc[ok, col])
    rows.append(r)
skill = pd.DataFrame(rows)
skill.to_csv(REP / "tables_skill_per_river.csv", index=False)
pd.set_option("display.width", 250)
print("\nMEDIAN over rivers, test years:")
print(skill.drop(columns="station").groupby("lead").median().round(3).to_string())

# Warning rule: 0.9 quantile > Q5 (vs observed > Q5). Coverage of the 10-90% range.
rows = []
for variant in ACTIVE:
    for k, g in te.groupby("lead"):
        q5 = g["station"].map(thr["Q5"])
        ev = g["obs"] > q5
        row = dict(variant=variant, lead=k)
        row["POD"], row["FAR"], row["CSI"] = pod_far_csi(ev, g[f"{variant}_q90"] > q5)
        if variant != "full":
            pass
        row["coverage_10_90"] = coverage(g["obs"], g[f"{variant}_q10"], g[f"{variant}_q90"])
        row["n_events_days"] = int(ev.sum())
        rows.append(row)
# persistence as a warning rule: warn if today's flow > Q5
for k, g in te.groupby("lead"):
    q5 = g["station"].map(thr["Q5"])
    ev = g["obs"] > q5
    row = dict(variant="persistence", lead=k, coverage_10_90=np.nan, n_events_days=int(ev.sum()))
    row["POD"], row["FAR"], row["CSI"] = pod_far_csi(ev, g["persistence"] > q5)
    rows.append(row)
warn = pd.DataFrame(rows)
warn.to_csv(REP / "tables_warning.csv", index=False)
print("\nWARNING (daily, Q5 rule) and COVERAGE:\n", warn.round(3).to_string(index=False))

# Peak size / timing error for floods above Q2 (median model, full variant)
rows = []
for variant in [v for v in ("full", "linear") if v in ACTIVE]:
    for (sid, k), g in te.groupby(["station", "lead"]):
        g = g.sort_values("date")
        pe = peak_errors(g["date"].values, g["obs"].values, g[f"{variant}_q50"].values, thr.loc[sid, "Q2"])
        if len(pe):
            pe["station"], pe["lead"], pe["variant"] = sid, k, variant
            rows.append(pe)
if rows:
    pe = pd.concat(rows)
    pe.to_csv(REP / "tables_peak_errors.csv", index=False)
    print("\nPEAK ERRORS (floods > Q2): median |size err| %, median signed size err % (negative = under-predicted), |timing| days")
    print(pe.groupby(["variant", "lead"]).agg(n=("peak_err_pct", "size"),
                                             abs_size_err_pct=("peak_err_pct", lambda s: s.abs().median()),
                                             signed_size_err_pct=("peak_err_pct", "median"),
                                             timing_days=("timing_err_days", lambda s: s.abs().median())).round(2).to_string())
