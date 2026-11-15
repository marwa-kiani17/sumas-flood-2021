"""Extra check: warning scores (Q5 rule) per river and per regime, model vs persistence.

Pooled scores can hide where a model helps. Slow snowmelt rivers favour persistence; rain-driven rivers
(the Nov 2021 type) favour a model with weather inputs. Output: reports/tables_warning_by_river.csv
Run:  python scripts/11_warning_by_river.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import yaml

from floodml.evaluate import pod_far_csi

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
PROC, INTERIM, REP = ROOT / CFG["paths"]["processed"], ROOT / CFG["paths"]["interim"], ROOT / "reports"
preds = pd.read_parquet(PROC / "preds.parquet")
thr = pd.read_csv(PROC / "thresholds.csv", dtype={"id": str}).set_index("id")
st = pd.read_csv(INTERIM / "stations.csv", dtype={"id": str}).set_index("id")
te = preds[preds["split"] == "test"]

rows = []
for (sid, k), g in te.groupby(["station", "lead"]):
    q5 = thr.loc[sid, "Q5"]
    ev = g["obs"] > q5
    if ev.sum() == 0:
        continue
    r = dict(station=sid, name=st.loc[sid, "name"], regime=st.loc[sid, "regime"], lead=k, flood_days=int(ev.sum()))
    for m, col in (("model", "full_q90"), ("persistence", "persistence")):
        pod, far, csi = pod_far_csi(ev, g[col] > q5)
        r[f"POD_{m}"], r[f"FAR_{m}"], r[f"CSI_{m}"] = pod, far, csi
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(REP / "tables_warning_by_river.csv", index=False)
pd.set_option("display.width", 220)
print(t[t["lead"] == 3].round(2).to_string(index=False))

# pooled by regime
out = []
for (reg, k), g in te.assign(regime=te["station"].map(st["regime"])).groupby(["regime", "lead"]):
    q5 = g["station"].map(thr["Q5"])
    ev = g["obs"] > q5
    r = dict(regime=reg, lead=k, flood_days=int(ev.sum()))
    for m, col in (("model", "full_q90"), ("persistence", "persistence")):
        r[f"POD_{m}"], r[f"FAR_{m}"], r[f"CSI_{m}"] = pod_far_csi(ev, g[col] > q5)
    out.append(r)
o = pd.DataFrame(out)
o.to_csv(REP / "tables_warning_by_regime.csv", index=False)
print("\nBy regime:\n", o.round(2).to_string(index=False))
