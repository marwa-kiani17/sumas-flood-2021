"""Step 8: flooded area (Sentinel-1) vs river level/flow on the same day.

Pairs each satellite date with daily-mean values at:
  - Nooksack River at Ferndale, WA (USGS 12213100): flow, the river whose overflow flooded Sumas Prairie
  - Sumas River near Huntingdon (ECCC 08MH029): water level
  - Fraser River at Mission (ECCC 08MH024): water level (tidal river, large basin)
DESCRIPTIVE ONLY: ~9 dates, not a hydraulic model. Daily means are compared with a ~06:20 local radar pass.
Run:  python scripts/08_area_vs_level.py      (after step 7)
Outputs: reports/figures/flooded_area_vs_level.png, flooded_area_timeseries.png, data/processed/area_vs_level.csv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from floodml.gauges import get_daily_eccc

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
INTERIM, PROC, FIG = (ROOT / CFG["paths"][k] for k in ("interim", "processed", "figures"))
RAW = ROOT / CFG["paths"]["raw"] / "gauges"

area = pd.read_csv(PROC / "sumas_area.csv", parse_dates=["date"]).set_index("date")
flow = pd.read_parquet(INTERIM / "flow_daily.parquet")
nook = flow[flow["station"] == "12213100"].set_index("date")["flow_m3s"]
sumas = get_daily_eccc("08MH029", RAW).set_index("date")
fraser = get_daily_eccc("08MH024", RAW).set_index("date")

df = pd.DataFrame({
    "area_common_km2": area["flooded_km2_common"],
    "area_full_km2": area["flooded_km2"].where(area["valid_frac"] > 0.95),  # only scenes covering the whole AOI
    "nooksack_flow_m3s": nook.reindex(area.index),
    "sumas_level_m": sumas["level_m"].reindex(area.index),
    "fraser_mission_level_m": fraser["level_m"].reindex(area.index),
})
df.to_csv(PROC / "area_vs_level.csv")
pd.set_option("display.width", 200)
print(df.round(2).to_string())

# ---------- figure 1: area vs level/flow ----------
panels = [("nooksack_flow_m3s", "Nooksack flow at Ferndale (m3/s)"),
          ("sumas_level_m", "Sumas R near Huntingdon level (m)"),
          ("fraser_mission_level_m", "Fraser R at Mission level (m)")]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (col, lab) in zip(axes, panels):
    ok = df[[col, "area_common_km2"]].dropna()
    ax.scatter(ok[col], ok["area_common_km2"], c="C3", zorder=3)
    for d, r in ok.iterrows():
        ax.annotate(d.strftime("%d %b"), (r[col], r["area_common_km2"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    rho = spearmanr(ok[col], ok["area_common_km2"])[0] if len(ok) > 3 else np.nan
    ax.set_title(f"Spearman rho = {rho:.2f}, n = {len(ok)}", fontsize=9)
    ax.set_xlabel(lab)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("flooded area, common region (km2)")
fig.suptitle("Flooded area vs river level (descriptive, few dates, not a hydraulic model)", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "flooded_area_vs_level.png", dpi=200)
plt.close()

# ---------- figure 2: area over time next to the Nooksack hydrograph ----------
t0, t1 = pd.Timestamp("2021-11-01"), pd.Timestamp("2021-12-31")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
a1.plot(df.index, df["area_common_km2"], "o-", color="C3", label="U-Net flooded area, common region (257 km2)")
a1.plot(df.index, df["area_full_km2"], "s--", color="C1", label="U-Net, whole area (423 km2, full-coverage scenes)")
a1.set_ylabel("flooded area (km2)")
a1.legend(fontsize=8)
a2.plot(nook.loc[t0:t1].index, nook.loc[t0:t1], color="C0")
a2.set_ylabel("Nooksack flow (m3/s)")
for d in df.index:
    a2.axvline(d, color="gray", lw=0.6, ls=":")
for ax in (a1, a2):
    ax.axvspan(pd.Timestamp("2021-11-14"), pd.Timestamp("2021-11-16"), color="red", alpha=0.12)
a1.set_title("Sumas Prairie flooded area (radar) and Nooksack hydrograph, Nov-Dec 2021 (dotted = radar passes, red = peak rain)", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "flooded_area_timeseries.png", dpi=200)
print("saved figures")
