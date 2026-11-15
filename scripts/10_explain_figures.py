"""Step 10: figures and the evaluation matrix for the plain-language docs (docs/*.md).

Everything is drawn from the real saved results (no invented numbers).
Outputs: docs/figures/*.png, reports/evaluation_matrix.md
Run:  python scripts/10_explain_figures.py     (after steps 4-8)
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
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.stats import spearmanr

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
PROC, REP = ROOT / CFG["paths"]["processed"], ROOT / "reports"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
preds = pd.read_parquet(PROC / "preds.parquet")
thr = pd.read_csv(PROC / "thresholds.csv", dtype={"id": str}).set_index("id")
skill = pd.read_csv(REP / "tables_skill_per_river.csv", dtype={"station": str})
warn = pd.read_csv(REP / "tables_warning.csv")
cv = pd.read_csv(REP / "tables_cv.csv", index_col=0)
pe = pd.read_csv(REP / "tables_peak_errors.csv")
te = preds[preds["split"] == "test"]


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", name)


# ---------------- 1. pipeline diagram ----------------
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.axis("off")
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.2)


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#444"))


ax.text(2.7, 5.0, "River forecasting (Part C)", ha="center", fontweight="bold")
ax.text(8.3, 5.0, "Flood mapping (Part A)", ha="center", fontweight="bold")
box(0.2, 3.9, 2.3, 0.8, "River flow\n(gauges)", "#cfe8ff")
box(2.9, 3.9, 2.3, 0.8, "Weather\n(rain, snow, temperature)", "#cfe8ff")
box(6.2, 3.9, 2.3, 0.8, "Radar images\n(Sentinel-1)", "#ffe3b8")
box(8.8, 3.9, 2.0, 0.8, "Labelled floods\n(Sen1Floods11)", "#ffe3b8")
box(1.5, 2.6, 2.7, 0.8, "Clean + build features\n(lags, rain sums, snow)", "#e4f5df")
box(6.9, 2.6, 3.0, 0.8, "Prepare images\n(crop, decibels, 3 channels)", "#fbe9c8")
box(1.5, 1.4, 2.7, 0.8, "LightGBM model\n(low / middle / high forecast)", "#e4f5df")
box(6.9, 1.4, 3.0, 0.8, "U-Net: water / not water\n(compared with a threshold)", "#fbe9c8")
box(1.5, 0.2, 2.7, 0.7, "Forecast + warning\n(Nov 2021 replay)", "#d3efd0")
box(6.9, 0.2, 3.0, 0.7, "Flood maps + flooded area (km2)", "#f7dca8")
box(4.45, 0.2, 2.0, 0.7, "Hydraulics link:\nflood levels, rating curve,\narea vs river level", "#e9dcf5")
for (x, y0, y1) in ((2.0, 3.9, 3.4), (3.7, 3.9, 3.4), (7.35, 3.9, 3.4), (9.8, 3.9, 3.4), (2.85, 2.6, 2.2),
                    (8.4, 2.6, 2.2), (2.85, 1.4, 0.9), (8.4, 1.4, 0.9)):
    arrow(x, y0, x, y1)
arrow(4.2, 0.55, 4.45, 0.55)
arrow(6.9, 0.55, 6.45, 0.55)
savefig("pipeline.png")

# ---------------- 2. NSE idea ----------------
sid = "08MH103"  # Chilliwack River: rain-fed, lively hydrograph
g = te[(te["station"] == sid) & (te["lead"] == 1)].set_index("date").loc["2019-10-01":"2020-02-29"]
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.3, 1]})
a.plot(g.index + pd.Timedelta(days=1), g["obs"], "k-", lw=2, label="observed")
a.plot(g.index + pd.Timedelta(days=1), g["persistence"], "C1--", lw=1.2, label="persistence ('tomorrow = today')")
a.plot(g.index + pd.Timedelta(days=1), g["full_q50"], "C0-", lw=1.2, label="LightGBM")
a.set_title("Chilliwack River, 1-day-ahead forecasts (test years)")
a.set_ylabel("flow (m3/s)")
a.legend(fontsize=8)
med = skill.groupby("lead")[["NSE_persistence", "NSE_climatology", "NSE_lgbm"]].median()
x = np.arange(3)
for i, (c, lab, col) in enumerate((("NSE_persistence", "persistence", "C1"), ("NSE_climatology", "climatology", "gray"),
                                   ("NSE_lgbm", "LightGBM", "C0"))):
    b.bar(x + i * 0.27, med[c], 0.27, label=lab, color=col)
b.set_xticks(x + 0.27)
b.set_xticklabels(["1 day", "2 days", "3 days"])
b.set_ylim(0, 1)
b.set_title("NSE, median of rivers\n(1 = perfect, 0 = as bad as the average)", fontsize=9)
b.legend(fontsize=8)
savefig("nse_idea.png")

# ---------------- 3. warning table (hits / misses / false alarms) ----------------
k = 2
gk = te[te["lead"] == k]
q5 = gk["station"].map(thr["Q5"])
ev, wn = gk["obs"] > q5, gk["full_q90"] > q5
hits, miss, fa, cn = int((ev & wn).sum()), int((ev & ~wn).sum()), int((~ev & wn).sum()), int((~ev & ~wn).sum())
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.axis("off")
ax.set_xlim(0, 7.5)
ax.set_ylim(0, 4.6)
# columns = what really happened (flood / no flood); rows = what the model said (warned / quiet)
cells = [(2.4, 2.3, f"HIT\n{hits}\nwarned, flood came", "#c9e8c4"), (4.9, 2.3, f"FALSE ALARM\n{fa}\nwarned, no flood", "#fde3b0"),
         (2.4, 0.5, f"MISS\n{miss}\nno warning, flood came", "#f5c3c3"), (4.9, 0.5, f"QUIET\n{cn}\nno warning, no flood", "#e6e6e6")]
for x0, y0, t, c in cells:
    ax.add_patch(Rectangle((x0, y0), 2.4, 1.7, fc=c, ec="#555"))
    ax.text(x0 + 1.2, y0 + 0.85, t, ha="center", va="center", fontsize=10)
ax.text(3.6, 4.2, "a flood really happened", ha="center", fontweight="bold", fontsize=9)
ax.text(6.1, 4.2, "no flood", ha="center", fontweight="bold", fontsize=9)
ax.text(0.1, 4.45, "2-day-ahead forecasts, test years, all rivers pooled", fontsize=8)
ax.text(2.2, 3.15, "model\nwarned", ha="right", va="center", fontsize=9)
ax.text(2.2, 1.35, "model\nquiet", ha="right", va="center", fontsize=9)
pod, far, csi = hits / (hits + miss), fa / (hits + fa), hits / (hits + miss + fa)
ax.text(0.1, 0.1, f"POD = hits / (hits + misses) = {pod:.2f}    FAR = false alarms / warnings = {far:.2f}    "
        f"CSI = hits / (hits + misses + false alarms) = {csi:.2f}", fontsize=7.5)
savefig("warning_table.png")

# ---------------- 4. IoU idea (toy grid) ----------------
grid = np.zeros((8, 8, 3)) + 0.92
truth = np.zeros((8, 8), bool)
pred = np.zeros((8, 8), bool)
truth[2:6, 1:5] = True
pred[3:7, 2:6] = True
grid[truth & ~pred] = (0.2, 0.4, 0.9)
grid[pred & ~truth] = (0.95, 0.6, 0.15)
grid[truth & pred] = (0.3, 0.7, 0.3)
inter, union = (truth & pred).sum(), (truth | pred).sum()
f1 = 2 * inter / (truth.sum() + pred.sum())
fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.imshow(grid)
ax.set_xticks(np.arange(-.5, 8))
ax.set_yticks(np.arange(-.5, 8))
ax.grid(color="white", lw=1)
ax.tick_params(labelbottom=False, labelleft=False, length=0)
ax.set_title(f"blue = true water only, orange = predicted only, green = both\nIoU = {inter}/{union} = {inter / union:.2f}   F1 = {f1:.2f}", fontsize=8)
savefig("iou_idea.png")

# ---------------- 5. coverage idea ----------------
g = te[(te["station"] == sid) & (te["lead"] == 1)].set_index("date").loc["2019-10-01":"2020-02-29"]
inside = (g["obs"] >= g["full_q10"]) & (g["obs"] <= g["full_q90"])
fig, ax = plt.subplots(figsize=(10, 3.6))
t = g.index + pd.Timedelta(days=1)
ax.fill_between(t, g["full_q10"], g["full_q90"], color="C0", alpha=0.3, label="forecast range (10-90%)")
ax.plot(t[inside], g["obs"][inside], "k.", label="observed, inside range")
ax.plot(t[~inside], g["obs"][~inside], "rx", label="observed, outside range")
ax.set_title(f"Coverage: {inside.mean():.0%} of these days fall inside the range (a perfect range gives 80%)", fontsize=10)
ax.set_ylabel("flow (m3/s)")
ax.legend(fontsize=8)
savefig("coverage_idea.png")

# ---------------- 6. evaluation matrix (figure + markdown) ----------------
_cvs = [cv] + [pd.read_csv(p, index_col=0) for p in sorted(REP.glob("tables_cv_seed*.csv"))]
unet_iou = pd.Series([c.loc["U-Net (ResNet-18)", "IoU"] for c in _cvs])
unet_f1 = pd.Series([c.loc["U-Net (ResNet-18)", "F1"] for c in _cvs])
w1 = lambda v, l: warn[(warn["variant"] == v) & (warn["lead"] == l)].iloc[0]
medk = skill.groupby("lead").median(numeric_only=True)
full_pe = pe[pe["variant"] == "full"] if "variant" in pe else pe
pk = full_pe.groupby("lead")["peak_err_pct"].median()
ab = pd.read_csv(PROC / "area_vs_level.csv", index_col=0)
rho = spearmanr(ab[["sumas_level_m", "area_common_km2"]].dropna().iloc[:, 0], ab[["sumas_level_m", "area_common_km2"]].dropna().iloc[:, 1])[0]
rows = [
    ("NSE (flow forecast)", "How close forecast flow is to real flow, compared with just using the average", "1 = perfect; 0 = same as the average",
     f"{medk.loc[1, 'NSE_lgbm']:.2f} / {medk.loc[2, 'NSE_lgbm']:.2f} / {medk.loc[3, 'NSE_lgbm']:.2f}  (persistence {medk.loc[1, 'NSE_persistence']:.2f} / {medk.loc[2, 'NSE_persistence']:.2f} / {medk.loc[3, 'NSE_persistence']:.2f})", "1 / 2 / 3 days"),
    ("KGE (flow forecast)", "Like NSE but also checks bias and spread", "1 = perfect",
     f"{medk.loc[1, 'KGE_lgbm']:.2f} / {medk.loc[2, 'KGE_lgbm']:.2f} / {medk.loc[3, 'KGE_lgbm']:.2f}", "1 / 2 / 3 days"),
    ("POD (warning)", "Share of real floods (above the 5-year level) that were warned", "1 = none missed",
     f"{w1('full', 1)['POD']:.2f} / {w1('full', 2)['POD']:.2f} / {w1('full', 3)['POD']:.2f}  (persistence {w1('persistence', 1)['POD']:.2f} / {w1('persistence', 2)['POD']:.2f} / {w1('persistence', 3)['POD']:.2f})", "1 / 2 / 3 days; pooled over rivers (snowmelt-dominated), see by-regime table in results.md"),
    ("FAR (warning)", "Share of warnings that were false alarms", "0 = no false alarms",
     f"{w1('full', 1)['FAR']:.2f} / {w1('full', 2)['FAR']:.2f} / {w1('full', 3)['FAR']:.2f}", "1 / 2 / 3 days"),
    ("CSI (warning)", "One combined warning score (hits, misses, false alarms)", "1 = perfect",
     f"{w1('full', 1)['CSI']:.2f} / {w1('full', 2)['CSI']:.2f} / {w1('full', 3)['CSI']:.2f}", "1 / 2 / 3 days"),
    ("Coverage 10-90%", "How often reality falls inside the forecast range", "about 0.80 expected",
     f"{w1('full', 1)['coverage_10_90']:.2f} / {w1('full', 2)['coverage_10_90']:.2f} / {w1('full', 3)['coverage_10_90']:.2f}", "1 / 2 / 3 days"),
    ("Peak error", "How far off the forecast flood peak is (negative = too low)", "0 = perfect",
     f"{pk.loc[1]:.0f}% / {pk.loc[2]:.0f}% / {pk.loc[3]:.0f}% (median)", "1 / 2 / 3 days"),
    ("IoU (water map)", "Overlap of mapped water with true water divided by their union", "1 = perfect",
     f"U-Net {unet_iou.mean():.2f} +/- {unet_iou.std():.2f} ({len(unet_iou)} seeds), threshold {cv.loc['Otsu global (fit on train)', 'IoU']:.2f}", "Sen1Floods11 test chips"),
    ("F1 (water map)", "Balance of precision and recall for water pixels", "1 = perfect",
     f"U-Net {unet_f1.mean():.2f} +/- {unet_f1.std():.2f}, threshold {cv.loc['Otsu global (fit on train)', 'F1']:.2f}", "Sen1Floods11 test chips"),
    ("Spearman (area vs level)", "Do bigger floods go with higher river level? (rank correlation)", "1 = always together",
     f"{rho:.2f} with Sumas level", "only 8 dates: descriptive"),
]
em = pd.DataFrame(rows, columns=["Score", "What it tells you", "Good value", "Our result", "Notes"])
lines = ["# Evaluation matrix\n", "One table with every score used in this project. Results are on data the model never saw.\n",
         "| " + " | ".join(em.columns) + " |", "|" + "---|" * len(em.columns)]
lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in em.values]
(REP / "evaluation_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

fig, ax = plt.subplots(figsize=(15, 5.2))
ax.axis("off")
tb = ax.table(cellText=em[["Score", "What it tells you", "Good value", "Our result"]].values,
              colLabels=["Score", "What it tells you", "Good value", "Our result"], loc="center", cellLoc="left",
              colWidths=[0.12, 0.34, 0.20, 0.34])
tb.auto_set_font_size(False)
tb.set_fontsize(8)
tb.scale(1, 1.9)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#dde8f5")
        cell.set_text_props(fontweight="bold")
savefig("evaluation_matrix.png")
print("saved reports/evaluation_matrix.md")
