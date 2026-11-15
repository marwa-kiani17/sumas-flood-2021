"""Step 5: Nov 2021 replay + SHAP.

Replay: forecasts "issued" every day 8-20 Nov 2021 for each river that flooded, shown with the 10-90%
band and the Q2/Q5/Q10 flood lines. Warning rule: the 0.9 quantile forecast exceeds Q5.
"Days of warning gained" = first issue day persistence would warn (today's flow already > Q5) minus
first issue day the model warns (any lead 1-3 days).
SHAP: shows how much each input pushed a forecast up or down (units: log1p(Q / mean Q)).
Run:  python scripts/05_replay_shap.py      (after step 4)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
INTERIM, PROC, FIG = (ROOT / CFG["paths"][k] for k in ("interim", "processed", "figures"))
REP = ROOT / "reports"
MODELS = PROC / "models"
r0, r1 = (pd.Timestamp(x) for x in CFG["dates"]["replay"])
VAR = sys.argv[1] if len(sys.argv) > 1 else "full"   # model variant: full (default) or linear
SUF = "" if VAR == "full" else f"_{VAR}"              # keeps the original figures; variants get a suffix

preds = pd.read_parquet(PROC / "preds.parquet")
flow = pd.read_parquet(INTERIM / "flow_daily.parquet")
thr = pd.read_csv(PROC / "thresholds.csv", dtype={"id": str}).set_index("id")
name = pd.read_csv(INTERIM / "stations.csv", dtype={"id": str}).set_index("id")["name"]
have = set(preds["station"].unique())
nov_ids = [s["id"] for s in CFG["stations"] if s.get("nov2021") and s["id"] in have and s["id"] in thr.index]
print("replay rivers:", nov_ids)

# ---------- replay figures ----------
rows = []
for sid in nov_ids:
    q2, q5, q10 = thr.loc[sid, ["Q2", "Q5", "Q10"]]
    obs = flow[flow["station"] == sid].set_index("date")["flow_m3s"].loc["2021-11-01":"2021-11-25"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, k in zip(axes, (1, 3)):
        g = preds[(preds["station"] == sid) & (preds["lead"] == k)
                  & preds["date"].between(r0, r1)].sort_values("date")
        tdate = g["date"] + pd.Timedelta(days=k)  # the day each forecast is FOR
        ax.plot(obs.index, obs.values, "k-", lw=2, label="observed")
        ax.fill_between(tdate, g[f"{VAR}_q10"], g[f"{VAR}_q90"], alpha=0.3, color="C0", label="model 10-90%")
        ax.plot(tdate, g[f"{VAR}_q50"], "C0-o", ms=3, label="model median")
        ax.plot(tdate, g["persistence"], "C1--", lw=1, label="persistence")
        for v, lab, c in ((q2, "Q2", "gray"), (q5, "Q5", "orange"), (q10, "Q10", "red")):
            ax.axhline(v, color=c, ls=":", lw=1)
            ax.text(obs.index[0], v, f" {lab}", color=c, va="bottom", fontsize=8)
        ax.set_title(f"{k}-day-ahead forecast", fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    axes[0].set_ylabel("flow (m3/s)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Nov 2021 replay: {name[sid]} ({sid})  [model: {VAR}]")
    plt.tight_layout()
    plt.savefig(FIG / f"nov2021_replay_{sid}{SUF}.png", dpi=200)
    plt.close()

    # first issue day a warning fired
    w = preds[(preds["station"] == sid) & preds["date"].between(r0, r1)]
    first = {}
    for variant in (VAR, "no_future"):
        fire = w[w[f"{variant}_q90"] > q5]["date"]
        first[variant] = fire.min()
    fire_p = w[w["persistence"] > q5]["date"]
    first["persistence"] = fire_p.min()
    flood_day = obs.loc["2021-11-08":"2021-11-20"][lambda s: s > q5].index.min()
    row = dict(river=sid, name=name[sid], Q5=q5, obs_peak=obs.loc["2021-11-08":"2021-11-20"].max(),
               first_obs_above_Q5=flood_day, **{f"warn_{VAR}": first[VAR]}, warn_no_future=first["no_future"],
               warn_persistence=first["persistence"])
    for m in (VAR, "no_future"):
        row[f"days_gained_{m}"] = (first["persistence"] - first[m]).days if pd.notna(first[m]) and pd.notna(first["persistence"]) else np.nan
    rows.append(row)
rep = pd.DataFrame(rows)
rep.to_csv(REP / f"tables_nov2021_warning{SUF}.csv", index=False)
pd.set_option("display.width", 250)
print(f"\nFirst day the warning fired (issue date), model = {VAR}:\n", rep.to_string(index=False))
print("peak forecast (lead 1, median) vs observed peak:")
for sid in nov_ids:
    g = preds[(preds["station"] == sid) & (preds["lead"] == 1) & preds["date"].between(r0, r1)]
    print(f"  {sid}: model median peak {g[f'{VAR}_q50'].max():.0f}, q90 peak {g[f'{VAR}_q90'].max():.0f}, "
          f"observed {flow[(flow['station'] == sid)].set_index('date')['flow_m3s'].loc[r0:r1].max():.0f}")

if VAR != "full":  # SHAP's TreeExplainer does not support linear-leaf trees
    print("\nSHAP skipped for non-default variants.")
    sys.exit()

# ---------- SHAP ----------
feats = pd.read_parquet(PROC / "features.parquet")
feats = feats[feats["station"].isin(have)]
test = feats[feats["split"] == "test"]
for k in (1, 3):
    booster = lgb.Booster(model_file=str(MODELS / f"full_lead{k}_q50.txt"))
    cols = booster.feature_name()
    X = test[cols].sample(min(4000, len(test)), random_state=CFG["seed"])
    expl = shap.TreeExplainer(booster)
    sv = expl.shap_values(X)
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.title(f"SHAP summary, median model, lead {k} day(s)")
    plt.tight_layout()
    plt.savefig(FIG / f"shap_summary_lead{k}.png", dpi=200, bbox_inches="tight")
    plt.close()

    if k == 1:  # waterfall for the Nov 2021 peak of up to 3 rivers (issue day = peak day - 1)
        for sid in nov_ids[:3]:
            o = flow[flow["station"] == sid].set_index("date")["flow_m3s"].loc[r0:r1]
            issue = o.idxmax() - pd.Timedelta(days=1)
            row = feats[(feats["station"] == sid) & (feats["date"] == issue)]
            if row.empty:
                continue
            xr = row[cols].iloc[[0]]
            s1 = expl.shap_values(xr)[0]
            ex = shap.Explanation(values=s1, base_values=float(np.ravel(expl.expected_value)[0]),
                                  data=xr.iloc[0].values, feature_names=cols)
            plt.figure()
            shap.plots.waterfall(ex, max_display=12, show=False)
            plt.title(f"Why the model forecast the Nov 2021 peak: {name[sid]}\n(issued {issue.date()}, lead 1 day)", fontsize=9)
            plt.savefig(FIG / f"shap_nov2021_{sid}.png", dpi=200, bbox_inches="tight")
            plt.close()
print("\nsaved figures to", FIG)
