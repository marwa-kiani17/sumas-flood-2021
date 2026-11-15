"""Step 3: flood frequency (GEV) + rating curves.

Outputs: data/processed/thresholds.csv, data/processed/rating_params.csv,
         reports/figures/return_periods.png, reports/figures/rating_curves.png
Flood thresholds are fitted on TRAINING years only (config dates.train) -> no leakage.
Run:  python scripts/03_hydraulics.py
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
from scipy import stats

from floodml.hydraulics import (RETURN_PERIODS, annual_maxima, bootstrap_ci, fit_gev, fit_gumbel,
                                flow_to_level, quantile, rating_curve, return_period)

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
INTERIM, PROC, FIG = (ROOT / CFG["paths"][k] for k in ("interim", "processed", "figures"))
RAW = ROOT / CFG["paths"]["raw"] / "gauges"
PROC.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
tr0, tr1 = CFG["dates"]["train"]

flow = pd.read_parquet(INTERIM / "flow_daily.parquet")
stn = pd.read_csv(INTERIM / "stations.csv", dtype={"id": str}).set_index("id")

rows, fits = [], {}
for sid, g in flow.groupby("station"):
    q = g.set_index("date")["flow_m3s"]
    peaks_tr = annual_maxima(q.loc[tr0:tr1])
    if len(peaks_tr) < 15:  # seasonal gauge (no winter data): use Mar-Oct maxima instead
        alt = annual_maxima(q.loc[tr0:tr1], months=(3, 10))
        if len(alt) > len(peaks_tr):
            print(f"{sid}: full-year rule gave {len(peaks_tr)} yrs, using Mar-Oct maxima ({len(alt)} yrs)")
            peaks_tr = alt
    if len(peaks_tr) < 15:
        print(f"{sid}: only {len(peaks_tr)} complete training years -> skipped")
        continue
    p = fit_gev(peaks_tr)
    gum = fit_gumbel(peaks_tr)
    lo, hi = bootstrap_ci(peaks_tr, seed=CFG["seed"])
    q_gev = quantile(p, RETURN_PERIODS)
    nov = q.loc["2021-11-08":"2021-11-20"].max()
    r = dict(id=sid, name=stn.loc[sid, "name"], n_train_years=len(peaks_tr))
    for T, a, l, h, gq in zip(RETURN_PERIODS, q_gev, lo, hi, stats.gumbel_r.ppf(1 - 1 / np.array(RETURN_PERIODS), *gum)):
        r.update({f"Q{T}": round(a, 1), f"Q{T}_lo": round(l, 1), f"Q{T}_hi": round(h, 1), f"Q{T}_gumbel": round(gq, 1)})
    r.update(nov2021_peak=round(nov, 1), nov2021_return_period=round(return_period(nov, p), 1))
    rows.append(r)
    fits[sid] = (peaks_tr, p, lo, hi, nov)

thr = pd.DataFrame(rows)
thr.to_csv(PROC / "thresholds.csv", index=False)
pd.set_option("display.width", 250)
print(thr[["id", "n_train_years", "Q2", "Q5", "Q10", "Q25", "Q10_gumbel", "nov2021_peak", "nov2021_return_period"]].to_string(index=False))

# Return-period plot: training peaks (empirical, Weibull plotting position), GEV curve, CI, Nov 2021.
n = len(fits)
cols = 3
fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(14, 3.2 * np.ceil(n / cols)))
T = np.logspace(np.log10(1.5), np.log10(100), 60)
for ax, (sid, (pk, p, lo, hi, nov)) in zip(axes.ravel(), fits.items()):
    s = np.sort(pk.values)
    # Weibull plotting position: the k-th largest of n peaks has return period (n+1)/k.
    rank_from_top = len(s) - np.arange(len(s))
    ax.plot((len(s) + 1) / rank_from_top, s, "o", ms=3, label="peaks 1981-2012")
    ax.plot(T, quantile(p, T), "-", label="GEV")
    ax.fill_between(RETURN_PERIODS, lo, hi, color="C0", alpha=0.2, label="90% CI")
    ax.axhline(nov, color="r", ls="--", lw=1, label="Nov 2021")
    ax.set_xscale("log")
    ax.set_title(f"{sid} {stn.loc[sid, 'name']}", fontsize=8)
    ax.tick_params(labelsize=7)
axes.ravel()[0].legend(fontsize=6)
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.supxlabel("return period (years)")
fig.supylabel("annual max flow (m3/s)")
plt.tight_layout()
plt.savefig(FIG / "return_periods.png", dpi=200)
plt.close()

# Rating curves: need level (m) + flow on the same day. Levels are in the cached ECCC CSVs.
# Fitted on 2013-2023 (channel changes make older ratings drift).
rat, fig_rows = [], []
for s in CFG["stations"]:
    if s["src"] != "eccc":
        continue
    raw = pd.read_csv(RAW / f"eccc_{s['id']}.csv", parse_dates=["date"]).set_index("date")
    d = raw.loc["2013-01-01":"2023-12-31", ["flow_m3s", "level_m"]].dropna()
    if len(d) < 300:
        print(f"{s['id']}: only {len(d)} paired level/flow days in 2013-2023, no rating curve")
        continue
    try:
        par, res = rating_curve(d["level_m"], d["flow_m3s"])
    except Exception as e:
        print(f"{s['id']}: rating fit failed ({e})")
        continue
    rat.append(dict(id=s["id"], a=par[0], b=par[1], h0=par[2], n=len(d),
                    rmse_m3s=float(np.sqrt((res ** 2).mean())), min_h=d["level_m"].min(), max_h=d["level_m"].max()))
    fig_rows.append((s["id"], d, par, res))
pd.DataFrame(rat).to_csv(PROC / "rating_params.csv", index=False)
print("\nrating curves:\n", pd.DataFrame(rat).round(3).to_string(index=False))
if fig_rows:
    fig, axes = plt.subplots(2, len(fig_rows), figsize=(3.2 * len(fig_rows), 6), squeeze=False)
    for j, (sid, d, par, res) in enumerate(fig_rows):
        hh = np.linspace(d["level_m"].min(), d["level_m"].max(), 100)
        axes[0, j].plot(d["level_m"], d["flow_m3s"], ".", ms=2)
        axes[0, j].plot(hh, par[0] * np.clip(hh - par[2], 0, None) ** par[1], "r-")
        axes[0, j].set_title(sid, fontsize=8)
        axes[1, j].plot(d["level_m"], res, ".", ms=2)
        axes[1, j].axhline(0, color="k", lw=0.5)
    axes[0, 0].set_ylabel("flow m3/s")
    axes[1, 0].set_ylabel("residual m3/s")
    plt.tight_layout()
    plt.savefig(FIG / "rating_curves.png", dpi=200)
print("saved figures")
