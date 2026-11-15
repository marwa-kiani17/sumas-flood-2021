"""Step 1: gauges. Look up, check, and build the cleaned daily flow table.

Search by name (nothing is guessed):
    python scripts/01_gauges.py --search "COQUIHALLA"
Check candidate IDs (prefix USGS with "usgs:"):
    python scripts/01_gauges.py --check 08MF068 usgs:12213100
Build everything for the stations in config.yaml (station table, cleaned data, hydrographs):
    python scripts/01_gauges.py --build
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from floodml.gauges import (clean_daily, get_daily_eccc, get_daily_usgs, get_station_meta,
                            get_usgs_meta, search_stations)

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
RAW = ROOT / CFG["paths"]["raw"] / "gauges"
INTERIM = ROOT / CFG["paths"]["interim"]
FIG = ROOT / CFG["paths"]["figures"]


def load_daily(src, code):
    return get_daily_usgs(code, RAW) if src == "usgs" else get_daily_eccc(code, RAW)


def check(sid):
    src, code = sid.split(":") if ":" in sid else ("eccc", sid)
    raw = load_daily(src, code)
    lev = raw.set_index("date")["level_m"].loc["2021-11-01":"2021-12-31"] if "level_m" in raw else None
    lev_txt = f"level {lev.notna().sum()}/61 days, max {lev.max():.2f} m" if lev is not None and lev.notna().any() else "no level"
    d = clean_daily(raw)
    valid = d["flow_m3s"].dropna()
    if valid.empty:
        return dict(id=sid, start=None, end=None, years=0, pct_missing=None, nov2021="no flow data", level_nov_dec2021=lev_txt)
    span = d.loc[valid.index[0]:valid.index[-1]]
    nov = d.loc["2021-11-08":"2021-11-20", "flow_m3s"]
    return dict(id=sid, start=valid.index[0].year, end=valid.index[-1].year,
                years=round(len(valid) / 365.25, 1),
                pct_missing=round(100 * span["flow_m3s"].isna().mean(), 1),
                nov2021=f"{nov.notna().sum()}/13 days, max {nov.max():.0f} m3/s" if nov.notna().any() else "NO",
                level_nov_dec2021=lev_txt)


def build():
    INTERIM.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    start, end = CFG["dates"]["start"], CFG["dates"]["end"]
    frames, table = [], []
    for s in CFG["stations"]:
        sid, src = s["id"], s["src"]
        meta = get_usgs_meta(sid) if src == "usgs" else get_station_meta(sid)
        d = clean_daily(load_daily(src, sid)).loc[start:end]
        d["station"] = sid
        frames.append(d.reset_index().rename(columns={"index": "date"}))
        v = d["flow_m3s"].dropna()
        nov = d.loc["2021-11-08":"2021-11-20", "flow_m3s"]
        table.append(dict(id=sid, name=s["name"], regime=s["regime"], area_km2=meta["area_km2"],
                          lon=meta["lon"], lat=meta["lat"],
                          start=v.index[0].year, end=v.index[-1].year,
                          pct_missing=round(100 * d["flow_m3s"].isna().mean(), 1),
                          mean_flow=round(v.mean(), 2),
                          nov2021_peak_m3s=round(nov.max(), 1) if nov.notna().any() else None,
                          nov2021_ok=bool(nov.notna().all())))
    tab = pd.DataFrame(table)
    tab.to_csv(INTERIM / "stations.csv", index=False)
    pd.concat(frames).to_parquet(INTERIM / "flow_daily.parquet", index=False)
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))
    allf = pd.concat(frames)
    print(f"\nrows={len(allf)}, date range {allf['date'].min().date()} to {allf['date'].max().date()}, "
          f"missing flow {100 * allf['flow_m3s'].isna().mean():.1f}%, interpolated {allf['interpolated'].sum()} days")

    # Hydrograph per river, Nov 2021 highlighted (the lower-left of the panel shows the full record).
    n = len(frames)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.0 * n), sharex=True)
    for ax, f, row in zip(axes, frames, table):
        ax.plot(f["date"], f["flow_m3s"], lw=0.5, color="tab:blue")
        ax.axvspan(pd.Timestamp("2021-11-08"), pd.Timestamp("2021-11-20"), color="red", alpha=0.4)
        ax.set_ylabel(row["id"], rotation=0, ha="right", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title("Daily mean flow (m3/s) per river; red = 8-20 Nov 2021")
    plt.tight_layout()
    plt.savefig(FIG / "hydrographs.png", dpi=200)
    print("saved", FIG / "hydrographs.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", nargs="+")
    ap.add_argument("--check", nargs="+")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    for term in a.search or []:
        print(f"\n=== stations in BC containing '{term}' ===")
        print(search_stations(term).to_string(index=False))
    if a.check:
        rows = []
        for sid in a.check:
            try:
                rows.append(check(sid))
            except Exception as e:  # report and continue
                rows.append(dict(id=sid, nov2021=f"ERROR {e}"))
        print("\n", pd.DataFrame(rows).to_string(index=False))
    if a.build:
        build()
