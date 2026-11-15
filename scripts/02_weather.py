"""Step 2: Open-Meteo (ERA5) weather per basin -> data/interim/weather_daily.parquet.

Run:  python scripts/02_weather.py
Resumable: responses are cached, so if the free-tier rate limit (HTTP 429) stops the run,
just run it again later.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import yaml

from floodml.weather import RateLimited, basin_points, get_point_daily

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
OM = CFG["openmeteo"]
RAW = ROOT / CFG["paths"]["raw"] / "openmeteo"
INTERIM = ROOT / CFG["paths"]["interim"]
start, end = CFG["dates"]["start"], CFG["dates"]["end"]

import time

st = pd.read_csv(INTERIM / "stations.csv", dtype={"id": str})
_rank = {i: n for n, i in enumerate(OM.get("priority", []))}
st = st.assign(_p=st["id"].map(_rank).fillna(99)).sort_values("_p", kind="stable").drop(columns="_p").reset_index(drop=True)
WAIT = "--wait" in sys.argv  # on the hourly limit, sleep ~1 h and continue by itself
OFFLINE = "--cache-only" in sys.argv  # build the parquet from cached rivers only, no API calls


def run_once():
    out, done = [], []
    try:
        for _, s in st.iterrows():
            pts = basin_points(s["lat"], s["lon"], s["area_km2"], OM["points_per_basin"])
            try:
                dfs = [get_point_daily(la, lo, start, end, RAW, OM["timezone"], OM["sleep_seconds"], offline=OFFLINE)
                       for la, lo in pts]
            except FileNotFoundError:  # only in --cache-only mode: this river is not downloaded yet
                print(f"{s['id']}: not in cache, skipped", flush=True)
                continue
            # Basin weather = mean over the points (point 0 is the gauge).
            basin = pd.concat(dfs).groupby("date").mean(numeric_only=True).reset_index()
            basin["station"] = s["id"]
            out.append(basin)
            done.append(s["id"])
            print(f"{s['id']}: {len(pts)} point(s), {len(basin)} days OK", flush=True)
    except RateLimited as e:
        print(f"\nRATE LIMITED by Open-Meteo ({e}). Finished: {done}.", flush=True)
    return out, done


out, done = run_once()
while WAIT and not OFFLINE and len(out) < len(st):
    print("Waiting 61 minutes for the hourly limit to reset ...", flush=True)
    time.sleep(61 * 60)
    out, done = run_once()

if len(out) == len(st) or (OFFLINE and out):
    w = pd.concat(out)
    w.to_parquet(INTERIM / "weather_daily.parquet", index=False)
    print(f"\nrivers with weather: {sorted(w['station'].unique())}"
          f"\nrows={len(w)}, {w['date'].min().date()} to {w['date'].max().date()}")
    print("missing % per column:\n", (100 * w.isna().mean()).round(1).to_string())
else:
    print(f"{len(out)}/{len(st)} rivers done; weather_daily.parquet is written only when all are done.")
