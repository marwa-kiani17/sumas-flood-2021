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

st = pd.read_csv(INTERIM / "stations.csv", dtype={"id": str})
out, done = [], []
try:
    for _, s in st.iterrows():
        pts = basin_points(s["lat"], s["lon"], s["area_km2"], OM["points_per_basin"])
        dfs = [get_point_daily(la, lo, start, end, RAW, OM["timezone"], OM["sleep_seconds"]) for la, lo in pts]
        # Basin weather = mean over the points. Point 0 (the gauge) is kept separately for the
        # "mid-basin" AR-storm rule later.
        basin = pd.concat(dfs).groupby("date").mean(numeric_only=True).reset_index()
        basin["station"] = s["id"]
        out.append(basin)
        done.append(s["id"])
        print(f"{s['id']}: {len(pts)} point(s), {len(basin)} days OK")
except RateLimited as e:
    print(f"\nRATE LIMITED by Open-Meteo ({e}). Finished: {done}. Run again later to resume.")

if len(out) == len(st):
    w = pd.concat(out)
    w.to_parquet(INTERIM / "weather_daily.parquet", index=False)
    print(f"\nrows={len(w)}, {w['date'].min().date()} to {w['date'].max().date()}")
    print("missing % per column:\n", (100 * w.isna().mean()).round(1).to_string())
else:
    print(f"{len(out)}/{len(st)} rivers done; weather_daily.parquet is written only when all are done.")
