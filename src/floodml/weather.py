"""Open-Meteo Historical (ERA5 reanalysis) download + per-basin daily weather.

ERA5 = a best-estimate record of past weather made by combining observations with a weather model.
Every response is cached as JSON in data/raw/openmeteo/ so a re-run never repeats a request.
Free tier is rate limited (cost grows with days x variables), so a 429 stops the run cleanly;
just run the script again later and it resumes from the cache.
"""
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
# Trimmed to cut API cost (free tier is billed by days x variables). The first 4 rivers were
# downloaded earlier with two extra columns (temperature_2m_min, soil_moisture_7_to_28cm); unused.
DAILY = ["precipitation_sum", "rain_sum", "snowfall_sum", "temperature_2m_max", "temperature_2m_mean"]
HOURLY = ["snow_depth", "soil_moisture_0_to_7cm"]


class RateLimited(Exception):
    pass


def basin_points(lat, lon, area_km2, n_points):
    """Sample points for one basin. Point 1 = the gauge itself.

    We have no basin polygons, so extra points are a ring around the gauge (N, E, S, W) at
    about half the equivalent basin radius (clipped to 5-40 km). This is an approximation:
    ring points can fall outside the real basin. ERA5 cells are ~25 km, so close points may
    share a cell.
    """
    pts = [(lat, lon)]
    r_km = min(max(0.5 * math.sqrt(area_km2 / math.pi), 5), 40) if area_km2 else 10
    for k in range(n_points - 1):
        b = math.radians(90 * k)
        pts.append((lat + r_km * math.cos(b) / 111.0,
                    lon + r_km * math.sin(b) / (111.0 * math.cos(math.radians(lat)))))
    return pts


def fetch_chunk(lat, lon, start, end, raw_dir, tz, sleep=1.0, tries=8, offline=False):
    """One cached request (daily + hourly variables) for lat/lon between start and end.
    offline=True: only read the cache, never call the API (raises FileNotFoundError if absent)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    f = raw_dir / f"{lat:.3f}_{lon:.3f}_{start}_{end}.json"
    if f.exists():
        return json.loads(f.read_text())
    if offline:
        raise FileNotFoundError(f.name)
    params = dict(latitude=round(lat, 3), longitude=round(lon, 3), start_date=start, end_date=end,
                  daily=",".join(DAILY), hourly=",".join(HOURLY), timezone=tz)
    for i in range(tries):
        time.sleep(sleep)  # be polite
        r = requests.get(URL, params=params, timeout=180)
        if r.status_code == 429:
            if "Minutely" in r.text:  # per-minute limit: just wait a minute and retry
                print("  minutely limit hit, waiting 65 s ...", flush=True)
                time.sleep(65)
                continue
            raise RateLimited(r.text[:200])  # hourly/daily limit: stop, resume later
        if r.ok:
            f.write_text(r.text)
            return r.json()
        time.sleep(2 ** i)
    r.raise_for_status()


def chunk_to_daily(js):
    """Daily table; hourly snow depth / soil moisture are averaged to daily means."""
    d = pd.DataFrame(js["daily"])
    d["time"] = pd.to_datetime(d["time"])
    h = pd.DataFrame(js["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    hd = h.set_index("time").resample("D").mean().reset_index()
    return d.merge(hd, on="time", how="left").rename(columns={"time": "date"})


def get_point_daily(lat, lon, start, end, raw_dir, tz, sleep=1.0, years_per_chunk=15, offline=False):
    """Whole period for one point, in chunks of `years_per_chunk` years."""
    y0, y1 = int(start[:4]), int(end[:4])
    parts = []
    for ys in range(y0, y1 + 1, years_per_chunk):
        ye = min(ys + years_per_chunk - 1, y1)
        s = start if ys == y0 else f"{ys}-01-01"
        e = end if ye == y1 else f"{ye}-12-31"
        parts.append(chunk_to_daily(fetch_chunk(lat, lon, s, e, raw_dir, tz, sleep, offline=offline)))
    return pd.concat(parts).drop_duplicates("date").reset_index(drop=True)
