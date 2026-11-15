"""River gauge download helpers (ECCC GeoMet for Canada, USGS NWIS for the Nooksack).

Hydrology terms:
- Discharge (flow): water volume passing a gauge per second, m3/s.
- Stage / level: height of the water surface at the gauge, metres.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

GEOMET = "https://api.weather.gc.ca/collections"
USGS = "https://waterservices.usgs.gov/nwis"


def _get(url, params=None, tries=4):
    """GET with retry + exponential backoff."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def search_stations(name_part, prov="BC"):
    """Find ECCC stations whose name contains `name_part` (case-insensitive).

    GeoMet has no name filter, so we page through all stations of the province.
    Returns a DataFrame: id, name, status, area_km2, lon, lat.
    """
    rows, offset = [], 0
    while True:
        r = _get(f"{GEOMET}/hydrometric-stations/items",
                 {"PROV_TERR_STATE_LOC": prov, "limit": 500, "offset": offset, "f": "json"}).json()
        feats = r["features"]
        for f in feats:
            p = f["properties"]
            lon, lat = f["geometry"]["coordinates"][:2] if f.get("geometry") else (None, None)
            rows.append(dict(id=p["STATION_NUMBER"], name=p["STATION_NAME"], status=p["STATUS_EN"],
                             area_km2=p["DRAINAGE_AREA_GROSS"], lon=lon, lat=lat))
        if len(feats) < 500:
            break
        offset += 500
    df = pd.DataFrame(rows)
    return df[df["name"].str.contains(name_part, case=False, regex=False)].reset_index(drop=True)


def get_station_meta(station_id):
    """Name, drainage area (km2), lon, lat of one ECCC station (from hydrometric-stations)."""
    r = _get(f"{GEOMET}/hydrometric-stations/items",
             {"STATION_NUMBER": station_id, "f": "json"}).json()["features"][0]
    p = r["properties"]
    return dict(name=p["STATION_NAME"], area_km2=p["DRAINAGE_AREA_GROSS"],
                lon=r["geometry"]["coordinates"][0], lat=r["geometry"]["coordinates"][1])


def get_usgs_meta(site):
    """Name, lon, lat of a USGS site (area is read from the site file when available, km2)."""
    r = _get(f"{USGS}/site/", {"format": "rdb", "sites": site, "siteOutput": "expanded"})
    lines = [l for l in r.text.splitlines() if l and not l.startswith("#")]
    hdr, row = lines[0].split("\t"), lines[2].split("\t")
    d = dict(zip(hdr, row))
    area_sqmi = pd.to_numeric(d.get("drain_area_va"), errors="coerce")
    return dict(name=d["station_nm"], lon=float(d["dec_long_va"]), lat=float(d["dec_lat_va"]),
                area_km2=area_sqmi * 2.58999 if pd.notna(area_sqmi) else None)


def get_daily_eccc(station_id, raw_dir):
    """Daily mean flow + level for one ECCC station. Cached as CSV in raw_dir."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"eccc_{station_id}.csv"
    if out.exists():
        return pd.read_csv(out, parse_dates=["date"])
    rows, offset = [], 0
    while True:
        r = _get(f"{GEOMET}/hydrometric-daily-mean/items",
                 {"STATION_NUMBER": station_id, "limit": 10000, "offset": offset, "f": "json"}).json()
        feats = r["features"]
        rows += [dict(date=f["properties"]["DATE"], flow_m3s=f["properties"]["DISCHARGE"],
                      level_m=f["properties"]["LEVEL"],
                      flow_flag=f["properties"]["DISCHARGE_SYMBOL_EN"]) for f in feats]
        if len(feats) < 10000:
            break
        offset += 10000
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(out, index=False)
    return df


def get_daily_usgs(site, raw_dir, start="1980-01-01", end="2023-12-31"):
    """Daily mean discharge for a USGS site, converted from ft3/s to m3/s."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"usgs_{site}.csv"
    if out.exists():
        return pd.read_csv(out, parse_dates=["date"])
    r = _get(f"{USGS}/dv/", {"format": "json", "sites": site, "parameterCd": "00060",
                             "startDT": start, "endDT": end}).json()
    series = r["value"]["timeSeries"]
    if not series:
        raise ValueError(f"USGS site {site}: no discharge data returned")
    vals = series[0]["values"][0]["value"]
    df = pd.DataFrame(vals)
    df["date"] = pd.to_datetime(df["dateTime"])
    df["flow_m3s"] = pd.to_numeric(df["value"], errors="coerce").where(lambda s: s >= 0) * 0.0283168
    df = df.rename(columns={"qualifiers": "flow_flag"})[["date", "flow_m3s", "flow_flag"]]
    df["flow_flag"] = df["flow_flag"].astype(str)
    df.to_csv(out, index=False)
    return df


def clean_daily(df, max_gap=3):
    """Put data on a full daily index; interpolate gaps of <= max_gap days and flag them.

    Longer gaps stay NaN (those rows are dropped from training later).
    """
    s = df.set_index("date")["flow_m3s"]
    s = s[~s.index.duplicated()].asfreq("D")
    missing = s.isna()
    # Label each run of NaNs with its length, only fill short runs.
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    short = missing & (run_len <= max_gap)
    filled = s.interpolate(limit_area="inside")
    out = pd.DataFrame({"flow_m3s": s.where(~short, filled), "interpolated": short})
    return out
