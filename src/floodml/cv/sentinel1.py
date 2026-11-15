"""Sentinel-1 radar scenes for Sumas Prairie from Microsoft Planetary Computer (free, anonymous).

Collection `sentinel-1-rtc` = radiometrically terrain-corrected backscatter (VV, VH), ~10 m pixels.
Scenes are read straight from the cloud, cropped to the area of interest (AOI), and converted to dB:
dB = 10*log10(backscatter). Calm flood water looks dark (low dB).
"""
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def search_scenes(aoi, start, end, collection="sentinel-1-rtc"):
    """All scenes touching the AOI between start and end (ISO dates). Returns list of pystac Items."""
    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    bbox = [aoi["lon_min"], aoi["lat_min"], aoi["lon_max"], aoi["lat_max"]]
    items = list(cat.search(collections=[collection], bbox=bbox, datetime=f"{start}/{end}").items())
    return sorted(items, key=lambda i: i.datetime)


def describe(items):
    """One line per scene: id, date, orbit direction, relative orbit, polarizations."""
    rows = []
    for i in items:
        p = i.properties
        rows.append(dict(id=i.id, date=i.datetime.strftime("%Y-%m-%d %H:%M"), orbit=p.get("sat:orbit_state"),
                         rel_orbit=p.get("sat:relative_orbit"), pol=p.get("sar:polarizations")))
    return rows


def read_band_db(item, band, aoi):
    """Crop one band (vv or vh) to the AOI and return dB values (float32) with NaN for no data,
    plus the rasterio transform and CRS of the crop."""
    href = planetary_computer.sign(item).assets[band].href
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, aoi["lon_min"], aoi["lat_min"],
                                                    aoi["lon_max"], aoi["lat_max"])
        win = from_bounds(left, bottom, right, top, src.transform)
        # keep only the part of the AOI the scene actually covers (swath edges)
        win = win.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
        arr = src.read(1, window=win).astype("float32")
        transform = src.window_transform(win)
        nodata = src.nodata
        crs = src.crs
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr <= 0] = np.nan
    return 10 * np.log10(arr), transform, crs
