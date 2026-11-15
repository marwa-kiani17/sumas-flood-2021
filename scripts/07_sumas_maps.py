"""Step 7: Sentinel-1 flood maps of Sumas Prairie.

    python scripts/07_sumas_maps.py --search     # list scenes for the 3 periods (no download)
    python scripts/07_sumas_maps.py --map        # U-Net on every orbit-13 scene -> area table + triptych
Same relative orbit for all scenes (config sumas.rel_orbit) so differences are flood, not viewing angle.
Outputs: data/processed/sumas_area.csv, data/processed/sumas/*.npz, reports/figures/sumas_flood_maps.png
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # torch + numpy OpenMP clash on Windows (see step 6)
os.environ["TORCH_HOME"] = str(ROOT / "data" / "cache" / "torch")
os.environ["HF_HOME"] = str(ROOT / "data" / "cache" / "hf")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from floodml.cv.sentinel1 import describe, read_band_db, search_scenes

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
S = CFG["sumas"]
PROC, FIG = ROOT / CFG["paths"]["processed"], ROOT / CFG["paths"]["figures"]
OUT = PROC / "sumas"

ap = argparse.ArgumentParser()
ap.add_argument("--search", action="store_true")
ap.add_argument("--map", action="store_true")
a = ap.parse_args()

if a.search:
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    for period in ("before", "during", "late"):
        items = search_scenes(S["aoi"], S[period][0], S[period][1])
        print(f"\n=== {period}: {S[period][0]} to {S[period][1]}  ({len(items)} scenes) ===")
        if items:
            print(pd.DataFrame(describe(items)).to_string(index=False))

if a.map:
    import torch

    from floodml.cv.map_flood import pixel_area_km2, predict_prob, to_reference_grid
    from floodml.cv.train_unet import make_unet

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_unet(pretrained=False)
    model.load_state_dict(torch.load(PROC / "unet_resnet18.pt", map_location=device))
    model.to(device).eval()

    items = [i for i in search_scenes(S["aoi"], S["before"][0], S["late"][1])
             if i.properties.get("sat:relative_orbit") == S["rel_orbit"]]
    print(f"{len(items)} scenes in relative orbit {S['rel_orbit']}:", [i.datetime.strftime('%m-%d') for i in items])
    otsu_thr = S["otsu_db"]

    ref = None
    rows, saved = [], {}
    for it in items:
        d = it.datetime.strftime("%Y-%m-%d")
        vv, tf, crs = read_band_db(it, "vv", S["aoi"])
        vh, _, _ = read_band_db(it, "vh", S["aoi"])
        valid = np.isfinite(vv) & np.isfinite(vh)
        water = (predict_prob(model, vv, vh, device) > 0.5) & valid
        otsu = (vv < otsu_thr) & valid
        if ref is None:  # first scene (the 'before' scene) defines the pixel grid
            ref = dict(tf=tf, crs=crs, shape=vv.shape, date=d)
        on = lambda arr: arr if (tf == ref["tf"] and vv.shape == ref["shape"]) else to_reference_grid(
            arr, tf, crs, ref["tf"], ref["crs"], ref["shape"])
        saved[d] = dict(water=on(water.astype("uint8")).astype(bool), otsu=on(otsu.astype("uint8")).astype(bool),
                        valid=on(valid.astype("uint8")).astype(bool), vv=on(np.nan_to_num(vv, nan=-30.0).astype("float32")))
        print(f"{d}: read {vv.shape}, valid {valid.mean():.0%}", flush=True)

    ref_d = ref["date"]
    km2 = pixel_area_km2(ref["tf"])
    # Permanent water = water in the 'before' scene OR dark (< Otsu threshold) in >= 80% of the full-coverage
    # scenes. The U-Net misses parts of the Fraser River in some scenes (wind-roughened surface), which
    # would make the river itself look like new flood; a simple dark-pixel frequency catches it, while the
    # flood pocket (drains within weeks) stays below 80%.
    V = np.stack([s["valid"] for s in saved.values()]).astype("float32")
    full = [d for d, s in saved.items() if s["valid"].mean() > 0.95]
    freq_o = np.mean([saved[d]["otsu"] for d in full], axis=0)
    river = freq_o >= 0.8
    perm = saved[ref_d]["water"] | river
    perm_o = saved[ref_d]["otsu"] | river
    common = V.min(0) > 0  # pixels seen by EVERY scene -> comparable area series
    print(f"permanent water: {perm.sum() * km2:.1f} km2; common area seen by all scenes: {common.sum() * km2:.0f} km2")
    for d, s in saved.items():
        both = s["valid"] & saved[ref_d]["valid"]
        flood = s["water"] & ~perm & both
        flood_o = s["otsu"] & ~perm_o & both
        rows.append(dict(date=d, valid_frac=round(float(s["valid"].mean()), 3),
                         flooded_km2=round(float(flood.sum()) * km2, 2),
                         flooded_km2_otsu=round(float((s["otsu"] & ~perm_o & both).sum()) * km2, 2),
                         flooded_km2_common=round(float((flood & common).sum()) * km2, 2),
                         flooded_km2_common_otsu=round(float((flood_o & common).sum()) * km2, 2),
                         aoi_km2=round(float(both.sum()) * km2, 1)))
        np.savez_compressed(OUT / f"{d}.npz", flood=flood, water=s["water"], valid=s["valid"])
    tab = pd.DataFrame(rows)
    tab.to_csv(PROC / "sumas_area.csv", index=False)
    print(f"\nreference (before) scene = {ref_d}; pixel = {km2 * 1e6:.0f} m2\n", tab.to_string(index=False))

    # ----- triptych: before / during / late -----
    show = [d for d in S["triptych"] if d in saved]
    fig, ax = plt.subplots(2, len(show), figsize=(5 * len(show), 8), squeeze=False)
    for j, d in enumerate(show):
        s = saved[d]
        ax[0, j].imshow(s["vv"], cmap="gray", vmin=-25, vmax=0)
        ax[0, j].set_title(f"{d}  VV radar (dark = smooth surface)", fontsize=9)
        m = np.zeros(s["vv"].shape + (3,), "float32") + 0.85
        m[s["valid"] == 0] = 1.0
        m[perm & s["valid"]] = (0.1, 0.3, 0.9)          # permanent water (blue)
        fl = s["water"] & ~perm & s["valid"] & saved[ref_d]["valid"]
        m[fl] = (0.9, 0.1, 0.1)                         # flooded (red)
        ax[1, j].imshow(m)
        ax[1, j].set_title(f"flooded {tab.set_index('date').loc[d, 'flooded_km2']:.1f} km2 (red); blue = permanent water", fontsize=9)
        for r in (0, 1):
            ax[r, j].axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "sumas_flood_maps.png", dpi=200)
    print("saved", FIG / "sumas_flood_maps.png")
