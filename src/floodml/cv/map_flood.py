"""Apply the trained U-Net to big Sentinel-1 scenes and measure flooded area.

A scene crop is ~2000x1900 pixels, but the U-Net was trained on 512x512 chips, so we cut the scene
into overlapping 512x512 tiles, predict each, average the overlaps, and stitch the result back.
Flooded area = water now AND NOT water in the 'before' reference scene (that removes rivers, lakes
and ponds that are always there).
"""
import numpy as np
import torch
from rasterio.warp import Resampling, reproject

from floodml.cv.dataset import to_input


@torch.no_grad()
def predict_prob(model, vv, vh, device, tile=512, stride=384, batch=4):
    """Water probability (0-1) for a whole scene by overlapping-tile prediction."""
    model.eval()
    H, W = vv.shape
    x = to_input(np.nan_to_num(vv, nan=-30.0), np.nan_to_num(vh, nan=-30.0))  # (3, H, W)
    Hp = tile + int(np.ceil(max(H - tile, 0) / stride)) * stride
    Wp = tile + int(np.ceil(max(W - tile, 0) / stride)) * stride
    xp = np.pad(x, ((0, 0), (0, Hp - H), (0, Wp - W)), mode="edge")
    prob, cnt = np.zeros((Hp, Wp), "float32"), np.zeros((Hp, Wp), "float32")
    coords = [(r, c) for r in range(0, Hp - tile + 1, stride) for c in range(0, Wp - tile + 1, stride)]
    for i in range(0, len(coords), batch):
        cs = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([xp[:, r:r + tile, c:c + tile] for r, c in cs])).to(device)
        pb = torch.sigmoid(model(xb)).squeeze(1).cpu().numpy()
        for (r, c), p in zip(cs, pb):
            prob[r:r + tile, c:c + tile] += p
            cnt[r:r + tile, c:c + tile] += 1
    return (prob / np.maximum(cnt, 1))[:H, :W]


def to_reference_grid(arr, src_tf, src_crs, ref_tf, ref_crs, shape, resampling=Resampling.nearest):
    """Resample an array onto the reference scene's pixel grid so scenes can be compared pixel by pixel."""
    dst = np.zeros(shape, dtype=arr.dtype)
    reproject(arr, dst, src_transform=src_tf, src_crs=src_crs, dst_transform=ref_tf, dst_crs=ref_crs,
              resampling=resampling)
    return dst


def pixel_area_km2(transform):
    return abs(transform.a * transform.e) / 1e6
