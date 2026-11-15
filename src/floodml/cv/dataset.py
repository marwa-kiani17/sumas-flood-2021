"""Sen1Floods11 hand-labelled chips: download + PyTorch dataset.

Each chip is a 512x512 Sentinel-1 radar image with 2 bands (VV, VH, in dB) and a label image:
-1 = no data (ignored in the loss), 0 = not water, 1 = water.
dB (decibel) = 10*log10(backscatter); calm water reflects the radar away, so it looks DARK (low dB).
Only the hand-labelled part (~430 chips, ~750 MB) is downloaded, never the full 14 GB.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
import torch
from torch.utils.data import Dataset

BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
HL = f"{BASE}/data/flood_events/HandLabeled"
SPLIT = f"{BASE}/splits/flood_handlabeled"
SPLIT_FILES = {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv", "test": "flood_test_data.csv"}


def _fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def split_pairs(raw_dir, split):
    """List of (s1_path, label_path) for a split, reading the official split CSV (cached)."""
    raw_dir = Path(raw_dir)
    csv = raw_dir / "splits" / SPLIT_FILES[split]
    _fetch(f"{SPLIT}/{SPLIT_FILES[split]}", csv)
    pairs = [l.strip().split(",") for l in csv.read_text().splitlines() if l.strip()]
    return [(raw_dir / "S1Hand" / a, raw_dir / "LabelHand" / b) for a, b in pairs]


def download(raw_dir, workers=8):
    """Download all hand-labelled S1 chips + labels listed in the 3 split files (skips existing)."""
    jobs = []
    for split in SPLIT_FILES:
        for s1, lab in split_pairs(raw_dir, split):
            jobs += [(f"{HL}/S1Hand/{s1.name}", s1), (f"{HL}/LabelHand/{lab.name}", lab)]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: _fetch(*j), jobs))
    return {s: split_pairs(raw_dir, s) for s in SPLIT_FILES}


def read_chip(s1_path, label_path):
    """Return VV, VH (dB, float32, NaN -> -30) and label (int8) arrays, each (H, W)."""
    with rasterio.open(s1_path) as f:
        s1 = f.read().astype("float32")
    with rasterio.open(label_path) as f:
        lab = f.read(1).astype("int8")
    s1 = np.nan_to_num(s1, nan=-30.0, posinf=-30.0, neginf=-30.0)
    return s1[0], s1[1], lab


def to_input(vv, vh):
    """3 channels scaled to 0-1: VV and VH clipped to [-30, 5] dB; VV-VH clipped to [-5, 20] dB.
    (VV-VH is mostly positive, so a [-30, 5] clip would saturate it; this is a deliberate choice.)"""
    a = (np.clip(vv, -30, 5) + 30) / 35
    b = (np.clip(vh, -30, 5) + 30) / 35
    c = (np.clip(vv - vh, -5, 20) + 5) / 25
    return np.stack([a, b, c]).astype("float32")


class FloodChips(Dataset):
    """Chips preloaded in RAM (float16) for fast training. augment = random flips + 90-degree turns."""

    def __init__(self, pairs, augment=False):
        self.augment = augment
        self.x, self.y = [], []
        for s1, lab in pairs:
            vv, vh, y = read_chip(s1, lab)
            self.x.append(to_input(vv, vh).astype("float16"))
            self.y.append(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i].astype("float32"), self.y[i]
        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).long()
