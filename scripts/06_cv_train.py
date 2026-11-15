"""Step 6: Sen1Floods11 -> Otsu baseline + U-Net (ResNet-18). Scores IoU/F1 for water on the test chips.

    python scripts/06_cv_train.py --download      # ~750 MB into data/raw/sen1floods11 (once)
    python scripts/06_cv_train.py                 # Otsu baselines + train U-Net + test scores
    python scripts/06_cv_train.py --epochs 3      # quick smoke test
Outputs: data/processed/unet_resnet18.pt, reports/tables_cv.csv, reports/figures/cv_examples.png
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Keep pretrained-weight caches INSIDE the project folder (not in the user's home directory).
# torch and numpy/MKL each ship an OpenMP runtime on Windows; this allows both to load (standard workaround).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] =str(ROOT / "data" / "cache" / "torch")
os.environ["HF_HOME"] = str(ROOT / "data" / "cache" / "hf")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from floodml.cv.dataset import FloodChips, download, read_chip, split_pairs, to_input
from floodml.cv.train_unet import counts, make_unet, otsu_threshold, predict_loader, scores, train

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
CV = CFG["cv"]
RAW = ROOT / CFG["paths"]["raw"] / "sen1floods11"
PROC, FIG, REP = ROOT / CFG["paths"]["processed"], ROOT / CFG["paths"]["figures"], ROOT / "reports"
for p in (PROC, FIG):
    p.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--download", action="store_true")
ap.add_argument("--epochs", type=int, default=CV["epochs"])
ap.add_argument("--seed", type=int, default=None, help="extra seed for an error bar; outputs get a _seedN suffix")
a = ap.parse_args()
SEED = CFG["seed"] if a.seed is None else a.seed
SUF = "" if a.seed is None else f"_seed{a.seed}"   # extra seeds never overwrite the original results

if a.download:
    pairs = download(RAW)
    print({k: len(v) for k, v in pairs.items()}, "chips downloaded to", RAW)
    sys.exit()

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
pairs = {s: split_pairs(RAW, s) for s in ("train", "valid", "test")}
print({k: len(v) for k, v in pairs.items()}, "chips")

# ---------- Otsu baselines on VV (dB): water = VV below threshold ----------
test_raw = [read_chip(*p) for p in pairs["test"]]
train_vv = np.concatenate([read_chip(*p)[0].ravel()[::50] for p in pairs["train"]])  # subsample pixels
global_thr = otsu_threshold(train_vv)
res = {}
for name, thr_fn in (("Otsu per-chip", lambda vv: otsu_threshold(vv.ravel())),
                     ("Otsu global (fit on train)", lambda vv: global_thr)):
    tp = fp = fn = 0
    for vv, vh, y in test_raw:
        c = counts((vv < thr_fn(vv)).astype(np.int8), y)
        tp, fp, fn = tp + c[0], fp + c[1], fn + c[2]
    res[name] = scores(tp, fp, fn)
print(f"global Otsu threshold = {global_thr:.1f} dB")

# ---------- U-Net ----------
dl = lambda ds, shuffle: DataLoader(ds, batch_size=CV["batch_size"], shuffle=shuffle, num_workers=0)
tr_ds, va_ds, te_ds = (FloodChips(pairs[s], augment=(s == "train")) for s in ("train", "valid", "test"))
model = make_unet(pretrained=True)
model, hist = train(model, dl(tr_ds, True), dl(va_ds, False), a.epochs, CV["lr"], device, PROC / f"unet_resnet18{SUF}.pt")
pd.DataFrame(hist).to_csv(REP / f"tables_cv_training{SUF}.csv", index=False)
res["U-Net (ResNet-18)"] = scores(*predict_loader(model, dl(te_ds, False), device))

tab = pd.DataFrame(res).T.round(3)
tab.to_csv(REP / f"tables_cv{SUF}.csv")
print("\nTEST SET (water class):\n", tab.to_string())
if SUF:  # extra seed: tables only, no example figure
    sys.exit()

# ---------- example figure: first 4 test chips with water ----------
idx = [i for i, (_, _, y) in enumerate(test_raw) if (y == 1).mean() > 0.05][:4]
fig, ax = plt.subplots(len(idx), 4, figsize=(11, 2.8 * len(idx)))
model.eval()
for r, i in enumerate(idx):
    vv, vh, y = test_raw[i]
    with torch.no_grad():
        p = (model(torch.from_numpy(to_input(vv, vh))[None].to(device)).squeeze() > 0).cpu().numpy()
    for c, (img, t, cm) in enumerate(((vv, "VV (dB)", "gray"), (np.ma.masked_equal(y, -1), "label", "Blues"),
                                      (vv < global_thr, "Otsu", "Blues"), (p, "U-Net", "Blues"))):
        ax[r, c].imshow(img, cmap=cm)
        ax[r, c].set_title(t, fontsize=8)
        ax[r, c].axis("off")
plt.tight_layout()
plt.savefig(FIG / "cv_examples.png", dpi=200)
print("saved", FIG / "cv_examples.png")
