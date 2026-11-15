"""Otsu baseline, U-Net training and water-class scores (IoU, F1).

U-Net: a network that labels every pixel ("water" / "not water"). Here the encoder is a ResNet-18
pretrained on ImageNet photos, fine-tuned on radar chips.
IoU (intersection over union) = overlap / union of predicted and true water; 1 = perfect.
Pixels labelled -1 (no data) are ignored everywhere.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F


# ---------------- Otsu ----------------
def otsu_threshold(values, bins=256):
    """Threshold that best separates two groups (dark water / brighter land) in a histogram."""
    hist, edges = np.histogram(values, bins=bins)
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0, w1 = np.cumsum(p), 1 - np.cumsum(p)
    m0 = np.cumsum(p * centers) / np.where(w0 == 0, 1, w0)
    mt = (p * centers).sum()
    m1 = (mt - np.cumsum(p * centers)) / np.where(w1 == 0, 1, w1)
    between = w0 * w1 * (m0 - m1) ** 2
    return centers[np.argmax(between)]


# ---------------- scores ----------------
def counts(pred, y):
    """TP, FP, FN for the water class, ignoring y == -1."""
    v = y >= 0
    p, t = pred[v] == 1, y[v] == 1
    return int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())


def scores(tp, fp, fn):
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return dict(IoU=iou, F1=f1, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))


# ---------------- loss ----------------
def bce_dice_loss(logits, y):
    """BCE + Dice on valid pixels only (y >= 0)."""
    valid = (y >= 0).float()
    t = (y == 1).float()
    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * valid).sum() + 1)
    return bce + dice


# ---------------- model ----------------
def make_unet(pretrained=True):
    import segmentation_models_pytorch as smp
    try:
        return smp.Unet("resnet18", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)
    except Exception as e:  # weights could not be downloaded
        print(f"!! could not load ImageNet weights ({e}); training from scratch")
        return smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)


@torch.no_grad()
def predict_loader(model, loader, device):
    """Returns total (tp, fp, fn) over a loader."""
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        pred = (model(x.to(device)).squeeze(1) > 0).cpu().numpy().astype(np.int8)
        a, b, c = counts(pred, y.numpy())
        tp, fp, fn = tp + a, fp + b, fn + c
    return tp, fp, fn


def train(model, train_loader, valid_loader, epochs, lr, device, ckpt_path):
    """AdamW + cosine schedule, mixed precision; keeps the checkpoint with the best valid IoU."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, hist = -1, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = bce_dice_loss(model(x).squeeze(1).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        s = scores(*predict_loader(model, valid_loader, device))
        hist.append(dict(epoch=ep, loss=tot / len(train_loader), **s))
        if s["IoU"] > best:
            best = s["IoU"]
            torch.save(model.state_dict(), ckpt_path)
        print(f"epoch {ep:2d}/{epochs}  loss {tot / len(train_loader):.3f}  valid IoU {s['IoU']:.3f} "
              f"F1 {s['F1']:.3f}  ({time.time() - t0:.0f}s){'  *best' if s['IoU'] == best else ''}", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, hist
